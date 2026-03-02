"""
eval_smolvlm.py — SmolVLM2-2.2B-Agentic-GUI ScreenSpot Evaluation

Evaluates ahmadw/SmolVLM2-2.2B-Agentic-GUI-GGUF via a running llama-server instance.
No transformers required — just llama.cpp + the openai Python package.

── Setup (on the VM) ────────────────────────────────────────────────────────

1. Build llama.cpp with CUDA support (one-time):
       git clone https://github.com/ggml-org/llama.cpp
       cd llama.cpp
       cmake -B build -DGGML_CUDA=ON
       cmake --build build --config Release -j$(nproc)

2. Download the model (Q4_K_M = 1.0 GB + 832 MB mmproj):
       huggingface-cli download ahmadw/SmolVLM2-2.2B-Agentic-GUI-GGUF \\
         --local-dir ~/smolvlm-gui

3. Start the server (keep this running in a separate tmux pane):
       ~/llama.cpp/build/bin/llama-server \\
         -m ~/smolvlm-gui/SmolVLM2-2.2B-Instruct-Agentic-GUI-Q4_K_M.gguf \\
         --mmproj ~/smolvlm-gui/SmolVLM2-2.2B-Instruct-Agentic-GUI-mmproj-f16.gguf \\
         -c 4096 -ngl 99 --port 8888 --chat-template smolvlm

4. Install the openai client if not present:
       pip install openai

── Usage ─────────────────────────────────────────────────────────────────────

    python eval_smolvlm.py --max-samples 20 --output-file debug_smol.json
    python eval_smolvlm.py --max-samples 100 --output-file results_smol.json
    python eval_smolvlm.py --server-url http://localhost:8888
"""

import argparse
import base64
import io
import json
import re
from collections import defaultdict

from datasets import load_dataset
from openai import OpenAI

# ── Prompt ────────────────────────────────────────────────────────────────────
# SmolVLM2-Agentic-GUI was trained on AGUVIS-style data.
# The model outputs actions like: click(x, y)  — coordinates in [0, 1].

SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the current screen. "
    "Output only the single action needed to complete the task. "
    "Use normalized [0, 1] coordinates where (0, 0) is the top-left corner.\n\n"
    "Action format:\n"
    "  click(x, y)                          — click at position\n"
    "  type(text)                            — type text\n"
    "  scroll(x, y, direction)              — scroll up/down/left/right\n"
    "  drag(x1, y1, x2, y2)                 — drag from one point to another\n"
    "  key(key_name)                        — press a keyboard key\n\n"
    "Output only the action, nothing else."
)


# ── Image encoding ────────────────────────────────────────────────────────────

def image_to_data_url(image):
    """Convert a PIL Image to a base64 data URL for the vision API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ── Coordinate parsing ────────────────────────────────────────────────────────

def parse_click_coords(text):
    """
    Extract click(x, y) coordinates from the model output.
    SmolVLM2-Agentic outputs [0, 1] normalized coordinates.

    Returns (x, y) as floats in [0, 1], or None if no click found.
    """
    # Primary: click(x, y)
    m = re.search(r"click\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Fallback: any two numbers in parentheses after 'click'
    m = re.search(r"click[^(]*\(\s*([0-9]*\.?[0-9]+)[,\s]+([0-9]*\.?[0-9]+)\s*\)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    return None


# ── Hit detection ─────────────────────────────────────────────────────────────

def is_hit(pred_x, pred_y, bbox):
    """
    Both SmolVLM2 output and ScreenSpot bboxes are in [0, 1] — no conversion needed.
    """
    x_min, y_min, x_max, y_max = bbox
    return x_min <= pred_x <= x_max and y_min <= pred_y <= y_max


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(client, model_name, image, instruction):
    data_url = image_to_data_url(image)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": f"Task: {instruction}"},
                ],
            },
        ],
        max_tokens=64,      # click(x, y) is short — no need for more
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    client = OpenAI(base_url=f"{args.server_url}/v1", api_key="local")

    # Probe the server to get the loaded model name (used as the model= parameter)
    try:
        models = client.models.list()
        model_name = models.data[0].id
        print(f"Connected to llama-server. Model: {model_name}")
    except Exception as e:
        print(f"ERROR: Could not connect to llama-server at {args.server_url}: {e}")
        print("Make sure the server is running (see setup instructions in the docstring).")
        return

    print(f"Loading ScreenSpot (split='{args.split}')...")
    dataset = load_dataset("rootsautomation/ScreenSpot", split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    total = len(dataset)
    print(f"Evaluating on {total} examples...\n")

    hits           = 0
    parse_failures = 0
    by_platform    = defaultdict(lambda: {"hits": 0, "total": 0})
    by_elem_type   = defaultdict(lambda: {"hits": 0, "total": 0})
    results        = []

    for i, example in enumerate(dataset):
        image       = example["image"].convert("RGB")
        instruction = example["instruction"]
        bbox        = example["bbox"]
        platform    = example.get("platform", "unknown")
        elem_type   = example.get("element_type", "unknown")

        raw_output = run_inference(client, model_name, image, instruction)
        coords     = parse_click_coords(raw_output)

        hit        = False
        pred_coords = None

        if coords is None:
            parse_failures += 1
        else:
            pred_x, pred_y = coords
            pred_coords = (pred_x, pred_y)
            hit = is_hit(pred_x, pred_y, bbox)

        if hit:
            hits += 1

        by_platform[platform]["total"]   += 1
        by_elem_type[elem_type]["total"] += 1
        if hit:
            by_platform[platform]["hits"]   += 1
            by_elem_type[elem_type]["hits"] += 1

        results.append({
            "index":        i,
            "instruction":  instruction,
            "platform":     platform,
            "element_type": elem_type,
            "bbox":         bbox,
            "pred_coords":  pred_coords,
            "raw_output":   raw_output,
            "hit":          hit,
        })

        done = i + 1
        print(
            f"[{done:>{len(str(total))}}/{total}]  "
            f"hit={hit}  running_acc={hits/done*100:.1f}%  "
            f"parse_fail={parse_failures}",
            end="\r",
        )

    print()

    click_acc       = hits / total * 100
    parse_fail_rate = parse_failures / total * 100

    print("\n" + "=" * 55)
    print("  SmolVLM2-2.2B-Agentic-GUI SCREENSPOT RESULTS")
    print("=" * 55)
    print(f"  Total examples   : {total}")
    print(f"  Click Accuracy   : {hits}/{total}  ({click_acc:.1f}%)")
    print(f"  Parse Failures   : {parse_failures}/{total}  ({parse_fail_rate:.1f}%)")

    print("\n  --- By Platform ---")
    for platform, counts in sorted(by_platform.items()):
        acc = counts["hits"] / counts["total"] * 100 if counts["total"] else 0
        print(f"  {platform:<12}: {counts['hits']}/{counts['total']}  ({acc:.1f}%)")

    print("\n  --- By Element Type ---")
    for elem_type, counts in sorted(by_elem_type.items()):
        acc = counts["hits"] / counts["total"] * 100 if counts["total"] else 0
        print(f"  {elem_type:<12}: {counts['hits']}/{counts['total']}  ({acc:.1f}%)")

    print("=" * 55 + "\n")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(
                {
                    "model":   model_name,
                    "summary": {
                        "total":              total,
                        "hits":               hits,
                        "click_accuracy":     click_acc,
                        "parse_failures":     parse_failures,
                        "parse_failure_rate": parse_fail_rate,
                        "by_platform":        dict(by_platform),
                        "by_element_type":    dict(by_elem_type),
                    },
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Full results saved to {args.output_file}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolVLM2-Agentic-GUI ScreenSpot Evaluation")
    parser.add_argument("--server-url",   type=str, default="http://localhost:8888",
                        help="llama-server base URL (default: http://localhost:8888).")
    parser.add_argument("--split",        type=str, default="test",
                        help="Dataset split (default: test).")
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="Limit to N examples for a quick smoke test.")
    parser.add_argument("--output-file",  type=str, default=None,
                        help="Path to write full per-example JSON results.")
    evaluate(parser.parse_args())
