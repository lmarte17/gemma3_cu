"""
eval_smolvlm.py — SmolVLM2-2.2B-Agentic-GUI ScreenSpot Evaluation

Evaluates ahmadw/SmolVLM2-2.2B-Agentic-GUI-GGUF via a running llama-server instance.
Uses the native llama.cpp /completion endpoint (not the OpenAI-compatible endpoint,
which has template tokenization issues with SmolVLM2 vision content).

── Setup (on the VM) ────────────────────────────────────────────────────────

1. Build llama.cpp with CUDA support (one-time):
       git clone https://github.com/ggml-org/llama.cpp
       cd llama.cpp
       cmake -B build -DGGML_CUDA=ON
       cmake --build build --config Release -j$(nproc)

2. Download the model (Q4_K_M = 1.0 GB + 832 MB mmproj):
       huggingface-cli download ahmadw/SmolVLM2-2.2B-Agentic-GUI-GGUF \\
         --local-dir ~/smolvlm-gui

3. Start the server (keep running in a separate tmux pane):
       ~/llama.cpp/build/bin/llama-server \\
         -m ~/smolvlm-gui/SmolVLM2-2.2B-Instruct-Agentic-GUI-Q4_K_M.gguf \\
         --mmproj ~/smolvlm-gui/SmolVLM2-2.2B-Instruct-Agentic-GUI-mmproj-f16.gguf \\
         -c 4096 -ngl 99 --port 8888 --chat-template smolvlm

4. Install requests if not present (usually already available):
       pip install requests

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

import requests
from datasets import load_dataset

MODEL_NAME = "SmolVLM2-2.2B-Instruct-Agentic-GUI-Q4_K_M.gguf"

# SmolVLM2 chat template tokens (used when building the raw prompt for /completion)
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"


# ── Image encoding ────────────────────────────────────────────────────────────

def _image_to_b64(image):
    """Convert a PIL Image to a raw base64 string (no data: prefix)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ── Coordinate parsing ────────────────────────────────────────────────────────

def parse_click_coords(text):
    """
    Extract click coordinates from SmolVLM2 output.
    Coordinates are in [0, 1] normalized space.
    Returns (x, y) as floats, or None if no click found.

    Handles both formats the model may produce:
      click(x=0.091, y=0.299)   ← keyword form (inside <code> blocks)
      click(0.091, 0.299)       ← positional form
    """
    # Keyword form: click(x=0.091, y=0.299)
    m = re.search(r"click\(\s*x\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*y\s*=\s*([0-9]*\.?[0-9]+)\s*\)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # Positional form: click(0.091, 0.299)
    m = re.search(r"click\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


# ── Hit detection ─────────────────────────────────────────────────────────────

def is_hit(pred_x, pred_y, bbox):
    """SmolVLM2 and ScreenSpot both use [0,1] — no conversion needed."""
    x_min, y_min, x_max, y_max = bbox
    return x_min <= pred_x <= x_max and y_min <= pred_y <= y_max


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(server_url, image, instruction):
    """
    Call the llama-server native /completion endpoint with image_data.

    The /v1/chat/completions (OpenAI-compatible) endpoint fails for SmolVLM2
    with 'Failed to tokenize prompt' because the smolvlm chat template doesn't
    handle the OpenAI vision content format correctly in llama.cpp.

    The native /completion endpoint accepts images via the image_data field
    alongside a [img-N] placeholder in the raw prompt (N must match image_data id).
    Note: <image> is a model-level token handled by the chat template layer;
    the raw /completion endpoint uses [img-1] as the placeholder instead.
    """
    b64 = _image_to_b64(image)

    # Manually apply the SmolVLM2 chat template.
    # Format: <|im_start|>user\n[img-1]\n{text}<|im_end|>\n<|im_start|>assistant\n
    # [img-1] is llama.cpp's /completion image placeholder (matches image_data id=1).
    user_text = (
        "You are a GUI agent. Output only a single action.\n"
        "Use normalized [0, 1] coordinates where (0,0) is top-left, (1,1) is bottom-right.\n"
        "Format: click(x, y)\n\n"
        f"Task: {instruction}"
    )
    prompt = (
        f"{_IM_START}user\n"
        f"[img-1]\n"
        f"{user_text}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )

    resp = requests.post(
        f"{server_url}/completion",
        json={
            "prompt":     prompt,
            "image_data": [{"data": b64, "id": 1}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stop": [_IM_END, _IM_START],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"].strip()


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    # Verify the server is reachable
    try:
        health = requests.get(f"{args.server_url}/health", timeout=5)
        health.raise_for_status()
        print(f"Connected to llama-server at {args.server_url}")
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

        raw_output = run_inference(args.server_url, image, instruction)
        coords     = parse_click_coords(raw_output)

        hit         = False
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
                    "model":   MODEL_NAME,
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
    parser.add_argument("--server-url",  type=str, default="http://localhost:8888",
                        help="llama-server base URL (default: http://localhost:8888).")
    parser.add_argument("--split",       type=str, default="test",
                        help="Dataset split (default: test).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to N examples for a quick smoke test.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to write full per-example JSON results.")
    evaluate(parser.parse_args())
