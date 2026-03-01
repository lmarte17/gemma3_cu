"""
eval.py — GUI-Libra ScreenSpot Benchmark Evaluation

Runs the fine-tuned model against the ScreenSpot grounding benchmark and reports:
  - Click Accuracy per platform (mobile, desktop, web)
  - Click Accuracy per element type (icon, text)
  - Action Type Accuracy (for type/scroll/swipe examples, if present)
  - Parse Failure Rate (% of outputs where no valid action could be extracted)

Usage (from inside scripts/):
    python eval.py --lora-path ../gemma3-gui-libra-lora-final
    python eval.py --lora-path ../gemma3-gui-libra-lora-final --split test --max-samples 200
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

from config import MODEL_ID


# ── System prompt (matches RL training data exactly) ─────────────────────────

SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of actions to complete the task. "
    "You need to choose actions from the the following list:\n\n"
    "action_type: Answer, action_target: None, value: Answer text, point_2d: None\n"
    "    ## Explanation: Return the final answer to the user's question\n\n"
    "action_type: Click, action_target: Element description, value: None, point_2d: [x, y]\n"
    "    ## Explanation: Tap or click a specific UI element and provide its coordinates\n\n"
    "action_type: Write, action_target: Element description or None, value: Text to enter, point_2d: [x, y] or None\n"
    "    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None\n\n"
    "action_type: LongPress, action_target: Element description, value: None, point_2d: [x, y]\n"
    "    ## Explanation: Press and hold on a specific UI element (mobile only) and provide its coordinates\n\n"
    "action_type: Scroll, action_target: None, value: \"up\" | \"down\" | \"left\" | \"right\", point_2d: None\n"
    "    ## Explanation: Scroll a view or container in the specified direction\n\n"
    "action_type: Swipe, action_target: Optional position or None, value: \"up\" | \"down\" | \"left\" | \"right\", point_2d: [x, y] or None\n"
    "    ## Explanation: Perform a swipe gesture on the screen in the given direction (mobile)\n\n"
    "action_type: Wait, action_target: None, value: Number of seconds, point_2d: None\n"
    "    ## Explanation: Pause execution to allow the UI to load or update\n\n"
    "action_type: NavigateHome, action_target: None, value: None, point_2d: None\n"
    "    ## Explanation: Navigate to the device's home screen\n\n"
    "action_type: NavigateBack, action_target: None, value: None, point_2d: None\n"
    "    ## Explanation: Press the system \"Back\" button\n\n"
    "action_type: OpenApp, action_target: None, value: App name, point_2d: None\n"
    "    ## Explanation: Launch an app by its name (mobile only)\n\n"
    "action_type: Terminate, action_target: None, value: End-task message, point_2d: None\n"
    "    ## Explanation: Signal the end of the current task with a final message"
)


# ── Coordinate parsing ────────────────────────────────────────────────────────

def parse_action(text):
    # Native RL dataset format: action_type: Click, ... point_2d: [x, y]
    pt_match = re.search(r"point_2d:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]", text)
    if pt_match:
        at_match = re.search(r"action_type:\s*(\w+)", text)
        action_type = at_match.group(1).lower() if at_match else "click"
        return {"type": action_type, "args": [pt_match.group(1), pt_match.group(2)]}

    # Fallback: custom format Action: click(x, y)
    match = re.search(r"Action:\s*([a-zA-Z_]+)\((.*?)\)", text)
    if not match:
        return None
    action_type = match.group(1)
    args = [a.strip().strip('"').strip("'") for a in match.group(2).split(",")]
    return {"type": action_type, "args": args}


# ── Hit detection ─────────────────────────────────────────────────────────────

def is_hit(pred_x, pred_y, bbox, img_w, img_h):
    """
    Returns True if (pred_x, pred_y) on the 0-1000 scale lands inside bbox.

    ScreenSpot stores bboxes as [x_min, y_min, x_max, y_max] normalised to [0, 1].
    We convert the model's 0-1000 prediction to [0, 1] before comparing.
    """
    px = pred_x / 1000.0
    py = pred_y / 1000.0
    x_min, y_min, x_max, y_max = bbox
    return x_min <= px <= x_max and y_min <= py <= y_max


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(lora_path):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("✅ MPS device detected.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        print("✅ CUDA device detected.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️  No hardware acceleration — CPU will be slow.")

    print(f"Loading base model ({MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    if lora_path:
        print(f"Applying LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print("⚠️  No LoRA path provided — evaluating the base model.")
        model = base_model

    model.eval()
    return model, processor, device, dtype


# ── Single-example inference ──────────────────────────────────────────────────

def run_inference(model, processor, device, dtype, image, instruction):
    user_text = (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        "Interaction History: \n(none)"
    )
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
        )

    input_len = inputs["input_ids"].shape[1]
    raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return raw


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    model, processor, device, dtype = load_model(args.lora_path)

    print(f"\nLoading ScreenSpot (split='{args.split}')...")
    dataset = load_dataset("rootsautomation/ScreenSpot", split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    total = len(dataset)
    print(f"Evaluating on {total} examples...\n")

    hits = 0
    parse_failures = 0

    # Breakdowns: platform (mobile/desktop/web) and element type (icon/text)
    by_platform = defaultdict(lambda: {"hits": 0, "total": 0})
    by_elem_type = defaultdict(lambda: {"hits": 0, "total": 0})

    results = []  # full log for --output-file

    for i, example in enumerate(dataset):
        image = example["image"].convert("RGB")
        instruction = example["instruction"]
        bbox = example["bbox"]               # [x_min, y_min, x_max, y_max] in [0,1]
        platform = example.get("platform", "unknown")
        elem_type = example.get("element_type", "unknown")
        img_w, img_h = image.size

        raw_output = run_inference(model, processor, device, dtype, image, instruction)
        parsed = parse_action(raw_output)

        hit = False
        pred_coords = None

        if parsed is None:
            parse_failures += 1
        else:
            try:
                pred_x = float(parsed["args"][0])
                pred_y = float(parsed["args"][1])
                pred_coords = (pred_x, pred_y)
                hit = is_hit(pred_x, pred_y, bbox, img_w, img_h)
            except (IndexError, ValueError):
                parse_failures += 1

        if hit:
            hits += 1

        by_platform[platform]["total"] += 1
        by_elem_type[elem_type]["total"] += 1
        if hit:
            by_platform[platform]["hits"] += 1
            by_elem_type[elem_type]["hits"] += 1

        results.append({
            "index": i,
            "instruction": instruction,
            "platform": platform,
            "element_type": elem_type,
            "bbox": bbox,
            "pred_coords": pred_coords,
            "raw_output": raw_output,
            "hit": hit,
        })

        # Progress
        done = i + 1
        running_acc = hits / done * 100
        print(
            f"[{done:>{len(str(total))}}/{total}]  "
            f"hit={hit}  running_acc={running_acc:.1f}%  "
            f"parse_fail={parse_failures}",
            end="\r",
        )

    print()  # newline after progress

    # ── Results summary ───────────────────────────────────────────────────────

    click_acc = hits / total * 100
    parse_fail_rate = parse_failures / total * 100

    print("\n" + "=" * 55)
    print("  SCREENSPOT EVALUATION RESULTS")
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
                    "summary": {
                        "total": total,
                        "hits": hits,
                        "click_accuracy": click_acc,
                        "parse_failures": parse_failures,
                        "parse_failure_rate": parse_fail_rate,
                        "by_platform": dict(by_platform),
                        "by_element_type": dict(by_elem_type),
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
    parser = argparse.ArgumentParser(description="GUI-Libra ScreenSpot Evaluation")
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the trained LoRA adapter directory. Omit to evaluate the base model.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to this many examples (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to write full per-example JSON results (e.g. results.json).",
    )
    evaluate(parser.parse_args())
