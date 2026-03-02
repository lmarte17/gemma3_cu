"""
eval_uitars.py — UI-TARS-2B-SFT ScreenSpot Benchmark Evaluation

Evaluates ByteDance's UI-TARS-2B-SFT on ScreenSpot.  UI-TARS is Qwen2-VL
based and outputs coordinates in [0, 1000] scale.

Dependencies (beyond the existing project requirements):
    pip install qwen-vl-utils

Usage:
    python eval_uitars.py --max-samples 100 --output-file results_uitars.json
    python eval_uitars.py --split test --output-file results_uitars_full.json
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import torch
from PIL import Image
from datasets import load_dataset

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False
    print("Warning: qwen-vl-utils not found. Install with: pip install qwen-vl-utils")
    print("Falling back to basic image handling — results may vary.\n")

# ── Transformers 5.x video-processor bug fix ─────────────────────────────────
# video_processor_class_from_name() crashes with TypeError when the internal
# `extractors` dict is None.  This affects Qwen2VL loading on transformers 5.x.
# Monkey-patch it to return None (= no video processor) instead of raising.
try:
    import transformers.models.auto.video_processing_auto as _vpa
    _orig_vp = _vpa.video_processor_class_from_name
    def _safe_vp(class_name):
        try:
            return _orig_vp(class_name)
        except TypeError:
            return None
    _vpa.video_processor_class_from_name = _safe_vp
except Exception:
    pass

try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    HAS_QWEN2VL = True
except ImportError:
    from transformers import AutoModelForCausalLM
    HAS_QWEN2VL = False

try:
    from transformers import AutoProcessor
    HAS_AUTO_PROCESSOR = True
except ImportError:
    HAS_AUTO_PROCESSOR = False


# ── Transformers 5.x compatibility fix ───────────────────────────────────────

def _fix_preprocessor_config(model_id):
    """
    Transformers 5.x Qwen2VLImageProcessor expects size = {shortest_edge, longest_edge}.
    UI-TARS's preprocessor_config.json uses size = {min_pixels, max_pixels} (older format).
    This patches the cached config file in-place so AutoProcessor can load it.

    sqrt(min_pixels=3136) = 56  →  shortest_edge = 56
    sqrt(max_pixels=2116800) ≈ 1455  →  longest_edge = 1455
    """
    model_slug = model_id.replace("/", "--")
    pattern = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        f"models--{model_slug}", "snapshots", "*", "preprocessor_config.json",
    )
    patched = False
    for config_path in glob.glob(pattern):
        with open(config_path) as f:
            config = json.load(f)
        size = config.get("size", {})
        if "min_pixels" in size and "shortest_edge" not in size:
            config["size"] = {
                "shortest_edge": int(math.sqrt(size["min_pixels"])),
                "longest_edge":  round(math.sqrt(size["max_pixels"])),
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Patched preprocessor config (transformers 5.x fix): {config_path}")
            patched = True
    return patched

MODEL_ID = "ByteDance-Seed/UI-TARS-2B-SFT"

# ── System prompt (COMPUTER_USE format from UI-TARS) ─────────────────────────
# UI-TARS was trained with this prompt structure. Using the correct prompt
# is critical — wrong prompt causes large accuracy drops.

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: <your reasoning about what to do next>
Action: <action>
```

## Action Space
click(start_box='<bbox>x1 y1 x2 y2</bbox>')
type(content='text to type')
scroll(start_box='<bbox>x1 y1 x2 y2</bbox>', direction='down' | 'up' | 'left' | 'right')
hotkey(key='key combination')
finished(content='task complete message')

## Notes
- Coordinates are in [0, 1000] scale relative to the screenshot dimensions
- start_box specifies the target element bounding box (x1 y1 = top-left, x2 y2 = bottom-right)
- If the box is a single point, use the same value for both corners: (x y x y)"""


# ── Coordinate parsing ────────────────────────────────────────────────────────

def parse_click_coords(text):
    """
    Extract click coordinates from UI-TARS output.

    Handles multiple formats the model may produce:
      click(start_box='<bbox>x1 y1 x2 y2</bbox>')   ← primary format
      click(start_box='(x1,y1,x2,y2)')               ← alternate bbox
      click(start_box='(x1,y1)')                     ← point form

    Returns (mid_x, mid_y) in [0, 1000] space, or None if no click found.
    """
    # Primary format: <bbox>x1 y1 x2 y2</bbox>
    m = re.search(
        r"click\(start_box='<bbox>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</bbox>'\)",
        text,
    )
    if m:
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Alternate: (x1,y1,x2,y2) with commas
    m = re.search(
        r"click\(start_box='?\(?(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)\)?'?\)",
        text,
    )
    if m:
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Point form: (x,y)
    m = re.search(
        r"click\(start_box='?\(?(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)\)?'?\)",
        text,
    )
    if m:
        return float(m.group(1)), float(m.group(2))

    # Point tag format: click(point='<point>x y</point>')
    m = re.search(
        r"click\(point='<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>'\)",
        text,
    )
    if m:
        return float(m.group(1)), float(m.group(2))

    return None


# ── Hit detection ─────────────────────────────────────────────────────────────

def is_hit(pred_x, pred_y, bbox):
    """
    UI-TARS outputs [0, 1000] scale.  ScreenSpot bboxes are [0, 1] normalized.
    Divide by 1000 before comparing.
    """
    px, py = pred_x / 1000.0, pred_y / 1000.0
    x_min, y_min, x_max, y_max = bbox
    return x_min <= px <= x_max and y_min <= py <= y_max


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    _fix_preprocessor_config(MODEL_ID)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("CUDA device detected.")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Warning: No GPU detected — inference will be slow.")

    print(f"Loading {MODEL_ID}...")
    # Use Qwen2VLProcessor directly — AutoProcessor triggers a NoneType bug in
    # transformers 5.x when auto-detecting the video processor class for Qwen2VL.
    if HAS_QWEN2VL:
        processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    elif HAS_AUTO_PROCESSOR:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    else:
        raise RuntimeError("No suitable processor class available.")

    if HAS_QWEN2VL:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    print("Model loaded.\n")
    return model, processor, device


# ── Single-example inference ──────────────────────────────────────────────────

def run_inference(model, processor, device, image, instruction):
    user_text = (
        "Please perform the next action to complete the task.\n\n"
        f"Task: {instruction}\n\n"
        "Interaction History:\n(none)"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if HAS_QWEN_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
    else:
        # Fallback: pass PIL image directly (works with recent transformers)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,         # greedy — more stable for eval
            repetition_penalty=1.05,
        )

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    model, processor, device = load_model()

    print(f"Loading ScreenSpot (split='{args.split}')...")
    dataset = load_dataset("rootsautomation/ScreenSpot", split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    total = len(dataset)
    print(f"Evaluating on {total} examples...\n")

    hits = 0
    parse_failures = 0
    by_platform  = defaultdict(lambda: {"hits": 0, "total": 0})
    by_elem_type = defaultdict(lambda: {"hits": 0, "total": 0})
    results = []

    for i, example in enumerate(dataset):
        image       = example["image"].convert("RGB")
        instruction = example["instruction"]
        bbox        = example["bbox"]
        platform    = example.get("platform", "unknown")
        elem_type   = example.get("element_type", "unknown")

        raw_output = run_inference(model, processor, device, image, instruction)
        coords = parse_click_coords(raw_output)

        hit = False
        pred_coords = None

        if coords is None:
            parse_failures += 1
        else:
            pred_x, pred_y = coords
            pred_coords = (pred_x, pred_y)
            hit = is_hit(pred_x, pred_y, bbox)

        if hit:
            hits += 1

        by_platform[platform]["total"]  += 1
        by_elem_type[elem_type]["total"] += 1
        if hit:
            by_platform[platform]["hits"]  += 1
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
    print("  UI-TARS-2B SCREENSPOT EVALUATION RESULTS")
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
                    "model": MODEL_ID,
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
    parser = argparse.ArgumentParser(description="UI-TARS-2B ScreenSpot Evaluation")
    parser.add_argument("--split",       type=str, default="test",
                        help="Dataset split (default: test).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to N examples for a quick smoke test.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to write full per-example JSON results.")
    evaluate(parser.parse_args())
