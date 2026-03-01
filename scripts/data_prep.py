import os
import io
import math
import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor
from config import DATASET_ID_RL, DATASET_ID_SFT, DEFAULT_DATASET, MODEL_ID, LOCAL_CACHE_DIR


# ---------------------------------------------------------------------------
# Action formatting helper (RL dataset)
# ---------------------------------------------------------------------------

def _format_action_str(gt_action, gt_point_2d, gt_input_text):
    """Convert RL dataset answer fields into the 'Action: ...' string the model learns to produce."""
    action_lower = (gt_action or "").lower().strip()
    has_coords = gt_point_2d and len(gt_point_2d) >= 2

    if action_lower == "click" and has_coords:
        x, y = round(gt_point_2d[0]), round(gt_point_2d[1])
        return f"Action: click({x}, {y})"
    elif action_lower == "type" and gt_input_text:
        return f'Action: type("{gt_input_text}")'
    elif action_lower in ("scroll", "swipe") and has_coords:
        x, y = round(gt_point_2d[0]), round(gt_point_2d[1])
        return f"Action: scroll({x}, {y})"
    elif action_lower == "hover" and has_coords:
        x, y = round(gt_point_2d[0]), round(gt_point_2d[1])
        return f"Action: hover({x}, {y})"
    elif action_lower == "long_press" and has_coords:
        x, y = round(gt_point_2d[0]), round(gt_point_2d[1])
        return f"Action: long_press({x}, {y})"
    elif has_coords:
        x, y = round(gt_point_2d[0]), round(gt_point_2d[1])
        return f"Action: {gt_action}({x}, {y})"
    else:
        return f"Action: {gt_action}"


# ---------------------------------------------------------------------------
# SFT formatter  (GUI-Libra-81K-SFT schema)
# ---------------------------------------------------------------------------

def format_example_for_gemma3(example):
    """
    Takes a single example from the GUI-Libra SFT dataset and formats it
    into the list of dictionaries expected by Gemma-3 `apply_chat_template`.
    """
    image = example.get("image") or example.get("screenshot")
    goal = example.get("goal") or example.get("instruction", "")
    reasoning = example.get("reasoning", "")
    action = example.get("action", "")

    model_response = f"Reasoning: {reasoning} Action: {action}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Goal: {goal}"}
            ]
        },
        {
            "role": "model",
            "content": [{"type": "text", "text": model_response}]
        }
    ]

    image_bytes = None
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()

    return {"messages": messages, "image_bytes": image_bytes}


# ---------------------------------------------------------------------------
# RL formatter  (GUI-Libra-81K-RL schema)
# ---------------------------------------------------------------------------

def format_example_for_gemma3_rl(example):
    """
    Formats a GUI-Libra-81K-RL example.

    Key optimisations vs the SFT formatter:
      - Images arrive as raw bytes in images[0]['bytes']; we store them
        directly without any PIL decode/re-encode (saves the most expensive
        CPU step in the format map).
      - difficulty_weight is derived from gt_bbox area: smaller target →
        harder → higher weight.  Stored as a float so the collator can
        multiply it into loss_weights at training time.
      - action_type is stored for optional stratified sampling.
    """
    # --- Image: raw bytes passthrough, no PIL decode+encode ---
    image_bytes = None
    images_field = example.get("images") or []
    if images_field:
        img = images_field[0]
        if isinstance(img, dict):
            raw = img.get("bytes")
        elif isinstance(img, (bytes, bytearray)):
            raw = img
        else:
            raw = img  # may be a PIL Image if HF auto-decoded

        if isinstance(raw, (bytes, bytearray)):
            image_bytes = bytes(raw)          # zero-copy store; collator handles decode
        elif hasattr(raw, "save"):            # PIL Image fallback
            buf = io.BytesIO()
            raw.convert("RGB").save(buf, format="JPEG", quality=95)
            image_bytes = buf.getvalue()

    # --- Text fields ---
    system_prompt = (example.get("system_prompt") or "").strip()
    context = (example.get("context") or "").strip()

    # --- Answer / ground-truth fields ---
    answer = example.get("answer") or {}
    gt_action     = answer.get("gt_action", "")
    gt_point_2d   = answer.get("gt_point_2d") or []
    gt_bbox       = answer.get("gt_bbox") or []
    gt_input_text = answer.get("gt_input_text") or ""
    image_height  = answer.get("image_height") or 1
    image_width   = answer.get("image_width") or 1

    action_str = _format_action_str(gt_action, gt_point_2d, gt_input_text)

    # --- Difficulty weight from bbox area (no inference needed) ---
    # Smaller target element → harder to hit → higher loss weight (capped at 5×)
    difficulty_weight = 1.0
    if len(gt_bbox) == 4 and image_height > 0 and image_width > 0:
        x1, y1, x2, y2 = gt_bbox
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        total_area = image_height * image_width
        if bbox_area > 0:
            normalized_area = bbox_area / total_area
            difficulty_weight = float(min(5.0, max(1.0, 1.0 / math.sqrt(normalized_area))))

    # --- Build Gemma-3 chat messages ---
    user_content = [{"type": "image"}, {"type": "text", "text": context}]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "model", "content": [{"type": "text", "text": action_str}]})

    return {
        "messages": messages,
        "image_bytes": image_bytes,
        "difficulty_weight": difficulty_weight,
        "action_type": gt_action,
    }


# ---------------------------------------------------------------------------
# Stratified sampling helper (RL only, non-streaming)
# ---------------------------------------------------------------------------

def _stratify_by_action(dataset, target_size):
    """
    Balance dataset by action_type.  Runs after the format map so
    action_type is already a plain-text column (no image I/O).

    If target_size is None, balances to the count of the rarest action type.
    Otherwise distributes target_size evenly across types.
    """
    action_types = dataset.unique("action_type")
    action_types = [a for a in action_types if a]  # drop empty strings
    if not action_types:
        return dataset

    n_per_type = (target_size // len(action_types)) if target_size else None

    subsets = []
    for at in action_types:
        sub = dataset.filter(lambda x, t=at: x["action_type"] == t, num_proc=16)
        if n_per_type is not None:
            n = min(n_per_type, len(sub))
        else:
            n = len(sub)
        subsets.append(sub.select(range(n)))

    # Trim all subsets to the smallest one if no target_size given
    if n_per_type is None:
        min_n = min(len(s) for s in subsets)
        subsets = [s.select(range(min_n)) for s in subsets]

    combined = concatenate_datasets(subsets).shuffle(seed=42)
    if target_size:
        combined = combined.select(range(min(target_size, len(combined))))
    return combined


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_id=DEFAULT_DATASET,
    sample_size=None,
    cache_dir=None,
    stream=False,
    processor=None,   # pass in from train.py to avoid loading twice
    stratify=False,   # RL only: balance examples by action type
):
    is_rl = (dataset_id == DATASET_ID_RL)

    # GUI-Libra-81K-RL only has a "validation" split on HuggingFace
    split = "validation" if is_rl else "train"
    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, split=split, streaming=stream)

    # For non-stratified runs, cap early to avoid formatting unused examples
    if sample_size and not stratify:
        print(f"Sampling {sample_size} examples (streaming: {stream})...")
        if stream:
            dataset = dataset.take(sample_size)
        else:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

    # Choose formatter based on dataset
    formatter = format_example_for_gemma3_rl if is_rl else format_example_for_gemma3

    print("Formatting messages...")
    formatted_dataset = dataset.map(formatter, num_proc=None if stream else 16)

    # Stratified sampling runs after format map (action_type is a plain column now)
    if stratify and is_rl and not stream:
        print("Stratifying by action type...")
        formatted_dataset = _stratify_by_action(formatted_dataset, sample_size)
        print(f"  {len(formatted_dataset)} examples after stratification")
    elif sample_size and stratify:
        # Non-RL with stratify flag — just cap
        formatted_dataset = formatted_dataset.select(range(min(sample_size, len(formatted_dataset))))

    # Load processor only if not already provided (avoids loading twice from train.py)
    if processor is None:
        print(f"Loading processor: {MODEL_ID}")
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    def tokenize_and_prepare(batch):
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in batch["messages"]
        ]
        # Text-only, no padding — collator dynamically pads each batch to its actual
        # max length, keeping logits small and preventing OOM on long-vocab models.
        inputs = processor.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=2048
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    print("Tokenizing dataset (text only — images handled on-the-fly in collator)...")
    # remove_columns drops 'messages'; all other columns (image_bytes, difficulty_weight,
    # action_type) survive automatically and are passed through to the collator.
    tokenized_dataset = formatted_dataset.map(
        tokenize_and_prepare,
        batched=True,
        batch_size=32,
        num_proc=None if stream else 16,
        remove_columns=["messages"]
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Saving to {cache_dir}")
        tokenized_dataset.save_to_disk(cache_dir)
        print("Done!")

    return tokenized_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI-Libra Data Preparation for Gemma-3")
    parser.add_argument("--dataset", type=str, choices=["rl", "sft"], default="rl",
                        help="Which dataset to process (RL or SFT).")
    parser.add_argument("--sample", type=int,
                        help="Number of examples to process.")
    parser.add_argument("--stream", action="store_true",
                        help="Use dataset streaming.")
    parser.add_argument("--stratify", action="store_true",
                        help="(RL only) Balance examples by action type before training.")
    parser.add_argument("--cache-dir", type=str, default=LOCAL_CACHE_DIR,
                        help="Local directory to save processed data.")

    args = parser.parse_args()
    dataset_id = DATASET_ID_SFT if args.dataset == "sft" else DATASET_ID_RL

    print(f"Starting pipeline for {args.dataset.upper()} dataset. Cache dir: {args.cache_dir}")
    process_dataset(
        dataset_id=dataset_id,
        sample_size=args.sample,
        cache_dir=args.cache_dir,
        stream=args.stream,
        stratify=args.stratify,
    )
