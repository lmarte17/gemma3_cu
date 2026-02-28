"""
Loads a subset of the GUI-Libra SFT dataset from locally downloaded files.

Only the web-relevant image groups are used:
  - guiact-web  (images extracted from guiact-web.tar.gz.part-*)
  - mind2web    (images extracted from mind2web.tar.gz.part-000)

Expected layout under --data-dir after download and extraction:

    GUI-Libra-web-subset/data/
      annotations/
        guiact-web-reasoning_and_grounding_changecoord.json
        guiact-web-reasoning_and_grounding_changecoord_1000.json
        guiact-web-*noreason*.json   (any noreason variants)
        mind2web-reasoning_and_grounding_changecoord.json
        mind2web-reasoning_and_grounding_changecoord_1000.json
        mind2web-*noreason*.json
      images/                        ← raw *.tar.gz.part-* files (downloaded by huggingface-cli)
      images_extracted/
        guiact-web/                  ← extracted images for guiact-web
        mind2web/                    ← extracted images for mind2web

BEFORE running this script, download only the web-subset files and stream-extract them:

    # Download only the files we need (~85 GB, vs 163 GB for the full repo)
    huggingface-cli download GUI-Libra/GUI-Libra-81K-SFT \
      --repo-type dataset \
      --include "data/images/guiact-web.tar.gz.part-*" \
      --include "data/images/mind2web.tar.gz.part-*" \
      --include "data/annotations/guiact-web-*" \
      --include "data/annotations/mind2web-*" \
      --local-dir ~/GUI-Libra-web-subset

    cd ~/GUI-Libra-web-subset/data
    mkdir -p images_extracted/guiact-web images_extracted/mind2web

    # Stream-extract directly from parts (avoids writing a merged archive to disk)
    cat images/guiact-web.tar.gz.part-* | tar -xz -C images_extracted/guiact-web/
    rm images/guiact-web.tar.gz.part-*    # free ~84 GB before extracting mind2web

    cat images/mind2web.tar.gz.part-* | tar -xz -C images_extracted/mind2web/
    rm images/mind2web.tar.gz.part-*

Standalone usage:
    # Inspect JSON schema before running the full pipeline:
    python data_prep_local.py --peek

    # Test-load 50 examples and print dataset info:
    python data_prep_local.py --sample 50

    # Full load and save to disk for reuse:
    python data_prep_local.py --cache-dir ../data/processed_web
"""

import os
import re
import json
import glob
import argparse
from pathlib import Path
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor

from config import MODEL_ID, LOCAL_CACHE_DIR

# Annotation file name prefixes that select the web datasets we want
WEB_PREFIXES = ("guiact-web-", "mind2web-")

# Maps annotation file prefix → images_extracted/ subdirectory name
IMAGE_SUBDIR_MAP = {
    "guiact-web": "guiact-web",
    "mind2web":   "mind2web",
}

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_annotation_files(annotations_dir):
    """Return sorted list of all web-prefix JSON files in annotations_dir."""
    all_json = glob.glob(os.path.join(annotations_dir, "*.json"))
    return sorted(
        f for f in all_json
        if any(Path(f).name.startswith(p) for p in WEB_PREFIXES)
    )


def _load_json_file(filepath):
    """Load a JSON file that is either a list or a dict with a data/annotations key."""
    with open(filepath, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "annotations", "samples", "train"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    raise ValueError(
        f"Unrecognised JSON structure in {filepath}. "
        "Expected a top-level list or a dict with a 'data'/'annotations' key. "
        "Run with --peek to inspect the file."
    )


def _guess_image_subdir(annotation_filename):
    """Infer which images/ subdirectory to use from the annotation filename."""
    name = Path(annotation_filename).name
    for prefix, subdir in IMAGE_SUBDIR_MAP.items():
        if name.startswith(prefix):
            return subdir
    return ""


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

def _build_image_index(images_dir, image_subdir):
    """
    Walk images_extracted/<image_subdir>/ recursively and build a
    {basename: full_path} index.  This handles any internal directory
    structure the tar archive may have created (e.g. images_extracted/
    guiact-web/images/ or images_extracted/mind2web/mind2web/).
    """
    subdir_path = os.path.join(images_dir, image_subdir)
    index = {}
    if not os.path.isdir(subdir_path):
        return index
    for root, _, files in os.walk(subdir_path):
        for fname in files:
            if Path(fname).suffix.lower() in _IMAGE_EXTS:
                # Keep the last writer if there are duplicate basenames
                index[fname] = os.path.join(root, fname)
    return index


def _open_image(image_field, image_index):
    """
    Look up an image by basename in the pre-built index.
    Falls back to treating image_field as a literal absolute path.
    Returns an open PIL Image or None.
    """
    basename = Path(image_field).name
    full_path = image_index.get(basename) or (image_field if os.path.isfile(image_field) else None)
    if full_path is None:
        return None
    try:
        return Image.open(full_path).convert("RGB")
    except Exception:
        return None


def _extract_from_conversations(conversations):
    """
    Parse a conversations list (ShareGPT / LLaVA format) into (goal, model_response).

    Supports both:
      {"from": "human", "value": "..."}  and  {"role": "user", "content": "..."}

    The human turn becomes the goal; the gpt/assistant turn becomes the model
    response (used as-is, since it already contains the "Reasoning: ... Action: ..."
    or "Action: ..." format produced by the dataset).

    Returns (None, None) if the expected turns are missing.
    """
    if not isinstance(conversations, list):
        return None, None

    human_text = None
    gpt_text = None
    for turn in conversations:
        role = turn.get("from") or turn.get("role") or ""
        text = (turn.get("value") or turn.get("content") or "").strip()
        if role in ("human", "user") and human_text is None:
            human_text = text
        elif role in ("gpt", "assistant", "model") and gpt_text is None:
            gpt_text = text

    if not human_text or not gpt_text:
        return None, None

    # Strip leading <image> token that some LLaVA-style datasets embed in the human turn
    human_text = re.sub(r"<image>\s*", "", human_text).strip()
    return human_text, gpt_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_local_web_dataset(data_dir, processor, sample_size=None):
    """
    Load and tokenize the web subset of the SFT data from local files.

    Args:
        data_dir:    Root data directory containing annotations/ and images/.
        processor:   Already-loaded Gemma-3 AutoProcessor (passed in to avoid
                     loading it twice when called from train.py).
        sample_size: Optional cap on total examples loaded (for quick tests).

    Returns:
        A tokenized HuggingFace Dataset with input_ids, attention_mask,
        and pixel_values — drop-in replacement for process_dataset().
    """
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir      = os.path.join(data_dir, "images_extracted")

    ann_files = _find_annotation_files(annotations_dir)
    if not ann_files:
        raise FileNotFoundError(
            f"No guiact-web / mind2web JSON files found in {annotations_dir}.\n"
            "Expected file names starting with 'guiact-web-' or 'mind2web-'.\n"
            "Run with --peek to verify the annotations directory."
        )

    print(f"Found {len(ann_files)} annotation file(s):")
    for f in ann_files:
        print(f"  {Path(f).name}")

    # Build a {basename: full_path} index for every image subdir once up-front.
    # This walks the extracted tree regardless of how deep the tar placed the files,
    # so it works whether images landed in images_extracted/guiact-web/images/ or
    # images_extracted/mind2web/mind2web/ or any other depth.
    image_indexes = {}
    for subdir in set(IMAGE_SUBDIR_MAP.values()):
        idx = _build_image_index(images_dir, subdir)
        image_indexes[subdir] = idx
        if idx:
            print(f"  Indexed {len(idx):,} images under images_extracted/{subdir}/")
        else:
            print(f"  ⚠️  No images found under images_extracted/{subdir}/ — check extraction.")

    records = []   # raw dicts: {"messages": [...], "image": PIL.Image}
    missing_fields = 0
    missing_image  = 0

    for ann_file in ann_files:
        if sample_size and len(records) >= sample_size:
            break

        subdir  = _guess_image_subdir(ann_file)
        entries = _load_json_file(ann_file)
        idx     = image_indexes.get(subdir, {})
        print(f"  Parsing {Path(ann_file).name} → {len(entries)} entries  (index size: {len(idx):,})")

        # --- Print the actual keys and resolved values for the first entry ---
        if entries:
            first = entries[0]
            print(f"    Schema keys : {list(first.keys())}")
            convs = first.get("conversations")
            if convs and isinstance(convs, list):
                for i, turn in enumerate(convs[:2]):
                    role = turn.get("from") or turn.get("role", "?")
                    text = (turn.get("value") or turn.get("content") or "")[:100]
                    print(f"    conversations[{i}] ({role}): {repr(text)}")
            else:
                print(f"    image_field : {repr(first.get('image') or first.get('img_filename') or first.get('screenshot') or first.get('img_path') or '(not found)')}")
                print(f"    goal        : {repr((first.get('goal') or first.get('instruction') or first.get('task_instruction') or first.get('query') or '(not found)'))[:80]}")
                print(f"    action      : {repr((first.get('action') or first.get('answer') or first.get('response') or '(not found)'))[:80]}")

        for entry in entries:
            if sample_size and len(records) >= sample_size:
                break

            image_field = (entry.get("image") or entry.get("img_filename")
                           or entry.get("screenshot") or entry.get("img_path") or "")

            # --- conversations format (ShareGPT / LLaVA style) ---
            conversations = entry.get("conversations")
            if conversations:
                goal, model_response = _extract_from_conversations(conversations)
                if not goal or not model_response:
                    missing_fields += 1
                    continue
            else:
                # Flat key format
                goal      = (entry.get("goal") or entry.get("instruction")
                             or entry.get("task_instruction") or entry.get("query") or "")
                reasoning = entry.get("reasoning") or entry.get("thought") or ""
                action    = (entry.get("action") or entry.get("answer")
                             or entry.get("response") or "")
                if not goal or not action:
                    missing_fields += 1
                    continue
                model_response = (
                    f"Reasoning: {reasoning} Action: {action}"
                    if reasoning else
                    f"Action: {action}"
                )

            if not image_field:
                missing_fields += 1
                continue

            basename = Path(image_field).name
            resolved_path = idx.get(basename) or (image_field if os.path.isfile(image_field) else None)
            if resolved_path is None:
                missing_image += 1
                continue

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
            records.append({"messages": messages, "image_path": resolved_path})

    if missing_fields:
        print(f"  ⚠️  Skipped {missing_fields} entries — field names not recognised (see schema keys above).")
    if missing_image:
        print(f"  ⚠️  Skipped {missing_image} entries — image file not found in index.")

    print(f"Total examples loaded: {len(records)}")

    if not records:
        raise RuntimeError(
            "No examples were loaded. Possible causes:\n"
            "  1. Images not extracted yet (run the tar commands in the docstring).\n"
            "  2. Image paths in JSON don't match the extracted directory structure.\n"
            "     Run --peek to inspect the JSON schema and compare paths."
        )

    # --- Tokenization (mirrors data_prep.process_dataset logic) ---
    print("Tokenizing dataset...")

    def tokenize_batch(batch):
        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        # Text-only, no padding — collator dynamically pads each batch to its actual
        # max length, keeping logits small and preventing OOM on long-vocab models.
        inputs = processor.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=2048,
        )
        return {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    raw_dataset = Dataset.from_list(records)
    tokenized = raw_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=32,
        num_proc=16,
        remove_columns=["messages"],  # keep image_path for the collator
    )
    return tokenized


# ---------------------------------------------------------------------------
# CLI — run standalone to inspect / pre-process data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local SFT data loader — web subset (guiact-web + mind2web) only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", type=str, default=_DEFAULT_DATA_DIR,
        help="Root data dir containing annotations/ and images/. "
             f"Defaults to {_DEFAULT_DATA_DIR}"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Cap total examples loaded (useful for a quick sanity check)."
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="If set, save the tokenized dataset here for reuse in training."
    )
    parser.add_argument(
        "--peek", action="store_true",
        help="Print the keys and first entry from each annotation file, then exit. "
             "Use this to discover the JSON schema before running the full pipeline."
    )
    args = parser.parse_args()

    ann_dir = os.path.join(args.data_dir, "annotations")

    if args.peek:
        import pprint
        files = _find_annotation_files(ann_dir)
        if not files:
            print(f"No matching files found in {ann_dir}")
            raise SystemExit(1)
        for f in files:
            entries = _load_json_file(f)
            print(f"\n{'='*60}")
            print(f"File : {Path(f).name}")
            print(f"Count: {len(entries)} entries")
            if entries:
                print("Keys :", list(entries[0].keys()))
                print("First entry:")
                pprint.pprint(entries[0], width=100)
        raise SystemExit(0)

    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    dataset = load_local_web_dataset(args.data_dir, processor=processor, sample_size=args.sample)
    print(dataset)

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Saving tokenized dataset to {args.cache_dir}")
        dataset.save_to_disk(args.cache_dir)
        print("Done.")
