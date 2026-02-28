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


def _open_image(image_field, images_dir, image_subdir):
    """Try several common path layouts. Returns an open PIL Image or None."""
    candidates = [
        image_field,                                                  # absolute / already correct
        os.path.join(images_dir, image_field),                       # images/<as-written>
        os.path.join(images_dir, image_subdir, image_field),         # images/<subdir>/<as-written>
        os.path.join(images_dir, image_subdir, Path(image_field).name),  # images/<subdir>/<basename>
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
    return None


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

    records = []   # raw dicts: {"messages": [...], "image": PIL.Image}
    missing = 0

    for ann_file in ann_files:
        if sample_size and len(records) >= sample_size:
            break

        subdir  = _guess_image_subdir(ann_file)
        entries = _load_json_file(ann_file)
        print(f"  Parsing {Path(ann_file).name} → {len(entries)} entries  (images/{subdir}/)")

        for entry in entries:
            if sample_size and len(records) >= sample_size:
                break

            # Flexible key resolution — handles multiple common JSON schemas
            image_field = (entry.get("image") or entry.get("img_filename")
                           or entry.get("screenshot") or entry.get("img_path") or "")
            goal        = (entry.get("goal") or entry.get("instruction")
                           or entry.get("task_instruction") or entry.get("query") or "")
            reasoning   = entry.get("reasoning") or entry.get("thought") or ""
            action      = (entry.get("action") or entry.get("answer")
                           or entry.get("response") or "")

            if not image_field or not goal or not action:
                missing += 1
                continue

            pil_image = _open_image(image_field, images_dir, subdir)
            if pil_image is None:
                missing += 1
                continue

            # Same output format as data_prep.py
            model_response = (
                f"Reasoning: {reasoning} Action: {action}"
                if reasoning else
                f"Action: {action}"
            )

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
            records.append({"messages": messages, "image": pil_image})

    if missing:
        print(f"  ⚠️  Skipped {missing} entries (missing image file or required fields).")

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
        inputs = processor(
            text=texts,
            images=batch["image"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,
        )
        result = {
            "input_ids":      inputs["input_ids"].numpy().tolist(),
            "attention_mask": inputs["attention_mask"].numpy().tolist(),
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].numpy().tolist()
        return result

    raw_dataset = Dataset.from_list(records)
    tokenized = raw_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=8,
        remove_columns=raw_dataset.column_names,
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
