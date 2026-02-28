import os
import io
import argparse
from datasets import load_dataset
from transformers import AutoProcessor
from config import DATASET_ID_RL, DATASET_ID_SFT, DEFAULT_DATASET, MODEL_ID, LOCAL_CACHE_DIR

def format_example_for_gemma3(example):
    """
    Takes a single example from the GUI-Libra dataset and formats it 
    into the list of dictionaries expected by Gemma-3 `apply_chat_template`.
    """
    # Extract data (adjust keys based on actual dataset schema)
    # The paper uses images natively but we assume Hugging Face format has 'image' or 'screenshot'
    image = example.get("image") or example.get("screenshot")
    goal = example.get("goal") or example.get("instruction", "")
    reasoning = example.get("reasoning", "")
    action = example.get("action", "")

    # Combine reasoning and action as the model's desired output
    model_response = f"Reasoning: {reasoning} Action: {action}"

    # Build the Gemma-3 multimodal chat format
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
            "content": [
                {"type": "text", "text": model_response}
            ]
        }
    ]
    
    image_bytes = None
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()
    return {
        "messages": messages,
        "image_bytes": image_bytes
    }

def process_dataset(dataset_id=DEFAULT_DATASET, sample_size=None, cache_dir=None, stream=False):
    print(f"Loading dataset: {dataset_id}")
    
    # Load dataset.
    dataset = load_dataset(dataset_id, split="train", streaming=stream)
    
    if sample_size:
        print(f"Sampling {sample_size} examples (streaming: {stream})...")
        if stream:
            # If streaming, take the first N examples efficiently
            dataset = dataset.take(sample_size)
        else:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

    print("Formatting messages...")
    # IterableDatasets (streaming) don't support num_proc in .map()
    formatted_dataset = dataset.map(format_example_for_gemma3, num_proc=None if stream else 16)

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
    tokenized_dataset = formatted_dataset.map(
        tokenize_and_prepare,
        batched=True,
        batch_size=32,
        num_proc=16,
        remove_columns=["messages"]  # keep image_bytes for the collator
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Saving to {cache_dir}")
        tokenized_dataset.save_to_disk(cache_dir)
        print("Done!")
    
    return tokenized_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI-Libra Data Preparation for Gemma-3")
    parser.add_argument("--dataset", type=str, choices=["rl", "sft"], default="rl", help="Which dataset to process (RL or SFT).")
    parser.add_argument("--sample", type=int, help="Number of examples to process. Useful for quick testing or limiting SFT dataset size.")
    parser.add_argument("--stream", action="store_true", help="Use dataset streaming. Highly recommended for the SFT dataset to avoid massive downloads.")
    parser.add_argument("--cache-dir", type=str, default=LOCAL_CACHE_DIR, help="Local directory to save processed data.")
    
    args = parser.parse_args()
    
    dataset_id = DATASET_ID_SFT if args.dataset == "sft" else DATASET_ID_RL
    
    print(f"Starting pipeline for {args.dataset.upper()} dataset. Cache dir: {args.cache_dir}")
    process_dataset(dataset_id=dataset_id, sample_size=args.sample, cache_dir=args.cache_dir, stream=args.stream)
