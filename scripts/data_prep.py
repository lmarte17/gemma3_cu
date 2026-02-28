import os
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
    
    return {
        "messages": messages,
        "image": image # Keep the image object around for the processor
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
    formatted_dataset = dataset.map(format_example_for_gemma3, num_proc=None if stream else 4)

    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def tokenize_and_prepare(batch):
        # apply_chat_template takes the list of messages and directly generates the input text string
        # with all the necessary <start_of_turn> control tokens.
        
        # We process the batch one by one for clarity, though processor can handle lists.
        # Gemma3Processor takes 'images' and 'text'.
        
        texts = [
             processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
             for msg in batch["messages"]
        ]
        
        # Pass texts and images to processor
        inputs = processor(
            text=texts,
            images=batch["image"],
            return_tensors="pt",
            padding="max_length", # We'll pad dynamically in the collator usually, but here for demo
            truncation=True,
            max_length=2048 # Keep this reasonable to prevent OOM
        )
        
        # We return the tokens as lists for HF Datasets compatibility.
        # pixel_values MUST be included so the vision encoder receives the images during training.
        result = {
            "input_ids": inputs["input_ids"].numpy().tolist(),
            "attention_mask": inputs["attention_mask"].numpy().tolist(),
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].numpy().tolist()
        return result

    print("Tokenizing dataset...")
    # NOTE: In a real robust pipeline, image processing (pixel_values) is often done on the fly
    # in the DataCollator to save massive disk space. For Phase 1, we will just format.
    tokenized_dataset = formatted_dataset.map(
        tokenize_and_prepare, 
        batched=True, 
        batch_size=8,
        remove_columns=formatted_dataset.column_names # Remove original columns to save space
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
