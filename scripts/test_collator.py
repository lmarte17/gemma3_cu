import torch
from transformers import AutoTokenizer
from collator import ActionAwareDataCollator
from config import MODEL_ID

def run_mask_test():
    print(f"Loading tokenizer {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Failed to load tokenizer (is huggingface-hub installed?): {e}")
        return

    # Simulate an input sequence
    simulated_text = "<start_of_turn>user\n<image>\nGoal: Click the buy button<end_of_turn><start_of_turn>model\nReasoning: I see the buy button at coordinates 500, 200. Action: click(500, 200)<end_of_turn>"
    
    # Tokenize
    inputs = tokenizer(simulated_text, return_tensors="pt")
    
    # Simulate the dataset batch format
    batch = [{
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0]
    }]
    
    # Initialize collator
    collator = ActionAwareDataCollator(tokenizer=tokenizer, action_weight=3.0)
    
    # Run collator
    output = collator(batch)
    loss_weights = output["loss_weights"][0]
    
    print("\nToken Weights Verification:")
    print("-" * 40)
    input_ids = output["input_ids"][0].tolist()
    
    for idx, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id])
        weight = loss_weights[idx].item()
        
        # We want to see normal tokens as 1.0, Action tokens as 3.0
        if weight > 1.0:
            print(f"[{weight:.1f}] {repr(token_str)}")
        else:
            # Print just a few for context
            if idx > len(input_ids) - 20: 
                print(f"[{weight:.1f}] {repr(token_str)}")

    print("-" * 40)
    print("Test Complete.")

if __name__ == "__main__":
    run_mask_test()
