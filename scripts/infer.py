import argparse
import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

from config import MODEL_ID

def parse_action(text):
    """
    Utility to extract the action and coordinates from the model's raw text output.
    Expected format is 'Reasoning: ... Action: click(x, y)'
    """
    action_match = re.search(r"Action:\s*([a-zA-Z_]+)\((.*?)\)", text)
    if action_match:
        action_type = action_match.group(1)
        args_str = action_match.group(2)
        # Try to parse coordinates safely
        args = [arg.strip() for arg in args_str.split(',')]
        return {
            "type": action_type,
            "args": args
        }
    return None

def infer(args):
    print("--- GUI-Libra Inference (Apple Silicon / MPS) ---")
    
    # 1. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16 # MPS prefers float16 over bfloat16 currently
        print("✅ MPS device detected. Acceleration enabled.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        print("✅ CUDA device detected.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️ No hardware acceleration found. Running on CPU will be extremely slow.")

    # 2. Loading Base Model
    print(f"Loading Base Model ({MODEL_ID})...")
    # For a 36GB Mac, a 4B parameter model in float16 takes ~8-10GB.
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    # 3. Apply PEFT/LoRA (If provided)
    if args.lora_path:
        print(f"Applying LoRA weights from {args.lora_path}...")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        print("⚠️ No LoRA path provided. Running the base model (will likely fail to output GUI actions).")
        model = base_model
        
    model.eval()

    # 4. Prepare Context
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to load image {args.image}: {e}")
        return

    # Gemma-3 Multimodal format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Goal: {args.goal}"}
            ]
        }
    ]

    print("Formatting prompt and preparing tensors...")
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device, dtype) # Move to MPS and convert image pixels to float16

    # 5. Generation
    # Since we want precise format ("Reasoning: ... Action: ..."), a lower temperature is usually better
    print(f"Generating action for goal: '{args.goal}'...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1
        )
        
    # We only want to decode the *newly generated* tokens, not the prompt.
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0][input_len:]
    raw_output = processor.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "="*40 + "\n")
    print("MODEL RAW OUTPUT:")
    print(raw_output)
    print("\n" + "="*40 + "\n")

    # 6. Parse Action Action 
    parsed_action = parse_action(raw_output)
    if parsed_action:
        print("✅ Parsed Action:")
        print(f"  Type: {parsed_action['type']}")
        print(f"  Coordinates/Args: {parsed_action['args']}")
    else:
        print("❌ Failed to parse a valid action from the output. Model may not be fine-tuned correctly or output format drifted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI-Libra Local Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to the screenshot.")
    parser.add_argument("--goal", type=str, required=True, help="The instruction/goal for the agent.")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to the trained LoRA adapter directory.")
    
    args = parser.parse_args()
    infer(args)
