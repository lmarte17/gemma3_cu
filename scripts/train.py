import os
import argparse
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import torch.nn as nn

# Import our custom modules from Phase 1
from config import MODEL_ID, DATASET_ID_SFT, DATASET_ID_RL, DEFAULT_DATASET
from data_prep import process_dataset
from collator import ActionAwareDataCollator

class ASFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Remove loss_weights before passing to the model
        loss_weights = inputs.pop("loss_weights", None)
        
        outputs = model(**inputs)
        
        # Extract logits and labels
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if loss_weights is not None and logits is not None and labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_loss_weights = loss_weights[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Apply our custom weights
            loss = loss * shift_loss_weights.view(-1)
            
            # Normalize by the number of non-ignored tokens
            mask = (shift_labels.view(-1) != -100).float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            # Fallback to standard loss if no weights or missing logits/labels
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def train(args):
    dataset_id = DATASET_ID_SFT if args.dataset == "sft" else DATASET_ID_RL
    print(f"--- Starting GUI-Libra Replication Pipeline ({dataset_id}) ---")

    # 1. Load the Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 2. Configure Quantization (for single GPU / Lightning Studio L4/A100)
    # Using 4-bit to save max memory, adjust to 8-bit if needed.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model: {MODEL_ID} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True # Often needed for newest models
    )
    
    # 3. Apply PEFT/LoRA
    model = prepare_model_for_kbit_training(model)
    
    if args.resume_from_checkpoint:
        print(f"Loading existing LoRA adapters from {args.resume_from_checkpoint}...")
        # Load the saved LoRA weights into the base model
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        print("Initializing fresh LoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
    print_trainable_parameters(model)

    # 4. Prepare Dataset
    sample_size = 10 if args.test_mode else args.sample

    if args.local_data_dir:
        # Local web-subset path: uses downloaded guiact-web + mind2web files.
        # The processor is already loaded above, so we pass it in to avoid reloading.
        print(f"Preparing local dataset from {args.local_data_dir}...")
        from data_prep_local import load_local_web_dataset
        dataset = load_local_web_dataset(
            data_dir=args.local_data_dir,
            processor=processor,
            sample_size=sample_size,
        )
    else:
        # HuggingFace streaming path (default)
        print("Preparing dataset (this might take a moment if not cached/streaming)...")
        dataset = process_dataset(
            dataset_id=dataset_id,
            sample_size=sample_size,
            stream=args.stream,
        )

    # 5. Initialize our Custom ASFT Collator
    collator = ActionAwareDataCollator(
        tokenizer=processor,
        action_weight=args.asft_weight # Usually 2.0 - 5.0
    )

    # 6. Setup Training Arguments
    warmup_steps = int(0.03 * args.max_steps) if args.max_steps else 100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # Recommended for newer GPUs (A100/L4)
        max_grad_norm=0.3, # crucial for stability with custom loss weights
        max_steps=args.max_steps if args.max_steps else -1,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        dataloader_num_workers=8,
        report_to="tensorboard" # Or wandb if configured
    )

    # 7. Initialize SFTTrainer
    # max_seq_length removed from SFTTrainer in newer TRL — truncation handled in data prep
    trainer = ASFTTrainer(
        model=model,
        train_dataset=dataset,
        # LoRA is always applied manually above (either fresh or resumed), so we never
        # let SFTTrainer rewrap the model. Always pass None here.
        peft_config=None,
        tokenizer=processor,
        args=training_args,
        data_collator=collator # Inject our ASFT magic
    )

    if args.test_mode:
        print("\n--- TEST MODE ---")
        print("Architecture loaded successfully. Skipping actual training loop.")
        # Optionally grab one batch from the dataloader to verify collator doesn't crash
        train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(train_dataloader))
        print("Successfully pulled one batch from ActionAwareDataCollator.")
        print("Batch keys:", batch.keys())
        return

    # 8. Train
    print("Starting training loop...")
    trainer.train()

    # 9. Save
    print(f"Saving finalized LoRA weights to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["rl", "sft"], default="rl")
    parser.add_argument("--stream", action="store_true", help="Stream the dataset to save memory/disk.")
    parser.add_argument("--sample", type=int, default=None, help="Limit dataset size.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--asft-weight", type=float, default=3.0, help="Weight multiplier for Action tokens.")
    parser.add_argument("--output-dir", type=str, default="./gemma3-gui-libra-lora")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a previously trained LoRA adapter directory to continue training.")
    parser.add_argument("--local-data-dir", type=str, default=None, help="Use locally downloaded web-subset data (guiact-web + mind2web) instead of HuggingFace streaming. Point to the root data dir containing annotations/ and images/.")
    parser.add_argument("--test-mode", action="store_true", help="Load architecture and single batch, then exit.")
    
    args = parser.parse_args()
    train(args)
