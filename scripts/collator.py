import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import PreTrainedTokenizerBase

@dataclass
class ActionAwareDataCollator:
    tokenizer: PreTrainedTokenizerBase
    action_weight: float = 3.0
    action_keyword: str = "Action:"
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Handle cases where input is already tensors or lists
        input_ids = [torch.tensor(f["input_ids"]) if isinstance(f["input_ids"], list) else f["input_ids"] for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) if isinstance(f["attention_mask"], list) else f["attention_mask"] for f in features]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # Standard causal LM labels (shifted inside the model usually, but we provide input_ids as labels)
        # We replace padding token id's in the labels by -100 so they are ignored by the loss function
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Initialize loss weights (1.0 for standard tokens)
        loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
        
        # We need to find the token sequence that corresponds to "Action:"
        # Often it's easier to find the onset of the action part by decoding or searching for specific tokens.
        # Let's get the token IDs for the action keyword.
        # Note: Depending on the tokenizer, " Action:" might be a single token or multiple.
        # A robust way is to decode the input ids up to the current index and check.
        # But for efficiency in the collator, it's better to pre-find these indices or use a heuristic.
        
        # For this ASFT implementation, any token that is part of the final action prediction gets upweighted.
        # We will iterate through each sequence in the batch.
        for i, seq in enumerate(input_ids):
            # Convert sequence to list for easier processing
            seq_list = seq.tolist()
            # Decode the sequence (warning: can be slow for large batches, but safe)
            decoded_seq = self.tokenizer.decode(seq_list, skip_special_tokens=False)
            
            # Find where the action keyword starts
            # Since we formatted as "Reasoning: {reasoning} Action: {action}", 
            # we can look for "Action:".
            idx = decoded_seq.rfind(self.action_keyword)
            
            if idx != -1:
                # We need to map string index back to token index
                # A heuristic: encode the prefix up to idx to get its length in tokens
                # We limit the token_start_idx to avoid an out-of-bounds error
                prefix = decoded_seq[:idx]
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                token_start_idx = min(len(prefix_tokens), len(loss_weights[i]) - 1)
                
                # Apply the multiplier to everything from the action keyword onwards
                if token_start_idx < len(loss_weights[i]):
                    loss_weights[i, token_start_idx:] = self.action_weight
                
            # Ensure padding tokens have 0 weight (though labels=-100 handles the loss ignoring anyway)
            loss_weights[i][attention_mask[i] == 0] = 0.0

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights
        }
        
        # Keep pixel values if they are present
        if "pixel_values" in features[0]:
             batch["pixel_values"] = torch.stack([torch.tensor(f["pixel_values"]) if isinstance(f["pixel_values"], list) else f["pixel_values"] for f in features])
             
        return batch
