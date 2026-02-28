**Project Goal:** Replicate the "GUI-Libra" training recipe using **Gemma-3-4B-it** as the base model. The objective is to create a "Native GUI Agent" that can reason about a screen and output precise click/type actions.

**Technical Stack:**

- **Base Model:** `google/gemma-3-4b-it` (Multimodal VLM).
- **Training Platform:** Lightning AI (using L4 or A100 GPUs via `pytorch-lightning` or `lit-gpt`).
- **Inference Target:** Local MacBook Pro (36GB RAM) using MPS acceleration.
- **Dataset:** `GUI-Libra/GUI-Libra-81K-RL` (Hugging Face).

**Core Methodology to Replicate (from the GUI-Libra Paper):**

1. **ASFT (Action-aware Supervised Fine-Tuning):** We are not just doing standard SFT. We need to implement a "token reweighting" strategy.

- _The Logic:_ The loss for tokens representing **actions** (e.g., `click`, `write`) and **coordinates** (e.g., `(x, y)`) must be weighted higher than the loss for reasoning/thought tokens. This prevents the "reasoning vs. grounding" trade-off described in the paper.

2. **Multimodal Prompting:** Gemma-3 uses a specific chat template. We need to format the GUI-Libra data (which includes screenshots and reasoning traces) into the `<start_of_turn>user\n<image>\nGoal: {goal}<end_of_turn>` format.
3. **Coordinate System:** GUI-Libra uses a normalized `0-1000` scale. We must ensure the Gemma-3 tokenizer handles these numeric coordinates correctly and that our preprocessing scales all screenshots to match.

**What I need from you (The Coding Agent):**

1. **Data Engineering:** Write a script using the `datasets` library to pull `GUI-Libra-81K-RL`, parse the JSON reasoning traces, and prepare the image-text pairs for the Gemma-3 processor.
2. **Training Script:** Build a Lightning AI-compatible training script.

- Incorporate `SFTTrainer` from the `trl` library.
- Help me implement a custom `compute_loss` or `DataCollator` that applies the **Action-Aware** weight multiplier (suggested 2.0x to 5.0x) to action-related tokens.

3. **Inference Script:** Once we have our LoRA adapters or full weights, write an optimized inference script for my MacBook that uses `torch.device("mps")` and handles screenshot-to-action loops.

**Current Constraints:**

- We want to use **LoRA** (Parameter Efficient Fine-Tuning) to keep the training memory footprint manageable on the Lightning Studio.
- The final model must output the reasoning followed by the action in this format: `Reasoning: ... Action: click(x, y)`.

---

### Key Technical Details for You (The Human)

- **Getting the Data**: The dataset is live on Hugging Face as [`GUI-Libra/GUI-Libra-81K-RL`](<https://www.google.com/search?q=%5Bhttps://huggingface.co/datasets/GUI-Libra/GUI-Libra-81K-RL%5D(https://huggingface.co/datasets/GUI-Libra/GUI-Libra-81K-RL)>). It contains the "Chain of Thought" reasoning steps that make the model smart.
- **The "Secret Sauce" Code**: When you the training code, make sure it looks for specific strings like `click(`, `write(`, and the numeric coordinates in the labels. It should apply a higher loss to these tokens. This is the **ASFT** (Action-aware Supervised Fine-Tuning) mentioned in the paper—it’s what makes the model actually "hit" the buttons instead of just talking about them.
- **Lightning AI Tip**: Since you are using a MacBook for your IDE but Lightning for training, use the **Lightning CLI** to sync your local code to the cloud Studio. Your agent can help you write the `cloud.yaml` configuration for this.
