# GUI-Libra: Full Training & Inference Guide on Google Cloud Compute Engine

This guide walks you through the entire end-to-end process of fine-tuning the `Gemma-3-4b-it` model on the GUI-Libra dataset using a single NVIDIA L4 GPU on Google Cloud, and then performing local inference on your MacBook Pro using Apple Silicon (MPS).

---

## Part 1: Setting up the GCP Compute Engine VM

We will use a deep learning VM image provided by Google, which comes pre-installed with PyTorch and standard GPU drivers.

1. **Log into Google Cloud Console** and navigate to **Compute Engine > VM instances**.
2. Click **Create Instance**.
3. **Machine Configuration**:
   - Name: `gui-libra-trainer`
   - Region: Choose one with L4 availability (e.g., `us-central1-a`).
   - Machine Family: `GPU`
   - GPU Type: `NVIDIA L4`
   - Number of GPUs: `1`
   - Machine Type: `g2-standard-4` (4 vCPU, 16 GB RAM) or `g2-standard-8` (8 vCPU, 32 GB RAM recommended for data loading).
4. **Boot Disk**:
   - Click **Change**.
   - OS: `Deep Learning on Linux`
   - Version: Select the latest `PyTorch` image (e.g., `PyTorch 2.1 with CUDA 12.1 M110`).
   - Size:
     - **Option A (HuggingFace streaming):** `100 GB` is sufficient — no images are stored locally.
     - **Option B (local web-subset):** At least `300 GB` — the guiact-web + mind2web archives are ~85 GB downloaded, ~100–200 GB extracted, plus model weights.
5. **Firewall/Networking**: Standard settings are fine.
6. Click **Create**.
7. _Ensure you stop the VM when not in use to avoid hourly GPU charges._

---

## Part 2: Transferring Code & Setting Up the Environment

The recommended way to keep your local MacBook and the GCP VM aligned is by using Git (e.g., GitHub, GitLab). This prevents you from needing to constantly copy files over manually.

1. **Commit and Push your local code to GitHub**:
   On your MacBook, commit and push your code to your remote repository.

   ```bash
   git add .
   git commit -m "Sync latest training scripts"
   git push origin main
   ```

2. **SSH into the VM**:

   ```bash
   gcloud compute ssh your-gcp-username@gui-libra-trainer
   ```

3. **Clone or Pull the code on the VM**:
   If this is your first time on the VM, clone the repository:

   ```bash
   git clone https://github.com/your-username/gui-libra.git
   cd gui-libra/scripts
   ```

   If you have already cloned it previously, just pull the latest changes:

   ```bash
   cd ~/gui-libra/scripts
   git pull origin main
   ```

   _(Note: Ensure you are always inside the `scripts/` directory before running python commands because the modules import each other using relative paths)._

4. **Install Training Dependencies**:
   _(The Deep Learning image has PyTorch, but we need the HF ecosystem.)_

   ```bash
   pip install -U transformers datasets bitsandbytes peft trl accelerate tensorboard
   ```

5. **Authenticate with Hugging Face**:
   Gemma-3 is a gated model. You must accept the terms on the Hugging Face website and generate an access token.
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```

6. **Pre-download the Gemma-3-4B-IT base model**:
   The training and inference scripts load the model via `from_pretrained("google/gemma-3-4b-it")`, which downloads it on first use and caches it in `~/.cache/huggingface/`. Do this explicitly now so that:
   - Your HF token is validated against the gated model before a long training run starts.
   - The ~9 GB download completes in the foreground where you can see progress and errors.
   - Training begins immediately when you launch it, with no hidden download delay.

   ```bash
   huggingface-cli download google/gemma-3-4b-it
   ```

---

## Part 3: Running the Two-Stage Training (The Cloud GPU)

The GUI-Libra methodology requires a two-stage process: first, training on the large SFT dataset to learn general GUI interactions, and second, training on the RL (Reinforcement Learning / high quality) dataset to refine the reasoning and action generation.

Training can take hours. It is critical to run it inside a `tmux` session so that if your SSH connection drops or your laptop goes to sleep, the training continues running on the GCP server!

1. **Start a background session**:

   ```bash
   tmux new -s training
   ```

2. **Stage 1: Supervised Fine-Tuning (SFT)**:

   **Option A — Full HuggingFace streaming** (downloads the entire 100GB+ dataset on the fly):

   ```bash
   python train.py --dataset sft --stream --batch-size 2 --grad-accum 8 --output-dir ./gemma3-gui-libra-lora-sft
   ```

   **Option B — Local web-subset only** (~85 GB download; requires ≥300 GB disk total):

   > **Why not `git clone`?** The full repo is 163 GB. We use `huggingface-cli download --include` to pull only the two image archives and the 8 annotation files we care about.

   **Step 1 — Download only the web-subset files** (from the VM home directory `~`):

   ```bash
   huggingface-cli download GUI-Libra/GUI-Libra-81K-SFT \
     --repo-type dataset \
     --include "data/images/guiact-web.tar.gz.part-*" \
     --include "data/images/mind2web.tar.gz.part-*" \
     --include "data/annotations/guiact-web-*" \
     --include "data/annotations/mind2web-*" \
     --local-dir ~/GUI-Libra-web-subset
   ```

   **Step 2 — Stream-extract guiact-web, then delete its parts to free space:**
   _(Piping directly into tar avoids writing a merged `.tar.gz` to disk, cutting peak usage by ~84 GB.)_

   ```bash
   cd ~/GUI-Libra-web-subset/data
   mkdir -p images_extracted/guiact-web images_extracted/mind2web

   # guiact-web: stream-extract (two parts, ~84 GB compressed)
   cat images/guiact-web.tar.gz.part-* | tar -xz -C images_extracted/guiact-web/
   rm images/guiact-web.tar.gz.part-*    # free ~84 GB before extracting mind2web

   # mind2web: stream-extract (one part, ~1.2 GB compressed)
   cat images/mind2web.tar.gz.part-* | tar -xz -C images_extracted/mind2web/
   rm images/mind2web.tar.gz.part-*
   ```

   **Step 3 — Verify the JSON schema** (run once before training):

   ```bash
   cd ~/gui-libra/scripts
   python data_prep_local.py --data-dir ~/GUI-Libra-web-subset/data --peek
   ```

   **Step 4 — Run Stage 1 training on the web subset:**

   ```bash
   python train.py --local-data-dir ~/GUI-Libra-web-subset/data --batch-size 2 --grad-accum 8 --output-dir ./gemma3-gui-libra-lora-sft
   ```

   > **Just want a quick pipeline test?** Use `mind2web` alone first (1.2 GB download, ~5 GB extracted, fits on a 100 GB disk):
   > ```bash
   > huggingface-cli download GUI-Libra/GUI-Libra-81K-SFT \
   >   --repo-type dataset \
   >   --include "data/images/mind2web.tar.gz.part-*" \
   >   --include "data/annotations/mind2web-*" \
   >   --local-dir ~/GUI-Libra-web-subset
   > ```
   > Then run the same extraction and training commands above — `data_prep_local.py` will automatically pick up only the mind2web files.

3. **Stage 2: Reinforcement Learning (RL) Fine-Tuning**:
   Once Stage 1 completes, run the script again on the RL dataset. The `--resume-from-checkpoint` flag loads the Stage 1 LoRA adapters so training continues from where it left off rather than starting from scratch.

   ```bash
   python train.py --dataset rl --stream --batch-size 2 --grad-accum 8 --resume-from-checkpoint ./gemma3-gui-libra-lora-sft --output-dir ./gemma3-gui-libra-lora-final
   ```

4. **Monitoring**:
   - Ensure the output prints `Loading Base Model...` followed by the LoRA parameters summary.
   - You can detach from the `tmux` session by pressing `Ctrl+B`, then immediately pressing `D`.
   - To check back later, SSH into the machine and run `tmux attach -t training`.

5. **Save Outputs**:
   When training finishes, the final LoRA adapter weights will be saved in `./gemma3-gui-libra-lora-final`.

---

## Part 4: Transferring Weights Back to Mac

Once training is complete, the LoRA adapters (which are very small, usually ~50-100MB) need to be brought back to your MacBook for inference.

1. **Archive the weights on the VM** _(run this from inside `~/scripts/`)_:

   ```bash
   tar -czvf lora-weights.tar.gz gemma3-gui-libra-lora-final/
   ```

2. **Download to your MacBook**:
   _(Open a terminal on your Mac, NOT the VM)_

   ```bash
   cd /Users/lmarte/Documents/Projects/gui-libra
   gcloud compute scp your-gcp-username@gui-libra-trainer:~/scripts/lora-weights.tar.gz .
   tar -xzvf lora-weights.tar.gz
   ```

3. **STOP THE VM**:
   Go to the GCP Console and click **Stop** on the instance so you don't get billed thousands of dollars for idle GPU time!

---

## Part 5: Local Inference (Apple Silicon MPS)

Now run the fine-tuned model directly on your MacBook. The `infer.py` script automatically detects MPS acceleration.

1. **Install Local Dependencies**:

   ```bash
   pip install torch torchvision peft transformers Pillow
   ```

2. **Take a Test Screenshot**:
   Save a screenshot named `test.png` in your project folder. Ensure it contains the UI element you want to test.

3. **Run Inference** _(from inside the `scripts/` directory)_:

   ```bash
   cd /Users/lmarte/Documents/Projects/gui-libra/scripts
   python infer.py --image ../test.png --goal "Click the submit button" --lora-path ../gemma3-gui-libra-lora-final
   ```

4. **Expected Output**:
   The script will load the base Gemma-3-4b-it model into your MacBook's unified memory, inject the LoRA weights, and output the coordinate prediction.

   ```
   ✅ MPS device detected. Acceleration enabled.
   Loading Base Model (google/gemma-3-4b-it)...
   Applying LoRA weights from ../gemma3-gui-libra-lora-final...
   Generating action for goal: 'Click the submit button'...

   ========================================
   MODEL RAW OUTPUT:
   Reasoning: I see the submit button located at the bottom right. Action: click(850, 920)
   ========================================

   ✅ Parsed Action:
     Type: click
     Coordinates/Args: ['850', '920']
   ```
