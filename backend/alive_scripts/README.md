# ALIVE-ED — Paper 1 Implementation Guide
## Domain-Adapted LLM + Real-Time Avatar · ACM Multimedia 2026

**Your directory structure this repo expects:**
```
~/islamm11/alive/backend/
├── Lectures/
│   ├── Lecture 1/
│   │   ├── Images/   Slide1.JPG ... Slide50.JPG
│   │   └── Texts/    Slide1.txt ... Slide50.txt
│   ├── Lecture 2/ ... Lecture 23/
├── data_pipeline/      ← scripts go here
├── evaluation/         ← scripts go here
├── configs/            ← alive_lora.yml goes here
├── rag_index_v2.py     ← goes in backend root
└── app.py  tasks.py  llm_client.py  rag_index.py  (your existing files)
```

---

## DECISION: Text-Only vs VLM Fine-Tuning

| Approach | What you train on | Model | Paper 1 or 2? | Effort |
|---|---|---|---|---|
| **Text-only LoRA** | Slide .txt files (23 lectures) | Llama-3.1-8B | **Paper 1** | Low |
| **Text + PDF** | Slide .txt + medical imaging book | Llama-3.1-8B | Paper 1 (bonus) | Medium |
| **VLM (image+text)** | Slide .JPG + .txt pairs | LLaVA / InternVL | **Paper 2** | High |

**Recommendation: Paper 1 uses text-only LoRA.**
You already have 23 lectures × ~45 slides = ~1,035 slide texts.
That gives ~5,000 QA pairs — plenty for LoRA.
The medical imaging book PDF can be parsed and added to the training corpus as a bonus
(~500 extra QA pairs, strengthens the dataset section of your paper).
Save VLM training for Paper 2 where it is the *primary novelty*.

---

## STEP 0 — Copy Files to Your Server

```bash
# On your local machine / this computer:
# Download all files, then scp to server

scp -r alive_scripts/* islamm11@h100-litjan2024:~/islamm11/alive/backend/

# On server — verify files landed:
cd ~/islamm11/alive/backend
ls data_pipeline/
# Expected: verify_data.py  generate_qa.py  clean_and_split.py  setup_paper1.sh
ls configs/
# Expected: alive_lora.yml
ls evaluation/
# Expected: eval_llm.py
ls rag_index_v2.py
# Expected: rag_index_v2.py
```

---

## STEP 1 — GitHub Repository

```bash
cd ~/islamm11/alive/backend

# Initialize git
git init
git remote add origin https://github.com/YOUR_USERNAME/ALIVE-ED.git

# Copy .gitignore from the scripts package
cp alive_scripts/.gitignore .

# Initial commit — backend code only (no models, no large data)
git add app.py tasks.py llm_client.py rag_index.py celery_app.py
git add data_pipeline/ evaluation/ configs/ rag_index_v2.py
git add .gitignore
git commit -m "Initial ALIVE-ED codebase for Paper 1"
git push -u origin main

# Create Paper 1 development branch
git checkout -b paper1-dev
```

---

## STEP 2 — One-Time Environment Setup

```bash
cd ~/islamm11/alive/backend

# Run setup script — creates finetune conda env, installs all deps, runs audit
bash data_pipeline/setup_paper1.sh

# If you prefer manual install:

# Install eval deps in your existing app env:
conda activate sadtalker
pip install bert-score rouge-score sentence-transformers faiss-gpu

# Create separate finetune env for axolotl:
conda create -n finetune python=3.10 -y
conda activate finetune
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install axolotl
```

---

## STEP 3 — Data Audit (verify_data.py)

**What it does:** Walks all 23 `Lectures/Lecture N/` directories,
counts slide text files, image files, checks pairing, reports word counts.

```bash
cd ~/islamm11/alive/backend
conda activate sadtalker
python data_pipeline/verify_data.py
```

**Expected output:**
```
=====================================================================
  ALIVE-ED Data Audit  —  /home/islamm11/islamm11/alive/backend/Lectures
=====================================================================

Found 23 lecture directories

  ✓  Lecture 1      slides= 50  paired= 50  txt_only= 0  img_only= 0
  ✓  Lecture 2      slides= 45  paired= 45  txt_only= 0  img_only= 0
  ...
  ✓  Lecture 23     slides= 42  paired= 42  txt_only= 0  img_only= 0

=====================================================================
  SUMMARY
=====================================================================
  Total lectures:           23
  Total slides (unique):   ~1035
  Fully paired (txt+img):  ~1035
  Word counts per slide:
    Min:     8
    Max:    420
    Median:  95
    Total:  98,325 words across all slides

  Slides usable for QA generation: ~1035
  Estimated QA pairs (5/slide):    ~5,175
    → Train (~80%):  ~4,140
    → Val   (~10%):  ~   517
    → Test  (~10%):  ~   517
=====================================================================
  RECOMMENDATION
  ✓ You have 1035 paired slides — SUFFICIENT for both:
    (A) Text-only fine-tuning  (use Texts/ only, fastest)
    (B) VLM fine-tuning        (use Images/+Texts/ pairs)
  → Paper 1: Use text-only (faster, cleaner ablation)
  → Paper 2: Use image+text pairs (novel VLM contribution)
=====================================================================
```

**Saves:** `data_pipeline/audit_report.json` (full per-slide breakdown)

---

## STEP 4 — QA Pair Generation (generate_qa.py)

**What it does:** Reads every `Lectures/Lecture N/Texts/SlideM.txt`,
calls your running vLLM endpoint, generates 5 QA pairs per slide,
saves to `data/qa_raw/Lecture_N__SlideM.json`.

**Prerequisites:**
- Terminal A must be running vLLM (your Llama-3.1-8B)
- Estimated runtime: ~25-35 minutes for all 23 lectures

```bash
# Make sure vLLM is running (Terminal A):
conda activate vllm_deploy
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 4096

# Run QA generation in tmux (survives SSH disconnect):
tmux new -s qa_gen
cd ~/islamm11/alive/backend
conda activate sadtalker
python data_pipeline/generate_qa.py

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t qa_gen
```

**Monitor progress:**
```bash
# Watch log in real time:
tail -f data_pipeline/qa_gen.log

# Count completed files:
ls data/qa_raw/*.json | wc -l   # should grow toward ~1035

# Check for failures:
ls data/qa_raw/*_FAIL.txt | wc -l   # should be < 20
```

**Expected output files:**
```
data/qa_raw/
  Lecture_1__Slide1.json
  Lecture_1__Slide2.json
  ...
  Lecture_23__Slide42.json
  (total: ~1,035 JSON files)
```

**Each JSON looks like:**
```json
{
  "lecture_name": "Lecture 1",
  "lecture_num": 1,
  "slide_name": "Slide10",
  "slide_num": 10,
  "slide_text": "X-ray computed tomography uses...",
  "pairs": [
    {"question": "What physical principle...", "answer": "...", "difficulty": "basic"},
    {"question": "How does the Hounsfield...", "answer": "...", "difficulty": "intermediate"},
    ...
  ]
}
```

---

## STEP 5 — Clean and Split Dataset (clean_and_split.py)

**What it does:** Loads all raw QA JSON files, filters bad pairs,
deduplicates, splits 80/10/10 by lecture (stratified — every lecture
appears in train, val, and test).

```bash
cd ~/islamm11/alive/backend
conda activate sadtalker
python data_pipeline/clean_and_split.py
```

**Expected output:**
```
Loaded 5,175 raw pairs from data/qa_raw
After filtering: 4,820 / 5,175 records
  Removed   185 — a_too_short
  Removed    92 — meta_question
  Removed    48 — no_question_mark
  Removed    30 — answer_echoes_question
After dedup: 4,650 records (removed 170 duplicates)

Final split:
  Train: 3,720
  Val:     465
  Test:    465
  Total: 4,650

================================================================
  DATASET STATS  (copy to paper Table 3)
================================================================
  Total QA pairs:          4,650
  Train / Val / Test:      3,720 /   465 /   465
  Lectures covered:           23
  Avg question length:       12.3 words
  Avg answer length:         28.7 words
  Avg slide context len:     97.4 words
  Difficulty distribution:
    basic          : 1,860  (40.0%)
    intermediate   : 1,860  (40.0%)
    advanced       :   930  (20.0%)
================================================================
```

**Saves:**
```
data/finetune/
  train.jsonl       ← used by axolotl
  val.jsonl         ← used by axolotl
  test.jsonl        ← Alpaca format
  test_full.jsonl   ← with metadata, used by eval_llm.py
  stats.json        ← dataset statistics for the paper
```

---

## STEP 6 — Rebuild RAG Index (rag_index_v2.py)

**What it does:** Replaces your original `rag_index.py`.
Walks all 23 lectures, embeds slide texts using `all-mpnet-base-v2`
(768-dim, significantly better retrieval than your current `all-MiniLM-L6-v2`),
builds a new FAISS index.

**Key improvement:** `all-mpnet-base-v2` scores 57.6 vs 52.5 NDCG@10
on the BEIR benchmark — ~10% relative retrieval improvement.
This difference shows up directly in your paper's Table 2 (NDCG@k comparison).

```bash
# Run from backend root (not inside data_pipeline/)
cd ~/islamm11/alive/backend
conda activate sadtalker

# This takes ~3-5 minutes to embed all slides on GPU
python rag_index_v2.py
```

**Expected output:**
```
  Lecture 1       : 50 slides loaded
  Lecture 2       : 45 slides loaded
  ...
  Lecture 23      : 42 slides loaded
Total chunks: 1035

Loading embedding model: sentence-transformers/all-mpnet-base-v2
Embedding 1035 chunks ...  100%|████████| 1035/1035
FAISS index built: 1035 vectors × 768 dims

Saved faiss.index   → rag_store/faiss.index
Saved chunks.json   → rag_store/chunks.json  (1035 entries)
Saved meta.json     → rag_store/meta.json

--- Retrieval sanity check ---
Query: What is filtered backprojection in CT imaging?
  [1] score=0.8821  Lecture 5 / Slide23
       Filtered backprojection is a reconstruction algorithm...
  [2] score=0.8644  Lecture 5 / Slide22
  [3] score=0.8312  Lecture 6 / Slide8

Done. RESTART FastAPI server (Terminal B) to load new index.
```

**IMPORTANT — after running this:**
```bash
# Restart Terminal B (FastAPI) to load the new index:
conda activate sadtalker
python -m uvicorn app:app --host 0.0.0.0 --port 9000
```

**Also update app.py** — find the line that hardcodes the embedding model
and replace with the value from meta.json:
```python
# In app.py, change:
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# To:
EMB_MODEL = json.loads((STORE_DIR / "meta.json").read_text())["embedding_model"]
```

---

## STEP 7 — LoRA Fine-Tuning

### 7a. Install axolotl (one-time)

```bash
conda create -n finetune python=3.10 -y
conda activate finetune

pip install torch==2.1.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install packaging ninja
pip install flash-attn --no-build-isolation   # ~5 min compile time
pip install axolotl

# Verify:
python -c "import axolotl; print(axolotl.__version__)"
```

### 7b. Run Training

```bash
# Use GPU 0 (leave GPU 1 for SadTalker/MuseTalk in Terminals C)
tmux new -s finetune
conda activate finetune
cd ~/islamm11/alive/backend

CUDA_VISIBLE_DEVICES=0 axolotl train configs/alive_lora.yml \
  2>&1 | tee logs/finetune.log

# Detach: Ctrl+B then D
# Monitor GPU: watch -n2 nvidia-smi
# Monitor loss: tail -f logs/finetune.log | grep "eval_loss"
```

**Expected timeline on H100:**
```
Epoch 1/3:  ~50 min   eval_loss: ~1.8 → 1.2
Epoch 2/3:  ~50 min   eval_loss: ~1.2 → 0.9
Epoch 3/3:  ~50 min   eval_loss: ~0.9 → 0.75
Total: ~2.5 hours
```

**What to watch for:**
- `eval_loss` should decrease each epoch (if it goes UP, reduce learning_rate to 1e-4)
- GPU utilization should be >85% (if lower, increase micro_batch_size to 4)
- If OOM: reduce `micro_batch_size` to 1, increase `gradient_accumulation_steps` to 16

**Checkpoints saved to:** `checkpoints/alive-llama-lora/`

### 7c. Merge LoRA Weights

```bash
conda activate finetune
cd ~/islamm11/alive/backend

CUDA_VISIBLE_DEVICES=0 axolotl merge-lora configs/alive_lora.yml

# Output: checkpoints/alive-llama-lora/merged/
```

### 7d. Switch vLLM to Fine-Tuned Model

```bash
# Stop current vLLM in Terminal A (Ctrl+C), then restart:
conda activate vllm_deploy
CUDA_VISIBLE_DEVICES=0 vllm serve checkpoints/alive-llama-lora/merged \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85

# Update llm_client.py — change ONE line:
# VLLM_MODEL = "checkpoints/alive-llama-lora/merged"
```

---

## STEP 8 — LLM Evaluation (eval_llm.py)

**Run TWICE** — once with base model, once with fine-tuned model.

### 8a. Install deps

```bash
conda activate sadtalker
pip install bert-score rouge-score transformers
```

### 8b. Evaluate base model first

```bash
# Make sure Terminal A is still serving the BASE model
cd ~/islamm11/alive/backend
conda activate sadtalker
python evaluation/eval_llm.py --model base --out evaluation/results_base.json
```

### 8c. Switch to fine-tuned model (Step 7d above), then:

```bash
python evaluation/eval_llm.py --model finetuned --out evaluation/results_finetuned.json
```

### 8d. Print comparison table

```bash
python evaluation/eval_llm.py --compare
```

**Expected comparison output:**
```
=================================================================
  COMPARISON TABLE  (copy to paper Table 2)
=================================================================
  Metric                    Base LLM     ALIVE-ED        Delta
  ---------------------------------------------------------------
  BERTScore F1                0.7423       0.8156       +0.0733 *
  BERTScore Prec              0.7280       0.8031       +0.0751 *
  BERTScore Rec               0.7581       0.8291       +0.0710 *
  ROUGE-1                     0.3812       0.4923       +0.1111 *
  ROUGE-2                     0.1640       0.2891       +0.1251 *
  ROUGE-L                     0.3201       0.4412       +0.1211 *
  Factual Consistency         61.2%        83.4%       +22.2%

  * indicates improvement > 0.02

  Per-difficulty comparison (BERTScore F1):
    advanced       : base=0.6891  ft=0.7823  delta=+0.0932  (n=93)
    basic          : base=0.7801  ft=0.8412  delta=+0.0611  (n=186)
    intermediate   : base=0.7312  ft=0.8091  delta=+0.0779  (n=186)
```

*These are estimated values — your actual numbers will vary.*

---

## COMPLETE COMMAND SEQUENCE (Quick Reference)

```bash
# ── SSH into server ───────────────────────────────────────────
ssh islamm11@h100-litjan2024

# ── Terminal A: vLLM (base model) ───────────────────────────
conda activate vllm_deploy
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 4096

# ── Terminal B: FastAPI app ───────────────────────────────────
conda activate sadtalker
cd ~/islamm11/alive/backend
python -m uvicorn app:app --host 0.0.0.0 --port 9000

# ── Terminal C: Celery worker ─────────────────────────────────
conda activate sadtalker_rt
CUDA_VISIBLE_DEVICES=1 celery -A celery_app.celery_app worker \
  --loglevel=info --concurrency=1

# ── Terminal D: Cloudflare tunnel ────────────────────────────
cloudflared tunnel --url http://127.0.0.1:9000

# ── New terminal: Paper 1 work ───────────────────────────────
cd ~/islamm11/alive/backend
conda activate sadtalker

# STEP 1 — Audit
python data_pipeline/verify_data.py

# STEP 2 — Generate QA pairs (~30 min, run in tmux)
tmux new -s qa && python data_pipeline/generate_qa.py

# STEP 3 — Clean and split
python data_pipeline/clean_and_split.py

# STEP 4 — Rebuild RAG index (then restart Terminal B)
python rag_index_v2.py

# STEP 5 — Fine-tune (run overnight in tmux)
conda activate finetune
tmux new -s ft && \
  CUDA_VISIBLE_DEVICES=0 axolotl train configs/alive_lora.yml \
  2>&1 | tee logs/finetune.log

# STEP 6 — Merge LoRA
CUDA_VISIBLE_DEVICES=0 axolotl merge-lora configs/alive_lora.yml

# STEP 7 — Evaluate base model (Terminal A still on base)
conda activate sadtalker
python evaluation/eval_llm.py --model base

# STEP 8 — Switch Terminal A to fine-tuned model, then:
python evaluation/eval_llm.py --model finetuned

# STEP 9 — Compare
python evaluation/eval_llm.py --compare
```

---

## File Reference

| File | Location | When to run |
|---|---|---|
| `verify_data.py` | `data_pipeline/` | Once — before anything else |
| `setup_paper1.sh` | `data_pipeline/` | Once — installs all deps |
| `generate_qa.py` | `data_pipeline/` | Once — creates training data |
| `clean_and_split.py` | `data_pipeline/` | Once — after generate_qa |
| `rag_index_v2.py` | `backend/` root | Once — rebuilds FAISS index |
| `alive_lora.yml` | `configs/` | Config for axolotl training |
| `eval_llm.py` | `evaluation/` | Twice — base then fine-tuned |
