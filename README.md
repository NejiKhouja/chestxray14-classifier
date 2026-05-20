# ChestX-ray14 Multi-label Classifier

Fine-tuning **DenseNet-121** on the NIH ChestX-ray14 dataset for multi-label classification of 15 chest pathologies, with a multi-modal architecture that fuses image features with patient clinical data (age + gender).

> **Research/educational project — not for clinical use.**

---

## Features

| Feature | Details |
|---|---|
| **Model** | DenseNet-121 backbone + clinical branch (age, gender) |
| **Classes** | 15 chest pathologies (Atelectasis, Cardiomegaly, …, Pneumothorax) |
| **Demo app** | Streamlit — upload X-ray → predictions + GradCAM + LLM explanation |
| **LLM** | Groq (`llama-3.3-70b-versatile`) explains predictions in plain language |
| **Evaluation** | Jupyter notebook — AUC-ROC per class, GradCAM visualization |

---

## Project Structure

```
ChestX-ray14/
├── src/
│   ├── model.py          # DenseNet-121 + clinical fusion architecture
│   ├── dataset.py        # Dataset class, transforms, label definitions
│   └── train.py          # Training loop (mixed precision, checkpointing)
├── app/
│   ├── app.py            # Streamlit demo app
│   └── utils.py          # Shared helpers: load_model, predict, GradCAM, Groq
├── notebooks/
│   ├── exploration.ipynb # EDA, preprocessing, class weight calculation
│   └── evaluate.ipynb    # Checkpoint evaluation, AUC-ROC, GradCAM, LLM
├── outputs/
│   └── checkpoints/
│       └── model_epoch5.pth
├── config.yaml           # Hyperparameters and paths
├── requirements.txt
└── .env.example          # Copy to .env and add GROQ_API_KEY
```

---

## Architecture

```
Image (224×224) ──► DenseNet-121 features ──► GAP ──► 1024-dim
                                                              │
Age, Gender ──────► FC(2→16) → ReLU → Dropout ──► 16-dim   │
                                                              ▼
                                              Concat → FC(1040→512) → FC(512→15)
```

The final 15 logits are passed through sigmoid for multi-label prediction.  
Sigmoid (not softmax) is used because a single image can show multiple conditions simultaneously — each class is treated as an independent binary prediction.

---

## The ML Pipeline

The project follows a linear pipeline from raw data to interactive demo. Each stage feeds into the next.

```
Raw Dataset
    │
    ▼
[exploration.ipynb]  ←── EDA, clean age outliers, encode gender,
    │                     normalize age, split train/test, save CSVs
    ▼
train.csv / test.csv
    │
    ▼
[src/train.py]       ←── load data, fine-tune DenseNet-121,
    │                     save checkpoint every 5 epochs
    ▼
outputs/checkpoints/model_epoch5.pth
    │
    ├──► [notebooks/evaluate.ipynb]  ←── load checkpoint, compute AUC-ROC,
    │                                     visualize GradCAM, call Groq
    │
    └──► [app/app.py]                ←── interactive Streamlit demo,
                                          upload image → predictions → LLM
```

**Why this structure?**  
Each stage is isolated — you can re-run just training without touching the app, or update the app logic without retraining. The `config.yaml` holds all shared paths and hyperparameters so nothing is hardcoded in multiple places.

### Stage 1 — Preprocessing (`exploration.ipynb`)

The raw CSV from NIH has two issues that must be fixed before training:

- **Age outliers**: a few entries have impossible ages (e.g., age > 120). These are dropped.
- **Gender as string**: the model expects a float tensor. `M → 0`, `F → 1`.
- **Age scale**: raw ages (0–100+) fed directly into a neural network create unstable gradients because they dwarf all other activations. Dividing by 100 puts age in the same `[0, 1]` range as normalized image pixels.

The notebook then splits images into train/test using the official NIH split files (`train_val_list.txt`, `test_list.txt`) and saves clean CSVs.

### Stage 2 — Training (`src/train.py`)

Three techniques handle the core challenges of this dataset:

**Class imbalance** — "Hernia" appears in only 141 of 86,512 training images (0.16%). Without correction, the model would just never predict it. `BCEWithLogitsLoss(pos_weight=...)` scales the loss for positive samples of each class — rare diseases contribute more to the gradient. Weights are capped at 50 to prevent extreme classes from dominating training.

**Compute efficiency** — Mixed precision training (`torch.cuda.amp`) stores activations in float16 instead of float32, roughly halving GPU memory use and speeding up training on modern GPUs, with no meaningful accuracy loss.

**Checkpointing** — A full DenseNet-121 training run takes hours. Saving every 5 epochs means if training is interrupted, it can resume from the last checkpoint rather than starting over.

### Stage 3 — Evaluation (`notebooks/evaluate.ipynb`)

**AUC-ROC** is the standard metric for this dataset because it measures ranking quality regardless of threshold — it answers "does the model rank sick patients higher than healthy ones?" rather than "did it guess right at 50%?". A score of 0.5 means random; published DenseNet baselines on this dataset reach ~0.80 mean AUC with full training.

The notebook also runs **GradCAM** (see below) on a sample image so you can visually sanity-check whether the model is looking at the right parts of the image.

---

## GradCAM — How It Works

GradCAM (Gradient-weighted Class Activation Mapping) answers: *"which pixels made the model predict this disease?"*

**The idea in plain terms:** the model's last convolutional layer produces a set of feature maps — each one detecting a different visual pattern. GradCAM asks "which of these feature maps mattered most for predicting class X?" by computing how much the class score changes when each feature map changes (i.e., the gradient). It then blends those feature maps together weighted by their importance, producing a single heatmap.

**In this project:**  
The target layer is `model.features.denseblock4` — the last dense block before global average pooling, where the feature maps are still spatially meaningful (7×7 grid over the original 224×224 image). After that, GAP collapses spatial information, so GradCAM would have nothing to look at.

```python
# simplified flow
output[0, class_idx].backward()         # compute gradients for one class
weights = gradients.mean(over H and W)  # importance of each feature map channel
cam     = relu(sum(weights * activations))  # weighted blend, ignore negatives
```

The resulting heatmap is resized back to 224×224 and overlaid on the original image. Red regions drove the prediction; blue regions were irrelevant.

**Why include it?**  
In medical imaging, a model that gives the right answer for the wrong reason is dangerous. GradCAM lets you check that the model is actually looking at the lung fields and not at artifacts like scanner labels or patient markers.

---

## LLM Integration — How and Why

### How it works

After the model produces a probability vector (15 floats, one per disease), that vector is formatted into a structured text prompt and sent to Groq's API. Groq runs `llama-3.3-70b-versatile` — a large language model — which reads the probabilities and writes a short plain-language interpretation.

```
Model output (15 floats)
        │
        ▼
[app/utils.py — explain_with_groq()]
        │  builds a prompt like:
        │  "Findings above 50%: Infiltration (67%), Atelectasis (54%)
        │   All probabilities: ...
        │   Explain in 2-3 sentences for a medical student."
        │
        ▼
Groq API  (llama-3.3-70b-versatile, max 220 tokens)
        │
        ▼
Plain-language explanation shown in Streamlit / notebook
```

### Why Groq specifically?

Groq runs LLMs on custom hardware (LPUs — Language Processing Units) that are significantly faster than GPU-based inference. Response times are typically under 1 second, which is important for a real-time demo app — waiting 10 seconds for an explanation after already waiting for model inference would break the user experience.

The `llama-3.3-70b-versatile` model was chosen because it has strong reasoning and medical knowledge from its training data, while still being fast enough on Groq hardware.

### Why add an LLM at all?

A raw probability vector like `[0.12, 0.03, 0.67, ...]` is not useful to most people. The LLM bridges the gap between the model's numerical output and a human-readable interpretation. This pattern — a task-specific model producing structured output, an LLM turning it into natural language — is increasingly common in applied AI systems.

It also makes the project more honest: the LLM is explicitly prompted to remind the reader that the model is partially trained and not clinically valid. The explanation is interpretive, not diagnostic.

**Important:** the LLM does not have access to the image. It only sees the probability scores. It cannot "look" at the X-ray — it is purely translating numbers into words.

---

## Setup

```bash
# 1. install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # GPU
pip install -r requirements.txt

# 2. add your Groq API key
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...

# 3. download the dataset
# https://nihcc.app.box.com/v/ChestXray-NIHCC
# place images in data/row/archive/images_001/ … images_012/
# place Data_Entry_2017.csv, train_val_list.txt, test_list.txt in data/row/archive/

# 4. run data preprocessing
jupyter notebook notebooks/exploration.ipynb
```

---

## Training

```bash
cd src
python train.py
```

Checkpoints are saved every 5 epochs to `outputs/checkpoints/`.

---

## Streamlit Demo

```bash
streamlit run app/app.py
```

1. Upload a chest X-ray image (PNG or JPEG)
2. Enter patient age and gender
3. Click **Analyze**
4. View GradCAM heatmap, probability chart, and Groq LLM explanation

---

## Evaluation Notebook

```bash
jupyter notebook notebooks/evaluate.ipynb
```

- Loads the saved checkpoint
- Runs batch inference on the test set (or a subset via `MAX_SAMPLES`)
- Computes per-class AUC-ROC
- Visualizes GradCAM for a sample image
- Gets a plain-language explanation from Groq

---

## Dataset

NIH Clinical Center — ChestX-ray14  
112,120 frontal-view X-ray images from 30,805 unique patients  
Source: <https://nihcc.app.box.com/v/ChestXray-NIHCC>

**Encoding used in preprocessing (`exploration.ipynb`):**
- `Patient Gender`: M → 0, F → 1
- `Patient Age`: raw_age / 100.0
