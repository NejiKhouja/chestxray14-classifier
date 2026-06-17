# ChestX-ray14 Multi-label Classifier

Fine-tuning **DenseNet-121** on the NIH ChestX-ray14 dataset for multi-label classification of **14 chest pathologies**, with a multi-modal architecture that fuses image features with patient clinical data (age + gender).

> **Research/educational project — not for clinical use.**

---

## What's the approach?

Two design choices drive better, cleaner results:

1. **14 pathology classes, not 15.** "No Finding" is *not* a trained class — it competes with real pathologies and (at 54% of images) drowns the rare ones. Instead, **a healthy scan is the absence of all 14 findings**, and we report a derived verdict:

   > **`No Finding / Healthy score = 1 − max(pathology probabilities)`**

   So the model still answers *"is this chest X-ray healthy?"* — it just does it the principled way that matches published CheXNet baselines.

2. **Undersample the majority.** 53.8% of images are healthy, which swamps the learning signal. Preprocessing **undersamples No-Finding images in the training set only** (keeping ~0.5 healthy per finding image), dropping healthy share from ~58% → ~33%. Validation and test keep their natural distribution for honest evaluation.

---

## The 14 classes

`Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax`

Outputs are independent sigmoids (multi-label), not softmax — a single image can show several conditions at once.

---

## Pipeline

```
raw NIH archive  (Data_Entry_2017.csv, train_val_list.txt, test_list.txt, images_*/)
        │
        ▼
[src/preprocess.py]   clean ages, encode gender, normalize age, expand 14 labels,
        │             official patient-wise split, UNDERSAMPLE No-Finding (train only)
        ▼
train.csv / val.csv / test.csv
        │
        ▼
[src/train.py]        DenseNet-121 + clinical branch, auto pos_weight, AMP,
        │             differential LR + OneCycle, per-class AUC, best checkpoint
        ▼
outputs/checkpoints/model_best.pth
        │
        ├──► [notebooks/train_kaggle.ipynb]   one self-contained notebook that runs
        │                                      the WHOLE pipeline + evaluation on Kaggle
        │
        └──► [app/app.py]                      Streamlit demo: upload X-ray →
                                               healthy/abnormal verdict + GradCAM + LLM
```

Each stage is independent: re-run preprocessing without touching training, or update the app without retraining. `config.yaml` holds shared paths and hyperparameters.

---

## Project structure

```
ChestX-ray14/
├── src/
│   ├── preprocess.py     # Stage 1 — clean + split + undersample → train/val/test.csv
│   ├── dataset.py        # Dataset, transforms, the 14 LABELS
│   ├── model.py          # DenseNet-121 + clinical fusion
│   └── train.py          # Stage 2 — training loop, auto pos_weight, per-class AUC
├── notebooks/
│   ├── exploration.ipynb   # EDA — label balance, co-occurrence, age/gender, samples
│   ├── preprocessing.ipynb # visual walk-through of the preprocessing pipeline
│   └── train_kaggle.ipynb  # self-contained: preprocess → train → evaluate (Kaggle-ready)
├── app/
│   ├── app.py            # Streamlit demo (healthy verdict + GradCAM + Groq)
│   └── utils.py          # load_model, predict, healthy_verdict, GradCAM, Groq
├── config.yaml           # paths + hyperparameters
├── requirements.txt
└── .env.example          # copy to .env, add GROQ_API_KEY
```

---

## Architecture

```
Image (224×224) ──► DenseNet-121 features ──► GAP ──► 1024-dim
                                                              │
Age, Gender ──────► FC(2→16) → ReLU → Dropout ──► 16-dim     │
                                                              ▼
                                              Concat → FC(1040→512) → FC(512→14)
```

---

## Quick start — Kaggle (recommended)

The fastest path is the self-contained notebook:

1. New Kaggle notebook → **Add Data** → [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data).
2. Upload `notebooks/train_kaggle.ipynb`.
3. Enable GPU (T4/P100) → **Run All**.

It locates the dataset automatically, preprocesses, trains for 15 epochs, and prints a per-class AUC table plus the derived **healthy/abnormal detector AUC**. The best checkpoint lands in `/kaggle/working/checkpoints/model_best.pth`.

---

## Quick start — local / scripts

```bash
pip install -r requirements.txt

# Stage 1 — preprocess (writes train/val/test.csv + prints pos_weights)
python src/preprocess.py --archive E:/archive --out data/processed

# Stage 2 — train (edit the PATHS block at the top of train.py first)
python src/train.py
```

`preprocess.py` flags: `--nf-ratio` (No-Finding undersampling, default 0.5), `--val-frac` (default 0.10), `--pos-cap` (default 10).

---

## Training details

| Technique | Why |
|---|---|
| **Undersampling + auto `pos_weight`** | Undersampling fixes the healthy-vs-sick imbalance; per-class `pos_weight` (computed from `train.csv`, capped at 10) handles residual rare-disease imbalance without the over-prediction that an uncapped weight causes. |
| **Differential learning rates** | Pretrained backbone gets a tiny LR (2e-5); the new head gets a larger one (2e-4). |
| **OneCycleLR + AMP + grad clipping** | Fast, stable convergence; float16 activations halve GPU memory. |
| **Best-checkpoint by mean val AUC** | Saved with per-class AUC so you can see exactly which pathologies are learning. |

Published DenseNet baselines reach ~0.80 mean AUC with full training.

---

## Streamlit demo

```bash
cp .env.example .env          # add GROQ_API_KEY=gsk_...
streamlit run app/app.py
```

Upload an X-ray + age/gender → headline **Healthy / Abnormal verdict**, GradCAM heatmap, per-class probabilities, and a plain-language Groq explanation.

---

## GradCAM

Answers *"which pixels drove this prediction?"* by weighting the last dense block's feature maps (`model.features.denseblock4`, a 7×7 grid) by their gradient w.r.t. the target class, then overlaying the heatmap on the image. In medical imaging this is a sanity check that the model looks at lung fields, not scanner labels.

---

## LLM explanation (Groq)

After inference, the 14 probabilities **and the derived healthy score** are formatted into a prompt and sent to Groq (`llama-3.3-70b-versatile`, sub-second on their LPU hardware) for a short plain-language interpretation. The LLM only sees the numbers — it cannot look at the image — and is prompted to remind the reader that the model is partially trained and not clinically valid.

---

## Dataset

NIH Clinical Center — ChestX-ray14 · 112,120 frontal X-rays from 30,805 patients
Source: <https://nihcc.app.box.com/v/ChestXray-NIHCC>

**Preprocessing encodings:** Gender `M→0, F→1` · Age `raw/100` · ages > 100 dropped · official patient-wise train/val/test split.
