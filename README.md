# GAN Academic Presentation & Demo Package
## Prepared by Prasanna

---

## Package Contents

```
GAN_Final_Package/
├── README.md                              ← This file
├── GAN_Presentation_Final.pptx            ← 33-slide detailed 16:9 presentation
├── demo1_tabular_gan/
│   ├── app.py                             ← Streamlit Tabular GAN Balancer (improved)
│   ├── requirements.txt
│   └── README.md
└── demo2_image_gan_finetune/
    ├── app.py                             ← Streamlit Image GAN Fine-tuner (improved)
    ├── requirements.txt
    └── README.md
```

---

## 1. Presentation — GAN_Presentation_Final.pptx

**33 slides · 16:9 format · Deep Navy + Teal colour theme**

| Slides | Topic |
|--------|-------|
| 1–2    | Title + Roadmap |
| 3–4    | Introduction & History Timeline |
| 5–6    | Mathematical Objective & Divergence Theory |
| 7–9    | Step-by-step Mechanism, Training Algorithm, Common Problems |
| 10     | Core Architecture Diagrams |
| 11–16  | GAN Variants: cGAN, DCGAN, WGAN/WGAN-GP, Pix2Pix, CycleGAN, SRGAN, StyleGAN, CTGAN |
| 17–23  | Applications: Image Generation, Image Translation, Super-Resolution, Data Augmentation, Healthcare, Emerging Applications |
| 24–26  | Deepfakes: Generation Mechanisms, Detection Methods, Ethics & Societal Implications |
| 27–28  | Demo 1 & Demo 2 explanations |
| 29–31  | Evaluation Metrics, GAN vs. Other Generative Models Comparison Table, Best Practices Checklist |
| 32     | Conclusion |
| 33     | 15 academic references (IEEE/ACM/NeurIPS/CVPR/ICCV) |

---

## 2. Demo 1 — Tabular GAN Balancer

### What it does
- Loads an imbalanced binary classification dataset (90% majority / 10% minority)
- Trains a lightweight **MLP-GAN** on scaled minority-class features
- Generates synthetic minority rows until the dataset is balanced
- Evaluates a Random Forest classifier **before and after** balancing on a real-data holdout
- Includes privacy check (Nearest-Neighbour Distance Ratio — NNDR)

### How to run
```bash
cd demo1_tabular_gan
pip install -r requirements.txt
streamlit run app.py
```

### Improvements over original
- Real-time progress bar (no UI freeze during training)
- Per-feature histogram overlays (real minority vs. synthetic minority)
- Privacy/memorisation check (NNDR)
- AUC-ROC added to evaluation metrics
- Seeded Random Forest for reproducibility
- Label smoothing in Discriminator training
- Batch Normalisation added to Generator

---

## 3. Demo 2 — Image GAN Fine-tuning

### What it does
- Accepts a ZIP of 5 folders (one per individual), 2–5 images each
- Preprocesses: Resize → CenterCrop(64) → RandomFlip → ColorJitter → Normalize
- Optionally loads pretrained G/D checkpoints
- Fine-tunes a **DCGAN-style** network (Generator + Discriminator)
- Displays per-epoch generated image grids
- Saves `finetuned_generator.pt`, `finetuned_discriminator.pt`, `generated_grid.png`

### How to run
```bash
cd demo2_image_gan_finetune
pip install -r requirements.txt
streamlit run app.py
```

### Improvements over original
- Three-tab layout: Setup / Training / Results
- Architecture summary displayed in UI
- Configurable nz, ngf, lr_g, lr_d separately
- Label smoothing on Discriminator (real labels = 0.9)
- Download ZIP of checkpoints + grid from UI
- Cleaner folder inspection with image count per class
- Channel-wise stats as lightweight quality proxy

---

## Ethical Notes

> These demos are for **academic and educational use only**.

- All generated data must be clearly labelled as synthetic.
- Do not train on personal images without explicit consent.
- Do not use Demo 2 outputs for identity impersonation or fraud.
- Synthetic tabular data is not guaranteed to be private — run NNDR/membership inference audits before sharing.

---

## References (key citations used in slides)

1. Goodfellow et al. (2014). NeurIPS. — Original GAN
2. Radford et al. (2015). — DCGAN
3. Arjovsky et al. (2017). — WGAN
4. Gulrajani et al. (2017). NeurIPS. — WGAN-GP
5. Isola et al. (2017). CVPR. — Pix2Pix
6. Zhu et al. (2017). ICCV. — CycleGAN
7. Ledig et al. (2017). CVPR. — SRGAN
8. Karras et al. (2019–2021). — StyleGAN family
9. Xu et al. (2019). NeurIPS. — CTGAN
10. Rossler et al. (2019). ICCV. — FaceForensics++
