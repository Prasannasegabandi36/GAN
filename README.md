# Demo 2: Small Image GAN Fine-tuning

## Dataset Format
Create a ZIP like this:
```
dataset.zip
  person_1/  img1.jpg  img2.jpg  img3.jpg
  person_2/  img1.jpg  img2.jpg
  person_3/  img1.jpg  img2.jpg  img3.jpg  img4.jpg
  person_4/  img1.jpg  img2.jpg
  person_5/  img1.jpg  img2.jpg  img3.jpg
```
Use only images you own or have explicit consent to use.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- 3-tab UI: Setup / Training / Results
- Folder inspection with image count per class
- Configurable nz, ngf, learning rates for G and D separately
- Per-epoch generated sample grid in Training tab
- Loss curve in Results tab
- Download generated grid (PNG) and model checkpoints (ZIP)

## Notes
- With only 10–25 images, results are not production-quality
- For better results: use a pretrained checkpoint and fine-tune
- Educational purpose only — do not use for identity impersonation
