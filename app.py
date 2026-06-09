"""
Demo 2: Fine-tune a Small DCGAN-style Model on Small Image Folders
===================================================================
Improvements over original:
- Tabbed layout: Setup / Training / Results
- Per-epoch sample grid shown in sidebar (not blocking)
- Model architecture summary displayed in UI
- FID-style feature statistics (InceptionV3 not available without internet,
  so we compute per-channel mean/std as a lightweight quality proxy)
- Cleaner folder browser with image count per class
- Download generated grid as PNG and checkpoints as ZIP
- Stronger ethical / consent warning with checklist
"""

import os
import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image GAN Fine-tuning",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️  Demo 2 — Fine-tune a DCGAN on Small Image Datasets")

st.error(
    "**Ethical & Legal Notice** — this demo is for educational purposes only.\n\n"
    "- ✅ Use only images you own or have explicit written consent to use.\n"
    "- ✅ Generated outputs must be clearly labelled as synthetic / AI-generated.\n"
    "- ❌ Do NOT use real people's faces without consent — this may be illegal.\n"
    "- ❌ Do NOT use this demo for identity impersonation, fraud, or harassment."
)

# ─── ARCHITECTURE ────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # State: nz × 1 × 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),          # 4×4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),          # 8×8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),          # 16×16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),               # 32×32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),                                         # 64×64×nc
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # Input: nc × 64 × 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),                  # 32×32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),  # 16×16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),  # 8×8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),  # 4×4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),                                       # 1×1×1
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def extract_zip_to_tmp(uploaded_file) -> str:
    root = tempfile.mkdtemp()
    zip_path = os.path.join(root, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)
    # Find first directory containing sub-folders
    for dirpath, dirnames, _ in os.walk(root):
        if dirnames:
            return dirpath
    return root


def tensor_to_pil_bytes(tensor):
    grid = utils.make_grid(tensor, nrow=4, normalize=True, value_range=(-1, 1))
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0E1117")
    ax.imshow(arr); ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0E1117")
    plt.close()
    buf.seek(0)
    return buf


def show_grid(tensor, title: str):
    buf = tensor_to_pil_bytes(tensor)
    st.image(buf, caption=title, use_container_width=True)


def image_stats(tensor) -> dict:
    """Channel-wise mean and std as a lightweight quality proxy."""
    arr = tensor.cpu().numpy()  # (N, C, H, W), range [-1, 1]
    means = arr.mean(axis=(0, 2, 3))
    stds  = arr.std(axis=(0, 2, 3))
    return {
        "Mean (R,G,B)": [round(float(m), 4) for m in means],
        "Std  (R,G,B)": [round(float(s), 4) for s in stds],
    }


# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_setup, tab_train, tab_results = st.tabs(["📁  Setup & Dataset", "🏋️  Training", "📊  Results"])

# ════════════════════════════════
# TAB 1 — SETUP
# ════════════════════════════════
with tab_setup:
    st.subheader("Dataset")
    st.markdown(
        "Create a ZIP file structured like this:\n"
        "```\n"
        "dataset.zip\n"
        "  person_1/  image1.jpg  image2.jpg ...\n"
        "  person_2/  image1.jpg  image2.jpg ...\n"
        "  person_3/  image1.jpg  image2.jpg ...\n"
        "  person_4/  image1.jpg  image2.jpg ...\n"
        "  person_5/  image1.jpg  image2.jpg ...\n"
        "```\n"
        "Each sub-folder = one identity / class. 2–5 images per folder is sufficient for demonstration."
    )

    uploaded_zip = st.file_uploader("Upload dataset ZIP", type=["zip"])
    local_path   = st.text_input("Or enter a local folder path (when running locally)", "")
    ckpt_g       = st.text_input("Optional: path to pretrained Generator checkpoint (.pt)", "")
    ckpt_d       = st.text_input("Optional: path to pretrained Discriminator checkpoint (.pt)", "")

    if uploaded_zip or (local_path and os.path.isdir(local_path)):
        try:
            data_root = extract_zip_to_tmp(uploaded_zip) if uploaded_zip else local_path
            probe = datasets.ImageFolder(data_root, transform=transforms.ToTensor())
            st.success(f"✅  Detected **{len(probe.classes)} folders**: `{probe.classes}`  |  Total images: **{len(probe)}**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Images per folder:**")
                for cls, idx in probe.class_to_idx.items():
                    count = sum(1 for _, l in probe.samples if l == idx)
                    st.markdown(f"- `{cls}`: {count} image(s)")
            with col2:
                st.markdown("**Augmentation pipeline:**")
                st.markdown(
                    "1. Resize to 72px (shorter edge)\n"
                    "2. CenterCrop 64×64\n"
                    "3. RandomHorizontalFlip (p=0.5)\n"
                    "4. ColorJitter (brightness/contrast/saturation ±15%)\n"
                    "5. ToTensor + Normalize to [-1, 1]"
                )
            st.session_state["data_root"] = data_root
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
    else:
        st.session_state.pop("data_root", None)

    st.subheader("Architecture Summary")
    nz_preview = 100; ngf_preview = 64
    G_preview = Generator(nz=nz_preview, ngf=ngf_preview)
    total_params = sum(p.numel() for p in G_preview.parameters())
    st.markdown(
        f"**Generator** — {total_params:,} parameters\n\n"
        "```\n"
        f"Input:  z  [{nz_preview} × 1 × 1]\n"
        f"  ConvTranspose2d  →  {ngf_preview*8} × 4 × 4   + BN + ReLU\n"
        f"  ConvTranspose2d  →  {ngf_preview*4} × 8 × 8   + BN + ReLU\n"
        f"  ConvTranspose2d  →  {ngf_preview*2} × 16 × 16 + BN + ReLU\n"
        f"  ConvTranspose2d  →  {ngf_preview}   × 32 × 32 + BN + ReLU\n"
        f"  ConvTranspose2d  →  3     × 64 × 64 + Tanh\n"
        "```\n\n"
        "**Discriminator** mirrors this path in reverse using strided Conv2d + LeakyReLU(0.2)."
    )


# ════════════════════════════════
# TAB 2 — TRAINING
# ════════════════════════════════
with tab_train:
    st.subheader("Training Settings")

    col_l, col_r = st.columns(2)
    with col_l:
        epochs    = st.slider("Fine-tuning epochs", 5, 200, 30)
        batch_sz  = st.slider("Batch size", 2, 32, 8)
        nz        = st.slider("Latent dimension (nz)", 64, 256, 100, step=8)
    with col_r:
        ngf       = st.select_slider("Generator feature maps (ngf)", [32, 64, 128], value=64)
        lr_g      = st.select_slider("Generator LR", [0.0001, 0.0002, 0.0005], value=0.0002)
        lr_d      = st.select_slider("Discriminator LR", [0.0001, 0.0002, 0.0005], value=0.0002)
        show_freq = st.slider("Show generated samples every N epochs", 1, 20, 5)

    start = st.button("▶  Start Fine-tuning", type="primary",
                      disabled="data_root" not in st.session_state)
    if "data_root" not in st.session_state and not start:
        st.info("Go to the **Setup** tab and upload your dataset first.")

    if start and "data_root" in st.session_state:
        data_root = st.session_state["data_root"]

        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = datasets.ImageFolder(data_root, transform=transform)

        if len(dataset.classes) < 2:
            st.error("Expected ≥ 2 sub-folders inside the ZIP. Check folder structure.")
            st.stop()

        loader = DataLoader(
            dataset,
            batch_size=min(batch_sz, len(dataset)),
            shuffle=True,
            drop_last=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.caption(f"Running on: **{device}**  |  Dataset: {len(dataset)} images, "
                   f"{len(dataset.classes)} classes")

        G = Generator(nz=nz, ngf=ngf).to(device)
        D = Discriminator(ndf=ngf).to(device)
        G.apply(weights_init); D.apply(weights_init)

        # Load pretrained checkpoints if provided
        if ckpt_g and os.path.exists(ckpt_g):
            G.load_state_dict(torch.load(ckpt_g, map_location=device))
            st.info(f"Loaded pretrained Generator from: {ckpt_g}")
        if ckpt_d and os.path.exists(ckpt_d):
            D.load_state_dict(torch.load(ckpt_d, map_location=device))
            st.info(f"Loaded pretrained Discriminator from: {ckpt_d}")

        opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        fixed_noise = torch.randn(16, nz, 1, 1, device=device)

        progress_bar = st.progress(0)
        log_area = st.empty()
        grid_area = st.empty()
        loss_records = []

        for epoch in range(1, epochs + 1):
            d_losses, g_losses = [], []
            for real, _ in loader:
                real = real.to(device)
                b = real.size(0)
                real_lbl = torch.ones(b, device=device) * 0.9   # label smoothing
                fake_lbl = torch.zeros(b, device=device)

                # Discriminator
                D.zero_grad()
                out_real = D(real)
                loss_real = criterion(out_real, real_lbl)
                noise = torch.randn(b, nz, 1, 1, device=device)
                fake = G(noise)
                out_fake = D(fake.detach())
                loss_fake = criterion(out_fake, fake_lbl)
                d_loss = loss_real + loss_fake
                d_loss.backward(); opt_d.step()

                # Generator
                G.zero_grad()
                out_g = D(fake)
                g_loss = criterion(out_g, torch.ones(b, device=device))
                g_loss.backward(); opt_g.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            avg_d = np.mean(d_losses); avg_g = np.mean(g_losses)
            loss_records.append({"Epoch": epoch, "D_loss": round(avg_d, 4), "G_loss": round(avg_g, 4)})
            progress_bar.progress(epoch / epochs)
            log_area.markdown(
                f"**Epoch {epoch}/{epochs}**  |  D loss: `{avg_d:.4f}`  |  G loss: `{avg_g:.4f}`"
            )

            if epoch % show_freq == 0 or epoch == epochs:
                with torch.no_grad():
                    samples = G(fixed_noise).detach().cpu()
                with grid_area.container():
                    show_grid(samples, f"Generated samples — epoch {epoch}")

        # Save
        out_dir = Path("/home/claude/generated_output")
        out_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            final_samples = G(fixed_noise).detach().cpu()
        buf = tensor_to_pil_bytes(final_samples)
        grid_bytes = buf.read()
        (out_dir / "generated_grid.png").write_bytes(grid_bytes)
        torch.save(G.state_dict(), out_dir / "finetuned_generator.pt")
        torch.save(D.state_dict(), out_dir / "finetuned_discriminator.pt")

        st.session_state["final_samples"] = final_samples
        st.session_state["loss_records"] = loss_records
        st.session_state["grid_bytes"] = grid_bytes
        st.session_state["out_dir"] = str(out_dir)
        st.success("✅  Fine-tuning complete! Go to the **Results** tab.")


# ════════════════════════════════
# TAB 3 — RESULTS
# ════════════════════════════════
with tab_results:
    if "final_samples" not in st.session_state:
        st.info("Run training first (in the **Training** tab).")
    else:
        st.subheader("Generated Image Grid (16 samples, 4×4)")
        show_grid(st.session_state["final_samples"], "Final generated samples")

        st.subheader("Training Loss Curves")
        loss_df = torch.zeros(1)  # placeholder
        import pandas as pd
        ldf = pd.DataFrame(st.session_state["loss_records"]).set_index("Epoch")
        st.line_chart(ldf)
        st.caption(
            "Healthy training: D_loss and G_loss oscillate around similar values (ideally ~1.4 for BCELoss).\n"
            "D_loss → 0: discriminator too strong — try lower lr_d or fewer D updates.\n"
            "G_loss spikes upward: generator falling behind — try higher lr_g or spectral norm."
        )

        # Lightweight stats
        stats = image_stats(st.session_state["final_samples"])
        st.subheader("Generated Image Statistics (quality proxy)")
        st.json(stats)
        st.caption(
            "With only 10–25 training images, these statistics will differ from real data. "
            "For research-grade evaluation, use FID (requires ≥ 2000 real images)."
        )

        st.subheader("Download Results")
        # Build a zip of outputs
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("generated_grid.png", st.session_state["grid_bytes"])
            out_dir = st.session_state["out_dir"]
            for pt_file in ["finetuned_generator.pt", "finetuned_discriminator.pt"]:
                fp = os.path.join(out_dir, pt_file)
                if os.path.exists(fp):
                    zf.write(fp, pt_file)
        zip_buf.seek(0)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️  Download generated_grid.png",
                data=st.session_state["grid_bytes"],
                file_name="generated_grid.png",
                mime="image/png",
            )
        with col2:
            st.download_button(
                "⬇️  Download checkpoints + grid (ZIP)",
                data=zip_buf.read(),
                file_name="gan_finetuned_outputs.zip",
                mime="application/zip",
            )
