# CGVQM: Computer Graphics Video Quality Metric
[![GitHub License](https://img.shields.io/github/license/IntelLabs/cgvqm)](https://github.com/IntelLabs/cgvqm/code/main/LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/cgvqm/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/cgvqm)
[![pytorch](https://img.shields.io/badge/PyTorch-v2.4.1-green?logo=pytorch)](https://pytorch.org/get-started/locally/)
![python-support](https://img.shields.io/badge/Python-3.10-3?logo=python)

[Dataset](https://github.com/IntelLabs/CGVQM) | [Paper](https://github.com/IntelLabs/CGVQM) (Coming Soon...)

<img src="media/teaser.png" width=35%>
<br /><br />

**CGVQM** is a *full-reference* video quality metric that predicts perceptual differences between pairs of videos.  
Like PSNR and SSIM, it compares a **ground-truth reference** to a **distorted** version (e.g. blurry, noisy, aliased).  

What sets CGVQM apart is that it is the first metric **calibrated for distortions from advanced rendering techniques**, accounting for both **spatial** and **temporal** artifacts.

### Features

- 📊 **Calibrated** on our [**CGVQD** dataset](https://github.com/IntelLabs/CGVQM), capturing complex artifacts from modern computer graphics.
- 🎚️ Outputs on an **interpretable perceptual scale** (e.g. *imperceptible* → *annoying*).
- 🗺️ Provides **error maps** to visualize *where* and *why* errors occur — not just a single score.
- 🖼️ Robust across a wide variety of content: from **stylized fantasy** to **photorealistic open worlds**, and across many rendering techniques.

CGVQM is implemented in **PyTorch**, optimized for **CUDA GPUs** (CPU also supported). See below for installation and usage instructions.

More details on the metric and dataset can be found in:

> CGVQM+D: Computer Graphics Video Quality Metric and Dataset.\
> A. Jindal, N. Sadaka, M. M. Thomas, A. Sochenov and A. Kaplanyan.\
> COMPUTER GRAPHICS forum, Volume 44 (2025), Number 8\
> https://doi.org/xxx/xxx

If you use the metric or the dataset in your research, please cite the paper above. 

## Quickstart
1. Obtain the CGVQM codebase, by cloning the repository:
```bash
git clone git@github.com:IntelLabs/CGVQM.git   # skip if a .zip is provided or you use Github GUI
```

2. Install requirements (numpy, torch, and torchvision):
```bash
pip install -r requirements.txt
```

After installation, run `python cgvqm.py`. This script provides a demo usage of CGVQM by comparing 'media/Dock_dist.mp4' with 'media/Dock_ref.mp4':

## Training
Coming Soon...