---
title: FSRCNN Super Resolution
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
---

## FSRCNN (PyTorch)

Implementation of [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367) in PyTorch.

## Install

```bash
pip install git+https://github.com/NicoCeresa/FSRCNN.git
```

## Usage

```python
import torch
from fsrcnn import FSRCNN, FSRCNN_s

# Full model (d=56, s=12, m=4)
model = FSRCNN(scale=3)

# Lightweight variant (d=32, s=5, m=1)
model = FSRCNN_s(scale=3)

lr = torch.randn(1, 3, 64, 64)
hr = model(lr)  # (1, 3, 192, 192)
```

## Results

Trained on the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset with L1 loss.

| Metric | Scale | Paper | This repo |
|--------|-------|-------|-----------|
| PSNR   | 2×    | 36.94 | 34.77     |
| PSNR   | 3×    | 33.16 | 32.05     |
| PSNR   | 4×    | 30.55 | 30.82     |

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>Cropped</center></td>
        <td><center>Bicubic ×3</center></td>
        <td><center>FSRCNN ×3</center></td>
    </tr>
    <tr>
        <td><center><img src="./images/cottage_og.png"></center></td>
        <td><center><img src="./images/cottage_crop.png"></center></td>
        <td><center><img src="./images/cottage_lr.png"></center></td>
        <td><center><img src="./images/cottage_hr.png"></center></td>
    </tr>
    <tr>
        <td><center><img src="./images/china_og.png"></center></td>
        <td><center><img src="./images/china_crop.png"></center></td>
        <td><center><img src="./images/china_lr.png"></center></td>
        <td><center><img src="./images/china_hr.png"></center></td>
    </tr>
</table>

## Citation

```bibtex
@inproceedings{dong2016accelerating,
  title     = {Accelerating the Super-Resolution Convolutional Neural Network},
  author    = {Dong, Chao and Loy, Chen Change and Tang, Xiaoou},
  booktitle = {European Conference on Computer Vision},
  pages     = {391--407},
  year      = {2016},
  publisher = {Springer}
}
```
