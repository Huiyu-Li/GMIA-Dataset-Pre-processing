## GMIA-Dataset Pre-processing &mdash; Official PyTorch implementation

![Teaser image](./docs/image_pre-processing.png)

**Data Exfiltration and Anonymization of Medical Images based on Generative Models**<br>
Huiyu Li<br>
https://inria.hal.science/tel-04875160<br> (Chapter3, p57-59)

## Motivation
The original images from the source dataset (e.g. MIMIC-CXR-JPG dataset) vary in size, making them unsuitable for neural network training. To standardize the raw images and ensure compatibility, we developed a preprocessing pipeline that resizes all images to a uniform size.

## Requirements

* 64-bit Python 3.9 pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.

## Getting started

**MIMIC-CXR-JPG**:
Step 0: Download the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).

Step1: Boundary crop and insert the new image size (optinal)

```.bash
python crop_black_boundary.py
```

Step2: Resize the image to required input size of the U-net<br>
Step3: Get the segmentation masks<br>
Step4: Zoom in the segmentation masks<br>
Step5: Get a squared bbox <br>
    Get the center point C of the lung mask bbox<br>
    Get the shortest distance from C to boundaries<br>
Step6: Crop the image with the squared bbox<br>
Step7 Resize the image into a uniform size (e.g. 512*512)<br>

```.bash
python python3 pre_processing.py
```

References:
1. [Lung segmentation](https://github.com/IlliaOvcharenko/lung-segmentation)

## Citation
```
@phdthesis{li:tel-04875160,
  TITLE = {{Data Exfiltration and Anonymization of Medical Images based on Generative Models}},
  AUTHOR = {Li, Huiyu},
  URL = {https://inria.hal.science/tel-04875160},
  SCHOOL = {{Inria \& Universit{\'e} Cote d'Azur, Sophia Antipolis, France}},
  YEAR = {2024},
  MONTH = Nov,
  KEYWORDS = {Privacy and security ; Steganography ; Medical image anonymization ; Identity-utility extraction ; Latent code optimization ; Data exfiltration attack ; Attaque d'exfiltration de donn{\'e}es ; Compression d'images ; Confidentialit{\'e} ; St{\'e}ganographie ; Anonymisation d'images m{\'e}dicales ; Extraction d'identit{\'e}-utilit{\'e} ; Optimisation du code latent},
  TYPE = {Theses},
  PDF = {https://inria.hal.science/tel-04875160v2/file/Huiyu_Thesis_ED_STIC.pdf},
  HAL_ID = {tel-04875160},
  HAL_VERSION = {v2},
}
```
