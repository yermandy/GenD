# Deepfake Detection that Generalizes Across Benchmarks (WACV 2026)

[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=FFF)](https://arxiv.org/abs/2508.06248)
[![Hugging Face Badge](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/collections/yermandy/gend)

This is the official repository for the paper:

**[Deepfake Detection that Generalizes Across Benchmarks](https://arxiv.org/abs/2508.06248)**.

### Abstract

> The generalization of deepfake detectors to unseen manipulation techniques remains a challenge for practical deployment. Although many approaches adapt foundation models by introducing significant architectural complexity, this work demonstrates that robust generalization is achievable through a parameter-efficient adaptation of one of the foundational pre-trained vision encoders. The proposed method, GenD, fine-tunes only the Layer Normalization parameters (0.03% of the total) and enhances generalization by enforcing a hyperspherical feature manifold using L2 normalization and metric learning on it.
>
> We conducted an extensive evaluation on 14 benchmark datasets spanning from 2019 to 2025. The proposed method achieves state-of-the-art performance, outperforming more complex, recent approaches in average cross-dataset AUROC. Our analysis yields two primary findings for the field: 1) training on paired real-fake data from the same source video is essential for mitigating shortcut learning and improving generalization, and 2) detection difficulty on academic datasets has not strictly increased over time, with models trained on older, diverse datasets showing strong generalization capabilities.
>
> This work delivers a computationally efficient and reproducible method, proving that state-of-the-art generalization is attainable by making targeted, minimal changes to a pre-trained foundational image encoder model.

## Inference using Hugging Face transformers

This example shows how to run inference with the pretrained GenD model from Hugging Face without other dependencies except `torch` and `transformers`. It expects that input images are already preprocessed by detector.

### Minimal dependencies

``` bash
conda create --name GenD python=3.12 uv -y
conda activate GenD
uv pip install torch==2.8.0 torchvision==0.23.0 transformers==4.56.2
```

### Inference with transformers

``` python
import requests
import torch
from PIL import Image

from src.hf.modeling_gend import GenD

# Other models can be found in https://huggingface.co/collections/yermandy/gend
# - yermandy/GenD_CLIP_L_14
# - yermandy/GenD_PE_L
# - yermandy/GenD_DINOv3_L
model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")

urls = [
    "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/DF/000_003/000.png?raw=true",
    "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/real/000/000.png?raw=true",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
tensors = torch.stack([model.feature_extractor.preprocess(img) for img in images])
logits = model(tensors)
probs = logits.softmax(dim=-1)

print(probs)
```

## Inference using Gradio UI

We provide a Gradio-based web UI for inference, install all dependencies as described in the [Training](#training) section, then run:

``` bash
python app/run.py
```

<video controls src="media/gradio.mp4" title="Title"></video>

## Training

### Set up environment

``` bash
conda create --name GenD python=3.12 uv -y
conda activate GenD
uv pip install -r requirements.txt
```

### Minimal example without external data

#### Training example

Examine `src/exp/examples.py`, each experiment name is defined as a key, a value overrides default configuration of `Config` object from `src/config.py`. For example, try to run `example-training` experiment:

``` bash
python run_exp.py example-training
```

#### Test example after the model is trained

``` bash
python run_exp.py example-test --from_exp example-training --test
```

Alternatively, you can try inference using one of our released models from Hugging Face:

``` bash
python run_exp.py GenD_CLIP--CDFv2-example --test
python run_exp.py GenD_PE--CDFv2-example --test
python run_exp.py GenD_DINO--CDFv2-example --test
```

### Full training

To fully train the model, you need to download datasets, preprocess them, and create files with paths to the images.

The training entry will be similar to the minimal example above.

All experiments (configs) from the paper are stored in the `src/exp` folder.

#### Prepare the dataset

Take for example [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset, follow these steps:

1. Download the dataset first from the [official source](https://github.com/ondyari/FaceForensics). The root of this dataset is `./FaceForensics`

2. Preprocess the dataset using `detector.py` script:

``` bash
python detector.py -i FaceForensics/manipulated_sequences/Deepfakes/c23/videos/ --mask_folder FaceForensics/masks/manipulated_sequences/Deepfakes/masks/videos/ -m at_least -n 32 -o datasets/FF/DF/ --det_thres 0.1 -s 1.3 --target_size none
```

Repeat the process for other manipulation methods and real videos. After processing everything, you will get a similar structure:

``` bash
datasets
└── FF
    ├── DF
    │   └── 000_003
    │       ├── 025.png
    │       └── 038.png
    ├── F2F
    │   └── 000_003
    │       ├── 019.png
    │       └── 029.png
    ├── FS
    │   └── 000_003
    │       ├── 019.png
    │       └── 029.png
    ├── NT
    │   └── 000_003
    │       ├── 019.png
    │       └── 029.png
    └── real
        └── 000
            ├── 025.png
            └── 038.png
```

3. Create files with paths to images similar to the ones in `config/datasets` directory. It can be done using:

``` bash
find datasets/FF/DF/* -type f | sort > config/datasets/FF/DF.txt
```

We manage links to files using `src/utils/files.py`.

### Cite

``` bibtex
@article{yermakov2025deepfake,
  title={Deepfake Detection that Generalizes Across Benchmarks},
  author={Yermakov, Andrii and Cech, Jan and Matas, Jiri and Fritz, Mario},
  journal={arXiv preprint arXiv:2508.06248},
  year={2025}
}
```
