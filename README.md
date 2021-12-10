# Neural Image Beauty Predictor Based on Bradley-Terry Model
This project use CNN models for the image beauty assessment. To better predict the image beauty assessment by Human Visual System (HVS), our dataset includes survey results of landscape, portrait and architecture images.

# PyTorch

PyTorch implementation of [Neural IMage Assessment](https://arxiv.org/abs/2111.10127) by Shiyu Li, Hao Ma and Xiangyu Hu. 


## Installing

### Requirement
```python
pip install -r requirements.txt
```

## Dataset

AVA dataset (used for pretrained models)

The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf)
You can get it from [here](https://github.com/mtobeiyf/ava_downloader)
Here are some examples of images with theire scores 
![result1](https://3.bp.blogspot.com/-_BuiLfAsHGE/WjgoftooRiI/AAAAAAAACR0/mB3tOfinfgA5Z7moldaLIGn92ounSOb8ACLcBGAs/s1600/image2.png)

Our dataset

The dataset is devided into landscape, portrait and architecture images. (https://drive.google.com/drive/folders/1t9FfFZCEGzQk8mxs-i7BgLptyaYnzVxe?usp=sharing)

Landscape
![Images](https://github.com/lishiyu0088/Neural_Bradley-Terry/tree/main/readme_images/L1.jpg)
![Results](https://github.com/lishiyu0088/Neural_Bradley-Terry/tree/main/readme_images/L1.png)

Portrait
![Images](https://github.com/lishiyu0088/Neural_Bradley-Terry/tree/main/readme_images/P1.jpg)
![Results](https://github.com/lishiyu0088/Neural_Bradley-Terry/tree/main/readme_images/P1.png)

Architecture
![Images](https://github.com/lishiyu0088/Neural_Bradley-Terry/tree/main/readme_images/B1.jpg)
![Results](https://github.com/lishiyu0088/Neural_Bradley-Terry/tree/main/readme_images/B1.png)
## Pre-train model (In Progress)

```bash

```


## Deployment (In progress)

```bash

```

## Usage
```

Usage: 
GPU: python main-gpu.py
CPU: python main-cpu.py

Options:
  -Alex  Train by Alex net
  -Squeeze  Train by Squeeze net
  -VGG  Train by VGG net
  -LSiM  Train by LSiM net


```


## Contributing

Contributing are welcome


## License

This project is licensed under the MIT License

## Acknowledgments

The code can only be used for academic purposes. Any commercial uses should be declared and permitted in advance.
