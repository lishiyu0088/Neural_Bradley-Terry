# Neural Image Beauty Predictor Based on Bradley-Terry Model
This project use CNN models for the image beauty assessment. To better predict the image beauty assessment by Human Visual System (HVS), our dataset includes survey results of landscape, portrait and architecture images.

# PyTorch

PyTorch implementation of [Neural Image Beauty Predictor Based on Bradley-Terry Model](https://arxiv.org/abs/2111.10127) by Shiyu Li, Hao Ma and Xiangyu Hu. 


## Installing

### Requirement
```python
pip install -r requirements.txt
```

## Dataset

### AVA dataset (used for pretrained models)

The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf)
You can get it from [here](https://github.com/mtobeiyf/ava_downloader). Before running the code, please create a new folder with the name "AVA_dataset".
Here are some examples of images with theire scores 
![result1](https://3.bp.blogspot.com/-_BuiLfAsHGE/WjgoftooRiI/AAAAAAAACR0/mB3tOfinfgA5Z7moldaLIGn92ounSOb8ACLcBGAs/s1600/image2.png)

### Our dataset

The dataset is devided into landscape, portrait and architecture images. Our dataset evaluates images in different groups. The scores are calculated by survey results according to Bradley-Terry model. Our dataset can be downloaded by https://drive.google.com/drive/folders/1t9FfFZCEGzQk8mxs-i7BgLptyaYnzVxe?usp=sharing. Before running the code, please put the images into the folder "inputs".

Landscape images and scores

![Images](https://raw.githubusercontent.com/lishiyu0088/Neural_Bradley-Terry/main/readme_images/L1.jpg)
<img src="https://raw.githubusercontent.com/lishiyu0088/Neural_Bradley-Terry/main/readme_images/L1.png" width="50%" height="50%">

Portrait images and scores

<img src="https://raw.githubusercontent.com/lishiyu0088/Neural_Bradley-Terry/main/readme_images/P1.jpg" width="50%" height="50%">
<img src="https://raw.githubusercontent.com/lishiyu0088/Neural_Bradley-Terry/main/readme_images/P1.png" width="50%" height="50%">

Architecture images and scores

![Images](https://raw.githubusercontent.com/lishiyu0088/Neural_Bradley-Terry/main/readme_images/B1.jpg)
<img src="https://raw.githubusercontent.com/lishiyu0088/Neural_Bradley-Terry/main/readme_images/B1.png" width="50%" height="50%">
## Pre-train model

The pretrained models of AVA dataset can be found in the folder result_train/alex, lsim, squeeze and vgg.

## Bradley-Terry model

The implementation of Bradley-Terry model in MATLAB can be found in the website http://personal.psu.edu/drh20/code/btmatlab/.

## Usage
```

AVA dataset by GPU: python main_AVA_GPU.py
AVA dataset by CPU: python main_AVA_GPU.py
Our dataset by GPU: python main_GPU.py
Our dataset by CPU: python main_GPU.py

Selection of models: --model vgg/alex/squeeze/lsim
Example of command: python main_GPU.py --model vgg
```


## Contributing

Contributing are welcome


## License

This project is licensed under the MIT License

## Acknowledgments

The code can only be used for academic purposes. Any commercial uses should be declared and permitted in advance.
