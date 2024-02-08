## Monai & Pytorch Pipeline with AimStack

![GitHub Issues](https://img.shields.io/github/issues/adnan-umich/monai-train.svg)[![Python application](https://github.com/adnan-umich/monai-train/actions/workflows/python-app.yml/badge.svg)](https://github.com/adnan-umich/monai-train/actions/workflows/python-app.yml)
<img width="1478" alt="Screenshot 2024-02-07 at 11 37 39 PM" src="https://github.com/adnan-umich/monai-train/assets/124732717/6eefafb1-3af7-4e2c-a871-336097cd7b4f">


# Requirements 
  * `python >= 3.9, <3.13`
  * `cuda/12.x`
  * `poetry`
```
(HPC) $ module load python cuda/12.1.1 poetry
```

# Installation 
```
  $ git clone git@github.com:adnan-umich/monai-train.git
  $ cd monai-train
  $ poetry install
```

### Quick start:
```
  $ module load python cuda poetry
  $ poetry shell
  $ (monai-train-py3.11) python -m monai-train --model   = ./example/model_unet.yaml
                                               --data    = /path/to/NIfTI/dataset \
                                               --output  = /output \
                                               --split   = 0.8 \
                                               --lr      = 0.0001 \
                                               --epochs  = 20 \
                                               --batch   = 5
```

### Configurations
```
--model : CNN model yaml file. 
--data: Location of the NIfTI dataset. It must contain a folder named imagesTr,
        which holds the training images that have been labeled. Additionally,
        there should be a folder named labelsTr containing the corresponding labels for the training sets.
        Furthermore, there should be a folder named imagesTs that contains the test images, which are unlabeled.
--output: Location to save output data, such as trained model, and weights.
--split : percentage of training set, remainder is validation set.
--lr : learning rate
--epochs : maximum number of epochs
--batch : maximum batch size for the training set. Validation set to 1.
--transfer (optional) : full path of pretrained model pickle file (.pth) for transfer learning.
```

