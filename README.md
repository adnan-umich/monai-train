### Installation 
is done by using the [`poetry install`]:

```
bash poetry install .
```

### Requirements 
are `python` `cuda/12.x` `poetry`: 
```
module load python cuda/12.1.1 poetry
```

### Quick start:
```
python -m monai-train --model=./example/model_unet.yaml --data=/path/to/nifti/dataset \
                      --output=/output \
                      --split=0.8 \
                      --lr=0.0001 \
                      --epochs=20 \
                      --batch=5
```

### Configurations
```
--model : CNN model yaml file. 
--data: Location of the NIfTI dataset. It must contain a folder named imageTr,
        which holds the training images that have been labeled. Additionally,
        there should be a folder named labelsTr containing the corresponding labels for the training sets.
        Furthermore, there should be a folder named imagesTs that contains the test images, which are unlabeled.
--split : percentage of training set, remainder is validation set.
--epochs : maximum number of epochs
--batch : maximum batch size for the training set. Validation set to 1.
--transfer (optional) : full path of pretrained model pickle file (.pth) for transfer learning.
```

