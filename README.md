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
--data : location of nifti dataset. Must contain `imageTr` (training set images), `labelsTr` (training set labels) and `imagesTs` (test set).
--split : percentage of training set, remainder is validation set.
--epochs : maximum number of epochs
--batch : maximum batch size for the training set. Validation set to 1.
--transfer (optional) : full path of pretrained model pickle file (.pth) for transfer learning.
```

