import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import yaml
import monai.networks.nets
from tqdm import tqdm
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureType,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    NormalizeIntensity,
    Spacingd,
    Invert,
    Invertd,
    ResizeD,
    Resize,
    LoadImage,
    Rotate,
    Randomizable,
    Transform,
    EnsureType,
)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, ThreadDataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from aim.pytorch import track_gradients_dists, track_params_dists
from matplotlib.widgets import Button, Slider


### USER INPUT REQUIRED ###
# Root path to where the data is located
DATA_DIR = "/nfs/turbo/dent-tomers/Olivia/monai-pipeline-data"
MODEL_NAME = "/nfs/turbo/dent-tomers/Olivia/output2/best_metric_model.pth" # Full path to model pickle file, including name. ex: /home/usr/model.pth
MODEL_CONFIG_FILE = "/home/ogott/monai-train/example/model_unet.yaml" # Full path to model configuration file, This could be the example/model_*.yaml or example/optuna_config.yaml
###
set_determinism(seed=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
with open(MODEL_CONFIG_FILE, 'r') as stream:
    config = yaml.safe_load(stream)

model_type = config['model']['type']
model = getattr(monai.networks.nets, model_type)(**config['model']['architecture'])

# Load model weights with CPU/GPU compatibility
model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode


# Automatically locates the 'imagesTs' folder
test_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTs", "*.nii.gz")))
test_data = [{"image": image} for image in test_images]

test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        ResizeD(keys="image", spatial_size=config['model']['image_size']),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 2.0, 2.0), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #ScaleIntensityd(keys=["image"], minv=0, maxv=256),
        EnsureType()
    ]
)
post_transforms = Compose(
    [
        EnsureType(),
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
    ]
)

# Additional post-processing (apply non-invertible transforms)
additional_post_transforms = Compose(
    [
    ]
)

# Loads data into the data loader
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = ThreadDataLoader(test_org_ds, batch_size=1, num_workers=0)
loader = LoadImage()

with torch.no_grad():
    for test_data in tqdm(test_org_loader):
        test_inputs = test_data["image"].to(device)
        roi_size = 40 # Adjustable parameter to present overall image dimensions
        sw_batch_size = 4
        slice = 40 # Adjustable parameter (slice to visualize in the plots). Note inference is performed on the entire 3D dataset.
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        test_data = [additional_post_transforms(i) for i in test_data]


        test_output = from_engine(["pred"])(test_data)

        original_image = loader(test_output[0].meta["filename_or_obj"])

        plt.figure("check", (8,8))
        plt.subplot(1,3,1)
        plt.imshow(original_image[:, :, slice], cmap="gray")

        mask_pred = np.zeros(original_image[:, :, slice].shape)
        mask_pred[test_output[0].detach().cpu().numpy()[1, :, :, slice]==1] = 1
        masked_pred = np.ma.masked_where(mask_pred == 0, mask_pred)
        plt.subplot(1, 3, 2)
        plt.imshow(original_image[:, :, slice], cmap="gray")
        plt.imshow(masked_pred, alpha=0.7)

        plt.subplot(1,3,3)
        plt.imshow(test_output[0].detach().cpu().numpy()[1, :, :, slice])

        plt.show()
