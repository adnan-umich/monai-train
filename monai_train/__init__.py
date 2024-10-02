import yaml
import torch
import aim
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import argparse
import time
import plotly.graph_objects as go
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    ResizeD,
    LoadImage,
    Rotate,
    Randomizable,
    Transform,
    RandAffined,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, UNETR, SwinUNETR, BasicUNet, SegResNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract, CrossValidation
from aim.pytorch import track_gradients_dists, track_params_dists
from .transformer import mtrain_transforms, kfold_transforms
from abc import ABC, abstractmethod
