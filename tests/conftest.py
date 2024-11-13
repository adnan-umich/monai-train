# conftest.py
import os
import glob
import pytest
from monai.data import CacheDataset, ThreadDataLoader
from monai_train.transformer import mtrain_transforms  # Update this path if needed

def pytest_addoption(parser):
    parser.addoption(
        "--image-dir",
        action="store",
        default='sample',
        help="Path to the directory containing sample images and labels."
    )
    parser.addoption(
        "--image-size",
        action="store",
        default="128,128,128",
        help="Image size for transformations (e.g., '128,128,64')."
    )
    parser.addoption(
        "--roi-size",
        action="store",
        default="64,64,64",
        help="ROI size for transformations (e.g., '64,64,32')."
    )

@pytest.fixture
def get_data(pytestconfig):
    """Fixture to load all images and labels from the specified directory."""
    data_dir = pytestconfig.getoption("--image-dir")
    if not data_dir or not os.path.isdir(data_dir):
        pytest.fail(f"The directory '{data_dir}' does not exist. Specify it using --image-dir.")
        
    # Check for required subfolders
    required_folders = ["imagesTr", "labelsTr", "imagesTs"]
    for folder in required_folders:
        if not os.path.isdir(os.path.join(data_dir, folder)):
            pytest.fail(f"The directory '{folder}' does not exist in '{data_dir}'.")

    # Check if each image in `imagesTr` has a corresponding label in `labelsTr`
    imagesTr_files = sorted(os.listdir(os.path.join(data_dir, "imagesTr")))
    labelsTr_files = sorted(os.listdir(os.path.join(data_dir, "labelsTr")))
    for image_file in imagesTr_files:
        if image_file not in labelsTr_files:
            pytest.fail(f"No matching label found for the image '{image_file}' in 'labelsTr' folder.")

    # Get list of all training images and their labels
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    return data_dicts
    
@pytest.fixture
def transform_pipeline(pytestconfig):
    """Initialize the transformation pipeline with the imported image_size and roi_size."""
    image_size = tuple(map(int, pytestconfig.getoption("--image-size").split(',')))
    roi_size = tuple(map(int, pytestconfig.getoption("--roi-size").split(',')))
    train_transforms, _ = mtrain_transforms(image_size, roi_size=roi_size)
    return train_transforms

@pytest.fixture
def validation_transform_pipeline(pytestconfig):
    """Initialize the transformation pipeline with the imported image_size and roi_size."""
    image_size = tuple(map(int, pytestconfig.getoption("--image-size").split(',')))
    roi_size = tuple(map(int, pytestconfig.getoption("--roi-size").split(',')))
    _, val_transforms = mtrain_transforms(image_size, roi_size=roi_size)
    return val_transforms
