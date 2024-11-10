# test_train_transformer.py
import pytest
from monai.data import CacheDataset, ThreadDataLoader


def test_get_data_structure(get_data):
    """Test to verify that get_data fixture returns a list of dictionaries with 'image' and 'label' keys."""
    assert isinstance(get_data, list), "get_data should return a list."
    assert len(get_data) > 0, "get_data should not be empty."

    # Verify each entry is a dictionary with 'image' and 'label' keys
    for entry in get_data:
        assert isinstance(entry, dict), "Each entry in get_data should be a dictionary."
        assert "image" in entry, "Each entry should contain an 'image' key."
        assert "label" in entry, "Each entry should contain a 'label' key."
        assert isinstance(entry["image"], str), "The 'image' key should have a string value."
        assert isinstance(entry["label"], str), "The 'label' key should have a string value."

def test_apply_train_transforms(get_data, transform_pipeline):
    """Load sample data and apply transforms."""
    train_ds = CacheDataset(data=get_data, transform=transform_pipeline, cache_rate=1.0, num_workers=4)
    train_loader = ThreadDataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    # Assert that train_loader is not None
    assert train_loader is not None, "DataLoader was not created successfully."
    
    # Ensure that at least one batch is returned from the DataLoader
    for batch in train_loader:
        # Assert that the batch contains data and transformations were applied
        assert batch is not None, "Batch is None. The transformations may not have been applied properly."
        assert isinstance(batch, dict), "The batch should be a dictionary with image and label keys."
        assert "image" in batch and "label" in batch, "Batch must contain 'image' and 'label' keys."
        
        # If you have specific assertions about the image or label shape, you can add them
        assert batch["image"].squeeze().shape == (128, 128, 128), "The image shape is incorrect after transformation."
        assert batch["label"].squeeze().shape == (128, 128, 128), "The label shape is incorrect after transformation."
        
        # If everything looks good, break after the first batch (optional)
        break
