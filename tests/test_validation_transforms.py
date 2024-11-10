# test_train_transformer.py
import pytest
from monai.data import CacheDataset, ThreadDataLoader

def test_apply_validation_transforms(get_data, validation_transform_pipeline):
    """Load sample data and apply transforms."""
    val_ds = CacheDataset(data=get_data, transform=validation_transform_pipeline, cache_rate=1.0, num_workers=0)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)
    # Assert that train_loader is not None
    assert val_loader is not None, "DataLoader was not created successfully."
    
    # Ensure that at least one batch is returned from the DataLoader
    for batch in val_loader:
        # Assert that the batch contains data and transformations were applied
        assert batch is not None, "Batch is None. The transformations may not have been applied properly."
        assert isinstance(batch, dict), "The batch should be a dictionary with image and label keys."
        assert "image" in batch and "label" in batch, "Batch must contain 'image' and 'label' keys."
        
        # If you have specific assertions about the image or label shape, you can add them
        assert batch["image"].squeeze().shape == (128, 128, 128), "The image shape is incorrect after transformation."
        assert batch["label"].squeeze().shape == (128, 128, 128), "The label shape is incorrect after transformation."
        
        # If everything looks good, break after the first batch (optional)
        break
