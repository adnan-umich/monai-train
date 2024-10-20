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
    Rotate,
    Randomizable,
    Transform,
    RandAffined,
)


class VerifyImageLabelDimensions(Transform):
    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        
        # Extract the shape directly from the MetaTensor objects
        image_shape = image.shape
        label_shape = label.shape
        
        # Check if the dimensions match
        if image_shape != label_shape:
            raise ValueError(f"Dimension mismatch: Image {image_shape}, Label {label_shape}")
        
        return data

def mtrain_transforms(image_size, roi_size):
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            VerifyImageLabelDimensions(),  # Add verification step here
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            #ResizeD(keys=["image", "label"],spatial_size=(256,256,256)), # Unet
            ResizeD(keys=["image", "label"],spatial_size=(image_size)), # Unetr
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                #spatial_size=(256,256,256), # Unet
                spatial_size=(roi_size), # Swin Unetr
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            #RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(image_size),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            VerifyImageLabelDimensions(),  # Add verification step here
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            #ResizeD(keys=["image", "label"],spatial_size=(256,256,256)), # Unet
            ResizeD(keys=["image", "label"],spatial_size=(image_size)), # Unetr
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    
    return train_transforms, val_transforms

def kfold_transforms(image_size, roi_size):
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            VerifyImageLabelDimensions(),  # Add verification step here
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            #ResizeD(keys=["image", "label"],spatial_size=(256,256,256)), # Unet
            ResizeD(keys=["image", "label"],spatial_size=(image_size)), # Unetr
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                #spatial_size=(256,256,256), # Unet
                spatial_size=(roi_size), # Swin Unetr
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            #RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(image_size),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            VerifyImageLabelDimensions(),  # Add verification step here
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            #ResizeD(keys=["image", "label"],spatial_size=(256,256,256)), # Unet
            ResizeD(keys=["image", "label"],spatial_size=(image_size)), # Unetr
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )
    
    return train_transforms, val_transforms