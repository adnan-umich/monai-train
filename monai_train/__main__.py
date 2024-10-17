import yaml
import torch
import aim
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
import plotly.graph_objects as go
import monai.networks.nets
import monai.losses
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,  # noqa: F401
    EnsureChannelFirstd,  # noqa: F401
    Compose,
    CropForegroundd,  # noqa: F401
    LoadImaged,  # noqa: F401
    Orientationd,  # noqa: F401
    RandCropByPosNegLabeld,  # noqa: F401
    SaveImaged,  # noqa: F401
    ScaleIntensityRanged,  # noqa: F401
    Spacingd,  # noqa: F401
    Invertd,  # noqa: F401
    ResizeD,  # noqa: F401
    LoadImage,  # noqa: F401
    Rotate,  # noqa: F401
    Randomizable,  # noqa: F401
    Transform,  # noqa: F401
    RandAffined,  # noqa: F401
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, Dataset, decollate_batch, ThreadDataLoader
from monai.config import print_config
from monai.apps import CrossValidation
from aim.pytorch import track_gradients_dists, track_params_dists
from abc import ABC, abstractmethod
from collections import defaultdict
import matplotlib
from monai_train.transformer import mtrain_transforms, kfold_transforms
from monai_train.earlystop import EarlyStopping
matplotlib.interactive(False)


class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """
    
    def __init__(
        self,
        data,
        transform,
        cache_rate=1.0,
        num_workers=4,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class CustomDataset(Dataset):
    def __init__(self, image_label_pairs):
        self.image_label_pairs = image_label_pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        sample = self.image_label_pairs[idx]
        image_path = sample['image']
        label_path = sample['label']
        # Load the image and label (modify this if using a specific loader, e.g., nibabel for .nii.gz)
        image = self.load_image(image_path)
        label = self.load_image(label_path)
        return image, label

    def load_image(self, path):
        # Implement image loading here (e.g., nibabel for .nii.gz files)
        # For now, this is a placeholder function
        return path
    
class CustomDataloader:
    def __init__(self, image_label_pairs, split_ratio=0.8, batch_size=1, seed=0):
        self.image_label_pairs = image_label_pairs
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.seed = seed

        # Group images
        self.grouped_images = self.group_images()

        # Split into training and validation sets
        self.training_set, self.validation_set = self.split_groups()

        # Create datasets
        self.train_dataset = CustomDataset(self.training_set)
        self.val_dataset = CustomDataset(self.validation_set)

    def group_images(self):
        """Group images by a common identifier (assumed to be part of the file name).
        Requested by Olivia & Tomer (Umich)"""
        groups = defaultdict(list)

        for pair in self.image_label_pairs:
            # Assuming the group identifier is part of the filename after "cropped filtered"
            group_id = pair['image'].split('_', 3)[0]  # Adjust as needed for your filenames
            groups[group_id].append(pair)
        return list(groups.values())

    def split_groups(self):
        """Split the groups into training and validation sets using torch.utils.data.random_split with a seed."""
        total_groups = len(self.grouped_images)
        split_index = int(total_groups * self.split_ratio)

        # Set up a generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)

        # Perform random split
        train_groups, val_groups = torch.utils.data.random_split(self.grouped_images, [split_index, total_groups - split_index], generator=generator)

        # Flatten the groups into individual image-label pairs
        training_set = [item for group in train_groups for item in group]
        validation_set = [item for group in val_groups for item in group]

        return training_set, validation_set


def get_data_dict(data_dir: str) -> list(): # type: ignore
    """
    Return data list for kfold preparation

    Args:
        data_dir (str): Path to the directory containing the data.
    
    Returns:
        list: A list containing all training & validation data.

    """
    try:
        # Check if data_dir exists
        if not os.path.isdir(data_dir):
            raise Exception(f"The directory '{data_dir}' does not exist.")            

        # Check for the presence of required folders
        required_folders = ["imagesTr", "labelsTr", "imagesTs"]
        for folder in required_folders:
            if not os.path.isdir(os.path.join(data_dir, folder)):
                raise Exception(f"The directory '{folder}' does not exist in '{data_dir}'.")

        # Check if each image in imagesTr has a corresponding label in labelsTr
        imagesTr_files = os.listdir(os.path.join(data_dir, "imagesTr"))
        labelsTr_files = os.listdir(os.path.join(data_dir, "labelsTr"))
        for image_file in imagesTr_files:
            if image_file not in labelsTr_files:
                raise Exception(f"No matching label found for the image '{image_file}' in 'labelsTr' folder.")

        print("All conditions met.")
    except Exception as e:
        print("Error:", e)

    # Get list of training images, and their labels
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    return data_dicts
    
def load_data(data_dir: str, split: float, cache_rate:float, workers: int, batch_size:int, image_size:tuple, roi_size:tuple, seed:int, group_similar:bool) -> list(): # type: ignore
    """
    Load data for training and validation.

    Args:
        data_dir (str): Path to the directory containing the data.
        split (float): Percentage of data to be used for training (0 to 1).
        train_transforms: Transformations to be applied to training data.
        val_transforms: Transformations to be applied to validation data.
        cache_rate (float): Percentage of data to cache.
        workers (int): Number of worker processes for data loading.
        batch_size (int): Batch size for data loading.

    Returns:
        list: A list containing the training DataLoader, validation DataLoader, and training CacheDataset.
    
    Raises:
        Exception: If the data directory or required folders do not exist, or if there are missing labels for images.

    """
    try:
        # Check if data_dir exists
        if not os.path.isdir(data_dir):
            raise Exception(f"The directory '{data_dir}' does not exist.")            

        # Check for the presence of required folders
        required_folders = ["imagesTr", "labelsTr", "imagesTs"]
        for folder in required_folders:
            if not os.path.isdir(os.path.join(data_dir, folder)):
                raise Exception(f"The directory '{folder}' does not exist in '{data_dir}'.")

        # Check if each image in imagesTr has a corresponding label in labelsTr
        imagesTr_files = os.listdir(os.path.join(data_dir, "imagesTr"))
        labelsTr_files = os.listdir(os.path.join(data_dir, "labelsTr"))
        for image_file in imagesTr_files:
            if image_file not in labelsTr_files:
                raise Exception(f"No matching label found for the image '{image_file}' in 'labelsTr' folder.")

        print("All conditions met.")
    except Exception as e:
        print("Error:", e)

    # Get list of all training images, and their labels
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    if group_similar:
        custom_dataloader = CustomDataloader(data_dicts, split_ratio=split, batch_size=batch_size, seed=seed)
        # Flatten the lists (since we grouped them earlier)
        training_set, validation_set = custom_dataloader.split_groups()
    else:
        # Split training data into training and validation
        train_size = int(split * len(data_dicts))
        val_size = len(data_dicts) - train_size
        training_set, validation_set = torch.utils.data.random_split(data_dicts, [train_size, val_size])

    """
    print("***********", end="\n\n")
    for x in training_set:
        print(x)
    print("***********", end="\n\n")
    for x in validation_set:
        print(x)
    import sys
    sys.exit(0)
    """

    # get transformations
    train_transforms, val_transforms = mtrain_transforms(image_size, roi_size=roi_size)

    # Create training dataloader
    train_ds = CacheDataset(data=training_set, transform=train_transforms, cache_rate=cache_rate, num_workers=workers)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create validation dataloader
    val_ds = CacheDataset(data=validation_set, transform=val_transforms, cache_rate=cache_rate, num_workers=workers)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, num_workers=0)

    return [train_loader, val_loader, train_ds, val_ds]

def gen_model(aim_run, model_type:str, architecture:dict, optimizer_type:str, metric:dict, learning_rate:float, b1:float, b2:float, weight_decay:float):
    model_type = model_type
    learning_rate = learning_rate
    metric_type = metric['type']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## MODEL ##
    model = getattr(monai.networks.nets, model_type)(**architecture).to(device)

     ## EVALUATION METRIC ##
    if metric_type == "FocalLoss":
        loss_function = getattr(monai.losses, metric_type)(to_onehot_y=True, use_softmax=True)
    else:
        loss_function = getattr(monai.losses, metric_type)(to_onehot_y=True, softmax=True)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    ## OPTIMIZATION ##
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), learning_rate, betas=(b1, b2), weight_decay=weight_decay)

    # log model metadata
    if aim_run is not None:
        aim_run["Model_metadata"] = architecture
        aim_run["Model"] = model_type

        Optimizer_metadata = {}
        for ind, param_group in enumerate(optimizer.param_groups):
            optim_meta_keys = list(param_group.keys())
            Optimizer_metadata[f"param_group_{ind}"] = {
                key: value for (key, value) in param_group.items() if "params" not in key
            }
        # log optimizer metadata
        aim_run["Optimizer_metadata"] = Optimizer_metadata

    return [model, loss_function, dice_metric, optimizer]

def execute():
    """ Check if we are using k-fold cross validation.
    """

    kfold = parse_args(create_parser())[-1]
    group_similar = parse_args(create_parser())[-2]

    if kfold is not None and kfold > 0:
        print(f"A {kfold}-fold based training will start.")
        kfold_training()
    else:
        print("Disabled k-fold training")
        if group_similar:
            print("Enabled group-similar")
        else:
            print("Disabled group-similar")
        train_no_kfold()

def train_no_kfold():
    # initialize a new Aim Run
    try:
        # Extracting variables
        config = parse_args(create_parser())[0]['model']
        data_dir, output_dir, transfer_learning, split, learning_rate, max_epochs, batch_size, seed, savemodel, group_similar, early_stopping, early_stopping_params, kfold = parse_args(create_parser())[1:]
        model_type = config['type']
        architecture = config['architecture']
        optimizer_dict = config['optimizer']
        beta_1 = config['beta_1']
        beta_2 = config['beta_2']
        weight_decay = config['weight_decay']
        metric_dict = config['metric']
        loss_type =  config['metric']['type']
        roi_size = config['validation_roi']
        image_size = config['image_size']
        slice_to_track = config['slice_to_track']
        
        # Creating a formatted print message
        print("\n=== Training Configurations ===")
        print(f"  Model Type       : {config['type']}")
        print(f"  Architecture  : {config['architecture']}")
        print(f"  Optimizer        : {config['optimizer']}")
        print(f"  Metric Type      : {config['metric']['type']}")
        print(f"  Validation ROI   : {config['validation_roi']}")
        print(f"  Data Directory   : {data_dir}")
        print(f"  Output Directory : {output_dir}")
        print(f"  Transfer Learning: {transfer_learning}")
        print(f"  Split            : {split}")
        print(f"  Learning Rate    : {learning_rate}")
        print(f"  beta 1           : {beta_1}")
        print(f"  beta 2           : {beta_2}")
        print(f"  weight decay     : {weight_decay}")
        print(f"  Max Epochs       : {max_epochs}")
        print(f"  Batch Size       : {batch_size}")
        print(f"  Image Size       : {image_size}")
        print("  No k-fold cross-valdiation")
        print(f"  Seed             : {seed}")
        if group_similar:
             print("Grouping similar images (on linear-transformations)")
        print("=============================\n")
    except:
        print("Training configuration failed to load. Exiting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    aim_run = aim.Run()
    set_determinism(seed=seed)
    # Step 0
    if not os.path.exists(output_dir):
        print(f'Output directory {output_dir} does not exist. Creating it.')
        os.makedirs(output_dir, exist_ok=True)

    # Step 1
    train_loader, val_loader, train_ds, val_ds = load_data(data_dir, split, 1.0, 4, batch_size=batch_size, image_size=image_size, roi_size=roi_size, seed=seed, group_similar=group_similar)
    # Step 2
    model, loss_function, dice_metric, optimizer = gen_model(aim_run, model_type, architecture, optimizer_dict, metric_dict, learning_rate, beta_1, beta_2, weight_decay)

    #### TRAINING STEPS BELOW ####
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    ### Transfer Learning ###
    if transfer_learning is not None:
        model.load_state_dict(torch.load(transfer_learning))
        model.eval()

    # log max epochs
    aim_run["max_epochs"] = max_epochs
    # log batch size
    aim_run["batch_size"] = batch_size
    # log whether kfold, 0 indicates no kfold cross validation
    aim_run["kfold"] = 0

    if early_stopping:
        early_stopper = EarlyStopping(min_epochs=early_stopping_params[0], patience=early_stopping_params[1], threshold=early_stopping_params[2])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        val_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            # track batch loss metric
            aim_run.track(loss.item(), name="batch_loss", context={"type": loss_type})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        # track epoch loss metric
        aim_run.track(epoch_loss, name="epoch_loss", context={"type": loss_type})

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        #### val loss start
        step = 0
        model.eval()
        for batch_data in val_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            #optimizer.zero_grad()
            outputs = sliding_window_inference(inputs=inputs, roi_size=roi_size, sw_batch_size=4, predictor=model)
            #outputs = model(inputs)
            _loss = loss_function(outputs, labels)
            _loss.backward()
            #optimizer.step()
            val_loss += _loss.item()
            print(f"{step}/{len(val_ds) // val_loader.batch_size}, " f"validation_loss: {_loss.item():.4f}")
            # track batch loss metric
            aim_run.track(_loss.item(), name="val_loss", context={"type": loss_type})

        val_loss /= step
        val_loss_values.append(val_loss)

        print(f"epoch {epoch + 1} average validation loss: {val_loss:.4f}")
        #### val loss end

        if (epoch + 1) % val_interval == 0:
            if (epoch + 1) % val_interval * 2 == 0:
                # track model params and gradients
                track_params_dists(model, aim_run)
                # THIS SEGMENT TAKES RELATIVELY LONG (Advise Against it)
                track_gradients_dists(model, aim_run)

            model.eval()
            with torch.no_grad():
                for index, val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    aim_run["validation_roi"] = roi_size
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

                    # tracking input, label and output images with Aim
                    output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()

                    aim_run.track(
                        aim.Image(val_inputs[0, 0, :, :, slice_to_track], caption=f"Input Image: {index}"),
                        name="validation",
                        context={"type": "input"},
                    )
                    aim_run.track(
                        aim.Image(val_labels[0, 0, :, :, slice_to_track], caption=f"Label Image: {index}"),
                        name="validation",
                        context={"type": "label"},
                    )
                    aim_run.track(
                        aim.Image(output, caption=f"Predicted Label: {index}"),
                        name="predictions",
                        context={"type": "labels"},
                    )

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # track val metric
                aim_run.track(metric, name="val_metric", context={"type": loss_type})

                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(output_dir, "best_metric_model.pth"))

                    best_model_log_message = f"saved new best metric model at the {epoch+1}th epoch"
                    aim_run.track(aim.Text(best_model_log_message), name="best_model_log_message", epoch=epoch + 1)
                    print(best_model_log_message)

                message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                message2 = f"\nbest mean dice: {best_metric:.4f} "
                message3 = f"at epoch: {best_metric_epoch}"

                aim_run.track(aim.Text(message1 + "\n" + message2 + message3), name="epoch_summary", epoch=epoch + 1)
                print(message1, message2, message3)

                # Check for early stopping
                if early_stopping:
                    if early_stopper.should_stop(epoch+1, metric):
                        print(f"Early stopping at epoch {epoch+1}")
                        break

    def inference_fig():
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_metric_model.pth")))
        model.eval()
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
                ### Plotly figure ###

                vol = val_data["image"][0, 0, :, :, :].detach().cpu()
                mask = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, :]

                volume = vol.T * mask.T
                r, c = volume[0].shape
                nb_frames = volume.shape[2]

                fig = go.Figure(frames=[go.Frame(data=go.Surface(
                    z=(nb_frames - k) * np.ones((r, c)),
                    surfacecolor=np.flipud(volume[nb_frames - 1 - k]),
                    cmin=0, cmax=1
                    ),
                    name=str(k) # you need to name the frame for the animation to behave properly
                    )
                    for k in range(nb_frames)])

                # Add data to be displayed before animation starts
                fig.add_trace(go.Surface(
                    z=nb_frames * np.ones((r, c)),
                    surfacecolor=np.flipud(volume[nb_frames-1]),
                    colorscale='Gray',
                    cmin=0, cmax=1,
                    colorbar=dict(thickness=20, ticklen=4)
                    ))

                def frame_args(duration):
                    return {
                            "frame": {"duration": duration},
                            "mode": "immediate",
                            "fromcurrent": True,
                            "transition": {"duration": duration, "easing": "linear"},
                        }

                sliders = [
                            {
                                "pad": {"b": 10, "t": 60},
                                "len": 0.9,
                                "x": 0.1,
                                "y": 0,
                                "steps": [
                                    {
                                        "args": [[f.name], frame_args(0)],
                                        "label": str(k),
                                        "method": "animate",
                                    }
                                    for k, f in enumerate(fig.frames)
                                ],
                            }
                        ]

                # Layout
                fig.update_layout(
                        title='Slices in volumetric data',
                        width=800,
                        height=800,
                        scene=dict(
                                    zaxis=dict(range=[0, nb_frames], autorange=False),
                                    aspectratio=dict(x=1, y=1, z=1),
                                    ),
                        updatemenus = [
                            {
                                "buttons": [
                                    {
                                        "args": [None, frame_args(50)],
                                        "label": "&#9654;", # play symbol
                                        "method": "animate",
                                    },
                                    {
                                        "args": [[None], frame_args(0)],
                                        "label": "&#9724;", # pause symbol
                                        "method": "animate",
                                    },
                                ],
                                "direction": "left",
                                "pad": {"r": 10, "t": 70},
                                "type": "buttons",
                                "x": 0.1,
                                "y": 0,
                            }
                        ],
                        sliders=sliders
                )

                ####################

                aim_run.track(aim.Figure(fig), name=f"test_prediction_{i}")   
                plt.close()
        return None
    try:
        inference_fig()
    except:
        print("Generate inference figures failed.")

    # finalize Aim Run
    aim_run.close()
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

def kfold_training():
    # initialize a new Aim Run
    try:
        # Extracting variables
        config = parse_args(create_parser())[0]['model']
        data_dir, output_dir, transfer_learning, split, learning_rate, max_epochs, batch_size, seed, savemodel, _, early_stopping, early_stopping_params, kfold = parse_args(create_parser())[1:]
        model_type = config['type']
        architecture = config['architecture']
        optimizer_dict = config['optimizer']
        beta_1 = config['beta_1']
        beta_2 = config['beta_2']
        weight_decay = config['weight_decay']
        metric_dict = config['metric']
        loss_type =  config['metric']['type']
        roi_size = config['validation_roi']
        image_size = config['image_size']
        slice_to_track = config['slice_to_track']
        
        # Creating a formatted print message
        print("\n=== Training Configurations ===")
        print(f"  Model Type       : {config['type']}")
        print(f"  Architecture  : {config['architecture']}")
        print(f"  Optimizer        : {config['optimizer']}")
        print(f"  Metric Type      : {config['metric']['type']}")
        print(f"  Validation ROI   : {config['validation_roi']}")
        print(f"  Data Directory   : {data_dir}")
        print(f"  Output Directory : {output_dir}")
        print(f"  Transfer Learning: {transfer_learning}")
        print(f"  beta 1           : {beta_1}")
        print(f"  beta 2           : {beta_2}")
        print(f"  weight decay     : {weight_decay}")
        print(f"  Split            : {split}")
        print(f"  Learning Rate    : {learning_rate}")
        print(f"  Max Epochs       : {max_epochs}")
        print(f"  Batch Size       : {batch_size}")
        print(f"  Image Size       : {image_size}")
        print(f"  {kfold}-fold cross-valdiation")
        print(f"  Seed             : {seed}")
    except:
        print("Training configuration failed to load. Exiting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    aim_run = aim.Run(experiment='Monai-Trainer')
    set_determinism(seed=seed)
    # Step 0
    if not os.path.exists(output_dir):
        print(f'Output directory {output_dir} does not exist. Creating it.')
        os.makedirs(output_dir, exist_ok=True)

    # Step 1
    folds = list(range(kfold))

    data_dicts = get_data_dict(data_dir)
    # get transformations
    train_transforms, val_transforms = kfold_transforms(image_size, roi_size=roi_size)

    cvdataset = CrossValidation(
        dataset_cls=CVDataset,
        data=data_dicts,
        nfolds=kfold,
        seed=seed,
        transform=train_transforms,
    )

    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=val_transforms) for i in range(kfold)]

    train_loaders = [ThreadDataLoader(train_dss[i], batch_size=batch_size, shuffle=True, num_workers=0) for i in folds]
    val_loaders = [ThreadDataLoader(val_dss[i], batch_size=1, num_workers=0) for i in folds]

    # log max epochs
    aim_run["max_epochs"] = max_epochs
    # log batch size
    aim_run["batch_size"] = batch_size
    # log whether kfold, 0 indicates no kfold cross validation
    aim_run["kfold"] = kfold

    #### TRAINING STEPS BELOW ####
    def train(fold, slice_to_track):
        print(f"=============== Training for fold {fold} ===============")
         # Step 2
        model, loss_function, dice_metric, optimizer = gen_model(aim_run, model_type, architecture, optimizer_dict, metric_dict, learning_rate, beta_1, beta_2, weight_decay)
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        val_loss_values = []
        metric_values = []
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        

        ### Transfer Learning ###
        if transfer_learning is not None:
            model.load_state_dict(torch.load(transfer_learning))
            model.eval()
        
        if early_stopping:
            early_stopper = EarlyStopping(min_epochs=early_stopping_params[0], patience=early_stopping_params[1], threshold=early_stopping_params[2])

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs} of fold {fold+1}")
            model.train()
            epoch_loss = 0
            val_loss = 0
            step = 0
            for batch_data in train_loaders[fold]:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{len(train_dss[fold]) // train_loaders[fold].batch_size}, " f"train_loss: {loss.item():.4f}")
                # track batch loss metric
                aim_run.track(loss.item(), name="batch_loss", context={"type": loss_type, 'kfold': fold})
        
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
        
            # track epoch loss metric
            aim_run.track(epoch_loss, name="epoch_loss", context={"type": loss_type, 'kfold': fold})
        
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            #### val loss start
            step = 0
            for batch_data in val_loaders[fold]:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                #optimizer.zero_grad()
                outputs = sliding_window_inference(inputs=inputs, roi_size=roi_size, sw_batch_size=4, predictor=model)
                _loss = loss_function(outputs, labels)
                _loss.backward()
                #optimizer.step()
                val_loss += _loss.item()
                print(f"{step}/{len(val_dss[fold]) // val_loaders[fold].batch_size}, " f"validation_loss: {_loss.item():.4f}")
                # track batch loss metric
                aim_run.track(_loss.item(), name="val_loss", context={"type": loss_type, 'kfold': fold})
        
            val_loss /= step
            val_loss_values.append(val_loss)
        
            print(f"epoch {epoch + 1} average validation loss: {val_loss:.4f}")
            #### val loss end
            
            if (epoch + 1) % val_interval == 0:
        
                model.eval()
                with torch.no_grad():
                    for index, val_data in enumerate(val_loaders[fold]):
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        aim_run["validation_roi"] = roi_size
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
        
                        # tracking input, label and output images with Aim
                        output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()
                        
                        aim_run.track(
                            aim.Image(val_inputs[0, 0, :, :, slice_to_track], caption=f"Input Image: {index}"),
                            name="validation",
                            context={"type": "input", 'kfold': fold},
                        )
                        aim_run.track(
                            aim.Image(val_labels[0, 0, :, :, slice_to_track], caption=f"Label Image: {index}"),
                            name="validation",
                            context={"type": "label", 'kfold': fold},
                        )
                        aim_run.track(
                            aim.Image(output, caption=f"Predicted Label: {index}"),
                            name="predictions",
                            context={"type": "labels", 'kfold': fold},
                        )
                        
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)
                    
                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # track val metric
                    aim_run.track(metric, name="val_metric", context={"type": loss_type, 'kfold': fold})
        
                    # reset the status for next validation round
                    dice_metric.reset()
        
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1        
        
                    message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    message2 = f"\nbest mean dice: {best_metric:.4f} "
                    message3 = f"at epoch: {best_metric_epoch}"
        
                    print(message1, message2, message3)

                # Check for early stopping
                if early_stopping:
                    if early_stopper.should_stop(epoch+1, metric):
                        print(f"Early stopping at epoch {epoch+1}")
                        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
                        return model
            
        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
        return model
    
    def save_model():
        # Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        config = parse_args(create_parser())[0]['model']
        data_dir, output_dir, transfer_learning, split, learning_rate, max_epochs, batch_size, seed, savemodel, group_similar, early_stopping, early_stopping_params, kfold = parse_args(create_parser())[1:]
        model_type = config['type']
        architecture = config['architecture']
        optimizer_dict = config['optimizer']
        beta_1 = config['beta_1']
        beta_2 = config['beta_2']
        weight_decay = config['weight_decay']
        metric_dict = config['metric']
        loss_type =  config['metric']['type']
        roi_size = config['validation_roi']
        image_size = config['image_size']
        slice_to_track = config['slice_to_track']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_determinism(seed=seed)
        # Step 0
        if not os.path.exists(output_dir):
            print(f'Output directory {output_dir} does not exist. Creating it.')
            os.makedirs(output_dir, exist_ok=True)

        # Step 1
        train_loader, val_loader, train_ds, val_ds = load_data(data_dir, 1.0, 1.0, 4, batch_size=batch_size, image_size=image_size, roi_size=roi_size, seed=seed, group_similar=group_similar)

        # Step 2
        model, loss_function, dice_metric, optimizer = gen_model(None, model_type, architecture, optimizer_dict, metric_dict, learning_rate, beta_1, beta_2, weight_decay)

        #### TRAINING STEPS BELOW ####
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        ### Transfer Learning ###
        if transfer_learning is not None:
            model.load_state_dict(torch.load(transfer_learning))
            model.eval()

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)

        # Training complete.
        torch.save(model.state_dict(), os.path.join(output_dir, "best_metric_model.pth"))
        return None
                
    [train(i, slice_to_track) for i in range(kfold)]
    # finalize Aim Run
    aim_run.close()
    if savemodel:
        save_model()

def create_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('MONAI Targets')
    g.add_argument(
        '--model',
        dest='model_file',
        type=str, help="path to model configuration file (yaml)")
    g.add_argument(
        '--data',
        dest='data_dir',
        type=str, help="path to training data (dir)")
    g.add_argument(
        '--output',
        dest='output_dir',
        type=str, help="path to output folder (dir)")
    g.add_argument(
        '--split',
        dest='split_percentage',
        type=float,
        default=0.8, help="fraction to split training data into training/validation pair")
    g.add_argument(
        '--lr',
        dest='learning_rate',
        default=0.0001,
        type=float, help="training optimizer's learning rate (float)")
    g.add_argument(
        '--epochs',
        dest='epochs',
        default=100,
        type=int, help="total number of epoch's per training")
    g.add_argument(
        '--batch',
        dest='batch_size',
        default=1,
        type=int, help="training batch size")
    g.add_argument(
        '--transfer',
        dest='transfer_learning',
        type=str, help="path to trained model file (/path/to/file/*.pth) for transfer learning (path)")
    g.add_argument(
        '--kfold',
        dest='kfold',
        type=int, help="total number of K-fold sessions, enable with any value >= 1")
    g.add_argument(
        '--savemodel',
        dest='savemodel',
        type=bool, help="save final trained model with the best mean-dice score")
    g.add_argument(
        '--seed',
        dest='seed',
        type=int, 
        default=0)
    g.add_argument(
        '--group-similar',
        dest='group_similar',
        action="count", help="group together similar images. The training/validation split will not split grouped images. Images are to be grouped by a shared unique ID")
    g.add_argument(
        '--show-config',
        dest='show_config',
        action="count")
    g.add_argument(
        '--early-stopping',
        dest='early_stopping',
        action="count")
    g.add_argument(
        '--min-epochs',
        dest='min_epochs',
        type=int, help="Minimum number of epochs before checking for early-stopping criteria")
    g.add_argument(
        '--patience',
        dest='patience',
        type=int, help="Number of epochs to wait if there is no significant improvement")
    g.add_argument(
        '--threshold',
        dest='threshold',
        type=float, help="Defines what is considered an 'improvement' in the score; if the score change is below this, it's treated as no improvement.")
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.model_file:
        with open(args.model_file, 'r') as stream:
            model = yaml.safe_load(stream)

    if args.data_dir:
        data_dir = args.data_dir
    if args.output_dir:
        output_dir = args.output_dir
    ###
    if args.split_percentage is not None:
        split_percentage = args.split_percentage
    else:
        split_percentage = 0.8
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.epochs is not None:
        epochs = args.epochs
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.kfold is not None:
        kfold = args.kfold
    else:
        kfold = None
    if args.savemodel is not None:
        savemodel = args.savemodel
    else:
        savemodel = False
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 0
    ###
    if args.transfer_learning:
        transfer_learning = args.transfer_learning
    else:
        transfer_learning = None
    
    # Check if early stopping is enabled and other parameters are defined
    if args.early_stopping:
        if args.min_epochs is None or args.patience is None or args.threshold is None:
            parser.error("--min-epochs, --patience, and --threshold are required when --early-stopping is enabled.")

    if args.show_config:
        print_config()
        exit()

    return [model, data_dir, output_dir, transfer_learning, split_percentage, learning_rate, epochs, batch_size, seed, savemodel, args.group_similar, args.early_stopping, [args.min_epochs, args.patience, args.threshold], kfold]


if __name__ == "__main__":
    #print_config()
    execute()
