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
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, UNETR, SwinUNETR, BasicUNet, SegResNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from aim.pytorch import track_gradients_dists, track_params_dists
from .transformer import train_transforms, val_transforms


def load_data(data_dir: str, split: float, train_transforms, val_transforms, cache_rate:float, workers: int, batch_size:int) -> list():
    # Get list of training images, and their labels
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    # Split training data into training and validation
    train_size = int(split * len(data_dicts))
    val_size = len(data_dicts) - train_size
    train_files, val_files = torch.utils.data.random_split(data_dicts, [train_size, val_size])

    # Create training dataloader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=workers)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=workers)

    # Create validation dataloader
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)

    return [train_loader, val_loader, train_ds]

def gen_model(aim_run, model_type:str, hyperparam:dict, optimizer:dict, metric:dict, learning_rate:float):
    model_type = model_type
    learning_rate = learning_rate
    metric_type = metric['type']
    optimizer_type = optimizer['type']
    device = torch.device("cuda:0")

    ## MODEL ##
    if model_type == "UNet":
        model = UNet(**hyperparam).to(device)
    elif model_type == "UNetr":
        model = UNETR(**hyperparam).to(device)
    elif model_type == "BasicUNet":
        model = BasicUNet(**hyperparam).to(device)

    ## EVALUATION METRIC ##
    if metric_type == "DiceLoss":
        loss_function = DiceLoss(to_onehot_y=True, softmax=metric['softmax'])
        dice_metric = DiceMetric(include_background=metric['include_background'], reduction=metric['reduction'])
    elif metric_type == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=metric['softmax'])
        dice_metric = DiceMetric(include_background=metric['include_background'], reduction=metric['reduction'])

    ## OPTIMIZATION ##
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=1e-5)

    # log model metadata
    aim_run["Model_metadata"] = hyperparam
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
    # initialize a new Aim Run
    device = torch.device("cuda:0")
    aim_run = aim.Run()
    model_type = parse_args(create_parser())[0]['model']['type']
    hyperparam = parse_args(create_parser())[0]['model']['hyperparam']
    optimizer_dict = parse_args(create_parser())[0]['model']['optimizer']
    learning_rate = parse_args(create_parser())[0]['model']['optimizer']['learning_rate']
    batch_size = parse_args(create_parser())[0]['model']['batch_size']
    metric_dict = parse_args(create_parser())[0]['model']['metric']
    data_dir = parse_args(create_parser())[1]
    output_dir = parse_args(create_parser())[2]
    transfer_learning = parse_args(create_parser())[3]

    # Step 0
    if not os.path.exists(output_dir):
        print(f'Output directory {output_dir} does not exist. Creating it.')
        os.makedirs(output_dir, exist_ok=True)

    # Step 1
    train_loader, val_loader, train_ds = load_data(data_dir, 0.8, train_transforms, val_transforms, 1.0, 4, batch_size=batch_size)
    # Step 2
    model, loss_function, dice_metric, optimizer = gen_model(aim_run, model_type, hyperparam, optimizer_dict, metric_dict, learning_rate)
    # Step 3
    max_epochs = metric_dict = parse_args(create_parser())[0]['model']['epochs']
    loss_type = parse_args(create_parser())[0]['model']['metric']['type']

    #### TRAINING STEPS BELOW ####
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    ### Transfer Learning ###
    if len(transfer_learning) > 1:
        model.load_state_dict(torch.load(transfer_learning))
        model.eval()

    # log max epochs
    aim_run["max_epochs"] = max_epochs
    # log batch size
    aim_run["batch_size"] = batch_size
    slice_to_track = 20

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
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            # track batch loss metric
            aim_run.track(loss.item(), name="batch_loss", context={"type": loss_type})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        # track epoch loss metric
        aim_run.track(epoch_loss, name="epoch_loss", context={"type": loss_type})

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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
                    roi_size = (48, 48, 48)
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

    def inference_fig():
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_metric_model.pth")))
        model.eval()
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                roi_size = (48, 48, 48) # used to be 160
                sw_batch_size = 4
                slice = 12
                val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
                # plot the slice [:, :, slice]
                #fig = plt.figure("check", (18, 6))
                
                fig = plt.figure("check", (18,6))
                plt.title(f"image {i}")
                plt.imshow(val_data["image"][0, 0, :, :, slice], cmap="gray")
                mask_pred = np.zeros(val_data["image"][0, 0, :, :, slice].shape)
                mask_pred[torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice]==1] = 1
                masked_pred = np.ma.masked_where(mask_pred == 0, mask_pred)
                plt.imshow(masked_pred, 'Spectral', interpolation='none', alpha=0.7)
                mask_org = np.zeros(val_data["image"][0, 0, :, :, slice].shape)
                mask_org[val_data["label"][0, 0, :, :, slice]==1] = 1
                masked_org = np.ma.masked_where(mask_org == 0, mask_org)
                plt.imshow(masked_org, 'ocean', interpolation='none', alpha=0.3)
                aim_run.track(aim.Image(fig), name=f"final_{index}")   
                plt.close()
                if i == 1:
                    break
        return None

    inference_fig()

    # finalize Aim Run
    aim_run.close()
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

def create_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('MONAI Targets')
    g.add_argument(
        '--model',
        dest='model_file',
        type=str)
    g.add_argument(
        '--data',
        dest='data_dir',
        type=str)
    g.add_argument(
        '--output',
        dest='output_dir',
        type=str)
    g.add_argument(
        '--transfer',
        dest='transfer_learning',
        type=str,
        default='')
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
    if args.transfer_learning:
        output_dir = args.transfer_learning
    return [model, data_dir, output_dir, transfer_learning]


if __name__ == "__main__":
    #print_config()
    execute()