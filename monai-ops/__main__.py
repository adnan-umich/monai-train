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
import sys, importlib
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_param_importances, plot_pareto_front, plot_timeline
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
from monai.utils import first, set_determinism
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
from abc import ABC, abstractmethod
from optuna.trial import TrialState

class Gen_Figures(object):
        # Record figures
        def __init__(self, aim_run):
            self.callback_data = None
            self.aim_run = aim_run

        def __call__(self, study, trial):
            self.callback_data = None
            try:
                aim_run = self.aim_run
                aim_run.track(aim.Figure(plot_optimization_history(study, target=lambda t: t.values[0], target_name="Mean DICE")), name=f"Optimization History Plot")
            
                fig = plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="Mean DICE")
                fig['layout']['height'] = 400
                fig['layout']['width'] = 1200
                # flip the colourbar
                fig.data[0].line.reversescale = not fig.data[0].line.reversescale
                fig.data[0].line.colorscale = 'purpor'
                aim_run.track(aim.Figure(fig), name=f"Plot Parallel Coordinate", context={"metric":"Mean DICE"}) 
                
                fig = plot_parallel_coordinate(study, target=lambda t: t.values[1], target_name="Average Loss")   
                fig['layout']['height'] = 400
                fig['layout']['width'] = 1200
                fig.data[0].line.reversescale = not fig.data[0].line.reversescale
                fig.data[0].line.colorscale = 'purpor'
                aim_run.track(aim.Figure(fig), name=f"Plot Parallel Coordinate", context={"metric":"Average Training Loss"})
                
                fig = plot_parallel_coordinate(study, target=lambda t: t.values[2], target_name="Validation Loss")
                fig['layout']['height'] = 400
                fig['layout']['width'] = 1200
                fig.data[0].line.reversescale = not fig.data[0].line.reversescale
                fig.data[0].line.colorscale = 'purpor'
                aim_run.track(aim.Figure(fig), name=f"Plot Parallel Coordinate", context={"metric":"Average Validation Loss"})

                aim_run.track(aim.Figure(plot_slice(study, target=lambda t: t.values[0], target_name="Mean DICE")), name=f"Plot Slice")   
                aim_run.track(aim.Figure(plot_param_importances(study, target=lambda t: t.values[0], target_name="Mean DICE")), name=f"Parameter Importance")   
                aim_run.track(aim.Figure(plot_timeline(study)), name=f"Plot Timeline")

                fig = optuna.visualization.plot_pareto_front(study)
                fig['layout']['height'] = 800
                fig['layout']['width'] = 1200
                aim_run.track(aim.Figure(fig), name=f"Pareto Front")
            except:
                return None

def execute():
    optuna_config, data_dir, output_dir, seed = parse_args(create_parser())
    set_determinism(seed=seed)
    aim_run = aim.Run(experiment='Monai-Optuna MLOps')

    if not os.path.exists(output_dir):
        print(f'Output directory {output_dir} does not exist. Creating it.')
        os.makedirs(output_dir, exist_ok=True)
    
    if optuna_config['optuna']['settings']['sampling'] == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "BaseSampler":
        sampler = optuna.samplers.BaseSampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "GridSampler":
        sampler = optuna.samplers.GridSampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "CmaEsSampler":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "PartialFixedSampler":
        sampler = optuna.samplers.PartialFixedSampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "NSGAIISampler":
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "MOTPESampler":
        sampler = optuna.samplers.MOTPESampler(seed=seed)
    elif optuna_config['optuna']['settings']['sampling'] == "IntersectionSearchSpace":
        sampler = optuna.samplers.IntersectionSearchSpace(seed=seed)

    aim_run["Model"] = optuna_config['model']['type']
    aim_run["Model_architecture"] = optuna_config['model']['architecture']
    aim_run["Optuna_settings"] = optuna_config['optuna'] 

    study = optuna.create_study(study_name="Monai-Optuna MLOps",sampler=sampler,directions=['maximize','minimize','minimize'])
    gen_figures = Gen_Figures(aim_run)
    study.optimize(nokfold_objective_training, n_trials=optuna_config['optuna']['settings']['trials'], callbacks=[gen_figures])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("\nPareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: Mean DICE score={}, Average Training Loss={}, Average Validation Loss={}".format(trial.values[0], trial.values[1], trial.values[2]))
        print("    Params: {}".format(trial.params))

    trial_with_highest_dice = max(study.best_trials, key=lambda t: t.values[0])
    print(f"\nTrial with highest mean dice score: ")
    print(f"\tnumber: {trial_with_highest_dice.number}")
    print(f"\tparams: {trial_with_highest_dice.params}")
    print(f"\tvalues (mean dice, average training loss, average validation loss): {trial_with_highest_dice.values}")

    trial_with_lowest_training_loss = min(study.best_trials, key=lambda t: t.values[1])
    print(f"\nTrial with lowest average training loss: ")
    print(f"\tnumber: {trial_with_lowest_training_loss.number}")
    print(f"\tparams: {trial_with_lowest_training_loss.params}")
    print(f"\tvalues (mean dice, average training loss, average validation loss): {trial_with_lowest_training_loss.values}")

    trial_with_lowest_validation_loss = min(study.best_trials, key=lambda t: t.values[2])
    print(f"\nTrial with lowest average validation loss: ")
    print(f"\tnumber: {trial_with_lowest_validation_loss.number}")
    print(f"\tparams: {trial_with_lowest_validation_loss.params}")
    print(f"\tvalues (mean dice, average training loss, average validation loss): {trial_with_lowest_validation_loss.values}")

    aim_run.close()
    return None

def nokfold_objective_training(trial):
    optuna_config, data_dir, output_dir, seed = parse_args(create_parser())
    slice_to_track = optuna_config['model']['slice_to_track']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = trial.suggest_int('epochs', optuna_config['optuna']['hyperparam']['epoch'][0], optuna_config['optuna']['hyperparam']['epoch'][1])
    lr = trial.suggest_float("lr", optuna_config['optuna']['hyperparam']['learning_rate'][0], optuna_config['optuna']['hyperparam']['learning_rate'][1], log=True)
    batch = trial.suggest_int("batch", optuna_config['optuna']['hyperparam']['batch'][0], optuna_config['optuna']['hyperparam']['batch'][1], log=True)
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    loss_type = trial.suggest_categorical("loss_func", ["DiceLoss", "DiceCELoss"])
    ## Load Data
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

    # Split training data into training and validation
    split = optuna_config['optuna']['settings']['split']
    train_size = int(split * len(data_dicts))
    val_size = len(data_dicts) - train_size
    train_files, val_files = torch.utils.data.random_split(data_dicts, [train_size, val_size])

    # get transformations
    train_transforms, val_transforms = mtrain.transformer.mtrain_transforms(optuna_config['model']['image_size'], roi_size=optuna_config['model']['validation_roi'])

    # Create training dataloader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)

    # Create validation dataloader
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    ## Generate model, loss function, optimizers    
    model_type = optuna_config['model']['type']
    metric_type = loss_type

    ## MODEL ##
    if model_type == "UNet":
        model = UNet(**optuna_config['model']['architecture']).to(device)
    elif model_type == "UNetr":
        model = UNETR(**optuna_config['model']['architecture']).to(device)
    elif model_type == "BasicUNet":
        model = BasicUNet(**optuna_config['model']['architecture']).to(device)

    ## EVALUATION METRIC ##
    if metric_type == "DiceLoss":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
    elif metric_type == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean")

    ## OPTIMIZATION ##
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

    Optimizer_metadata = {}
    for ind, param_group in enumerate(optimizer.param_groups):
        optim_meta_keys = list(param_group.keys())
        Optimizer_metadata[f"param_group_{ind}"] = {
            key: value for (key, value) in param_group.items() if "params" not in key
        }

     #### TRAINING STEPS BELOW ####
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(n_epochs):
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

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

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
            outputs = sliding_window_inference(inputs=inputs, roi_size=optuna_config['model']['validation_roi'], sw_batch_size=4, predictor=model)
            #outputs = model(inputs)
            _loss = loss_function(outputs, labels)
            _loss.backward()
            #optimizer.step()
            val_loss += _loss.item()

        val_loss /= step
        val_loss_values.append(val_loss)
        #### val loss end

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for index, val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, optuna_config['model']['validation_roi'], sw_batch_size, model)

                    # tracking input, label and output images with Aim
                    output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()

                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    #torch.save(model.state_dict(), os.path.join(output_dir, f"max_epochs{n_epochs}_lr{lr}_batch{batch}__model.pth"))

                    #best_model_log_message = f"saved new best metric model at the {epoch+1}th epoch"
                    #print(best_model_log_message)

                message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                message2 = f"\nbest mean dice: {best_metric:.4f} "
                message3 = f"at epoch: {best_metric_epoch}"

                #print(message1, message2, message3)
    return best_metric, epoch_loss, val_loss

def create_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('MONAI Optimization Targets')
    g.add_argument(
        '--optuna',
        dest='config_file',
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
        '--seed',
        dest='seed',
        type=int, 
        default=0)
    return parser

def parse_args(parser):
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as stream:
            optuna_config = yaml.safe_load(stream)
    if args.data_dir:
        data_dir = args.data_dir
    if args.output_dir:
        output_dir = args.output_dir
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 0

    return [optuna_config, data_dir, output_dir, seed]


if __name__ == "__main__":
    # Add the parent directory of monai-train and monai-ops to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    monai_train = "monai-train"
    mtrain = importlib.import_module(monai_train)
    execute()
