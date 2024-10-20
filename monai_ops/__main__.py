import yaml
import torch
import aim
import matplotlib.pyplot as plt
import os
import glob
import argparse
import plotly.graph_objects as go
import monai.losses
import monai.networks.nets
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
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, Dataset, decollate_batch, ThreadDataLoader
from monai.config import print_config
from monai.apps import CrossValidation
from aim.pytorch import track_gradients_dists, track_params_dists
from optuna.trial import TrialState
from functools import partial
import monai_train as mtrain
from collections import defaultdict


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
                fig.data[0].line.colorscale = 'purpor'
                aim_run.track(aim.Figure(fig), name=f"Plot Parallel Coordinate", context={"metric":"Average Training Loss"})
                
                fig = plot_parallel_coordinate(study, target=lambda t: t.values[2], target_name="Validation Loss")
                fig['layout']['height'] = 400
                fig['layout']['width'] = 1200
                fig.data[0].line.colorscale = 'purpor'
                aim_run.track(aim.Figure(fig), name=f"Plot Parallel Coordinate", context={"metric":"Average Validation Loss"})

                aim_run.track(aim.Figure(plot_slice(study, target=lambda t: t.values[0], target_name="Mean DICE")), name=f"Plot Slice")   
                aim_run.track(aim.Figure(plot_param_importances(study, target=lambda t: t.values[0], target_name="Mean DICE")), name=f"Parameter Importance")   
                aim_run.track(aim.Figure(plot_timeline(study)), name=f"Plot Timeline")

                fig = optuna.visualization.plot_pareto_front(study)
                fig['layout']['height'] = 800
                fig['layout']['width'] = 1200
                
                # Define sliders for adjusting the maximum value of x, y, and z axes
                sliders = [
                    {'steps': [
                        {'method': 'relayout', 'args': ['scene.xaxis.range[1]', 0.2], 'label': 'x-axis (0.2)'},
                        {'method': 'relayout', 'args': ['scene.xaxis.range[1]', 0.5], 'label': 'x-axis (0.5)'},
                        {'method': 'relayout', 'args': ['scene.xaxis.range[1]', 1.0], 'label': 'x-axis (1.0)'},
                        {'method': 'relayout', 'args': ['scene.xaxis.range[1]', 1.5], 'label': 'x-axis (1.5)'},
                    ],
                    'active': 0,
                    'y': 0,
                    'x': -0.1,
                    'len': 0.3,
                    'pad': {'t': 50, 'b': 10},
                    'currentvalue': {'visible': True, 'prefix': 'Max Value: '},
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                    },

                    {'steps': [
                        {'method': 'relayout', 'args': ['scene.yaxis.range[1]', 0.2], 'label': 'y-axis (0.2)'},
                        {'method': 'relayout', 'args': ['scene.yaxis.range[1]', 0.5], 'label': 'y-axis (0.5)'},
                        {'method': 'relayout', 'args': ['scene.yaxis.range[1]', 1.0], 'label': 'y-axis (1.0)'},
                        {'method': 'relayout', 'args': ['scene.yaxis.range[1]', 1.5], 'label': 'y-axis (1.5)'},
                    ],
                    'active': 0,
                    'y': 0,
                    'x': 0.3,
                    'len': 0.3,
                    'pad': {'t': 50, 'b': 20},
                    'currentvalue': {'visible': True, 'prefix': 'Max Value: '},
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                    },

                    {'steps': [
                        {'method': 'relayout', 'args': ['scene.zaxis.range[1]', 0.2], 'label': 'z-axis (0.2)'},
                        {'method': 'relayout', 'args': ['scene.zaxis.range[1]', 0.5], 'label': 'z-axis (0.5)'},
                        {'method': 'relayout', 'args': ['scene.zaxis.range[1]', 1.0], 'label': 'z-axis (1.0)'},
                        {'method': 'relayout', 'args': ['scene.zaxis.range[1]', 1.5], 'label': 'z-axis (1.5)'},
                    ],
                    'active': 0,
                    'y': 0,
                    'x': 0.7,
                    'len': 0.3,
                    'pad': {'t': 50, 'b': 20},
                    'currentvalue': {'visible': True, 'prefix': 'Max Value: '},
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                    }
                ]
                # Update layout with sliders
                fig.update_layout(
                    sliders=sliders,
                )
                aim_run.track(aim.Figure(fig), name=f"Pareto Front")
            except:
                return None

def execute():
    optuna_config, data_dir, output_dir, seed, save_best_model, group_similar = parse_args(create_parser())
    set_determinism(seed=seed)
    aim_run = aim.Run(experiment='Monai-Optuna MLOps')
    if not os.path.exists(output_dir):
        print(f'Output directory {output_dir} does not exist. Creating it.')
        os.makedirs(output_dir, exist_ok=True)

    # End to end pipeline process
    if save_best_model is not None and save_best_model in ["dice", "training", "validation", "all"]:
        _saving_model = True
    else:
        _saving_model = False
    
    sampler = getattr(optuna.samplers, optuna_config['optuna']['settings']['sampling'])(seed=seed)
    
    aim_run["Model"] = optuna_config['model']['type']
    aim_run["Model_architecture"] = optuna_config['model']['architecture']
    aim_run["Optuna_settings"] = optuna_config['optuna'] 
    
    study = optuna.create_study(study_name="Monai-Optuna MLOps",sampler=sampler,directions=['maximize','minimize','minimize'])
    gen_figures = Gen_Figures(aim_run)
    # implement partial objective function to inject aim_run object prior to optimization
    partial_nokfold_objective_training = partial(nokfold_objective_training, aim_run=aim_run)

    study.optimize(partial_nokfold_objective_training, n_trials=optuna_config['optuna']['settings']['trials'], callbacks=[gen_figures], show_progress_bar=True)
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

    best_trials = {"dice": trial_with_highest_dice.params,
                   "training": trial_with_lowest_training_loss.params,
                   "validation": trial_with_lowest_validation_loss.params
    }

    # Now that the experiment has completed, execute the model saving pipeline. 
    """
    params: {'epochs': 365, 'lr': 1.7918085415441013e-05, 'beta_1': 0.9194803051555897, 'beta_2': 0.8864194132082388, 
    'weight_decay': 1.589695836455197e-07, 'batch': 1, 'optimizer': 'Adam', 'loss_func': 'FocalLoss'}
    """
    if _saving_model:
        if save_best_model == "all":
            for x in ["dice", "training", "validation"]:
                saving_best_trial(id = aim_run.name,
                    sbm = x,
                    epochs = best_trials[x]['epochs'],
                    learning_rate = best_trials[x]['lr'],
                    batch = best_trials[x]['batch'],
                    optimizer = best_trials[x]['optimizer'],
                    beta_1 = best_trials[x]['beta_1'],
                    beta_2 = best_trials[x]['beta_2'],
                    weight_decay = best_trials[x]['weight_decay'],
                    loss_func = best_trials[x]['loss_func']
                )
        else:
            saving_best_trial(id = aim_run.name,
                sbm = save_best_model,
                epochs = best_trials[save_best_model]['epochs'],
                learning_rate = best_trials[save_best_model]['lr'],
                batch = best_trials[save_best_model]['batch'],
                optimizer = best_trials[save_best_model]['optimizer'],
                beta_1 = best_trials[save_best_model]['beta_1'],
                beta_2 = best_trials[save_best_model]['beta_2'],
                weight_decay = best_trials[save_best_model]['weight_decay'],
                loss_func = best_trials[save_best_model]['loss_func']
            )
             
    aim_run.close()
    return None

def saving_best_trial(id, sbm, epochs, learning_rate, batch, optimizer, beta_1, beta_2, weight_decay, loss_func):
    optuna_config, data_dir, output_dir, seed, save_best_model, group_similar = parse_args(create_parser())
    slice_to_track = optuna_config['model']['slice_to_track']
    accel = optuna_config['optuna']['settings']['accelerate']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=seed)

    ## Load hyper params
    n_epochs = epochs
    lr = learning_rate
    b1 = beta_1
    b2 = beta_2
    weight_decay = weight_decay
    batch = batch
    optimizer_type = optimizer
    loss_type = loss_func
    ## Load data
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

    split = optuna_config['optuna']['settings']['split']

    # Split training data into training and validation
    if group_similar:
        custom_dataloader = CustomDataloader(data_dicts, split_ratio=split, batch_size=batch, seed=seed)
        # Flatten the lists (since we grouped them earlier)
        training_set, validation_set = custom_dataloader.split_groups()
    else:
        # Split training data into training and validation
        train_size = int(split * len(data_dicts))
        val_size = len(data_dicts) - train_size
        training_set, validation_set = torch.utils.data.random_split(data_dicts, [train_size, val_size])

    # get transformations
    train_transforms, val_transforms = mtrain.transformer.mtrain_transforms(optuna_config['model']['image_size'], roi_size=optuna_config['model']['validation_roi'])

    # Create training dataloader
    train_ds = CacheDataset(data=training_set, transform=train_transforms, cache_rate=1.0, num_workers=8)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    # Create validation dataloader
    val_ds = CacheDataset(data=validation_set, transform=val_transforms, cache_rate=1.0, num_workers=8)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, num_workers=0)

    ## Generate model, loss function, optimizers    
    model_type = optuna_config['model']['type']
    metric_type = loss_type

    ## MODEL ##
    model = getattr(monai.networks.nets, model_type)(**optuna_config['model']['architecture']).to(device)

    ## EVALUATION METRIC ##
    if metric_type == "FocalLoss":
        loss_function = getattr(monai.losses, metric_type)(to_onehot_y=True, use_softmax=True)
    else:
        loss_function = getattr(monai.losses, metric_type)(to_onehot_y=True, softmax=True)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    ## OPTIMIZATION ##
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr, betas=(b1, b2), weight_decay=weight_decay)

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
    if accel == True:
        scaler = torch.cuda.amp.GradScaler()
        _scaler = torch.cuda.amp.GradScaler()
        print("\nThis instance of Monai-Ops is using the Pytorch Mixed-Precision acceleration. A side-effect \n \
           of this acceleration could be certain combination of hyperparameters can make trials fail. \n"
        )

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
            
            if accel == True:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
            if accel == True:
                with torch.cuda.amp.autocast():
                    outputs = sliding_window_inference(inputs=inputs, roi_size=optuna_config['model']['validation_roi'], sw_batch_size=4, predictor=model)
                    _loss = loss_function(outputs, labels)
                _scaler.scale(_loss).backward()
            else:
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
                    if accel == True:
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_inputs, optuna_config['model']['validation_roi'], sw_batch_size, model)
                            output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()
                    else:
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
                    torch.save(model.state_dict(), os.path.join(output_dir, f"{id}_{sbm}_model.pth"))
                    best_model_log_message = f"saved new best metric model at the {epoch+1}th epoch"
                    print(best_model_log_message)

                message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                message2 = f"\nbest mean dice: {best_metric:.4f} "
                message3 = f"at epoch: {best_metric_epoch}"
                print(message1, message2, message3)

def nokfold_objective_training(trial, aim_run):
    optuna_config, data_dir, output_dir, seed, save_best_model, group_similar = parse_args(create_parser())
    accel = optuna_config['optuna']['settings']['accelerate']
    slice_to_track = optuna_config['model']['slice_to_track']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = trial.suggest_int('epochs', optuna_config['optuna']['hyperparam']['epoch'][0], optuna_config['optuna']['hyperparam']['epoch'][1])
    lr = trial.suggest_float("lr", optuna_config['optuna']['hyperparam']['learning_rate'][0], optuna_config['optuna']['hyperparam']['learning_rate'][1], log=True)
    b1 = trial.suggest_float("beta_1", optuna_config['optuna']['hyperparam']['beta_1'][0], optuna_config['optuna']['hyperparam']['beta_1'][1], log=False)
    b2 = trial.suggest_float("beta_2", optuna_config['optuna']['hyperparam']['beta_2'][0], optuna_config['optuna']['hyperparam']['beta_2'][1], log=False)
    weight_decay = trial.suggest_float("weight_decay", optuna_config['optuna']['hyperparam']['weight_decay'][0], optuna_config['optuna']['hyperparam']['weight_decay'][1], log=False)
    batch = trial.suggest_int("batch", optuna_config['optuna']['hyperparam']['batch'][0], optuna_config['optuna']['hyperparam']['batch'][1], log=False)
    optimizer_type = trial.suggest_categorical("optimizer", optuna_config['optuna']['hyperparam']['optimizer'])
    loss_type = trial.suggest_categorical("loss_func", optuna_config['optuna']['hyperparam']['loss'])
    set_determinism(seed=seed)
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

    if group_similar:
        custom_dataloader = CustomDataloader(data_dicts, split_ratio=split, batch_size=batch, seed=seed)
        # Flatten the lists (since we grouped them earlier)
        training_set, validation_set = custom_dataloader.split_groups()
    else:
        # Split training data into training and validation
        train_size = int(split * len(data_dicts))
        val_size = len(data_dicts) - train_size
        training_set, validation_set = torch.utils.data.random_split(data_dicts, [train_size, val_size])

    # get transformations
    train_transforms, val_transforms = mtrain.transformer.mtrain_transforms(optuna_config['model']['image_size'], roi_size=optuna_config['model']['validation_roi'])

    # Create training dataloader
    train_ds = CacheDataset(data=training_set, transform=train_transforms, cache_rate=1.0, num_workers=8)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    # Create validation dataloader
    val_ds = CacheDataset(data=validation_set, transform=val_transforms, cache_rate=1.0, num_workers=8)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, num_workers=0)

    ## Generate model, loss function, optimizers    
    model_type = optuna_config['model']['type']
    metric_type = loss_type

    ## MODEL ##
    model = getattr(monai.networks.nets, model_type)(**optuna_config['model']['architecture']).to(device)

    ## EVALUATION METRIC ##
    if metric_type == "FocalLoss":
        loss_function = getattr(monai.losses, metric_type)(to_onehot_y=True, use_softmax=True)
    else:
        loss_function = getattr(monai.losses, metric_type)(to_onehot_y=True, softmax=True)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    ## OPTIMIZATION ##
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr, betas=(b1, b2), weight_decay=weight_decay)

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
    if accel == True:
        scaler = torch.cuda.amp.GradScaler()
        _scaler = torch.cuda.amp.GradScaler()
        print("\nThis instance of Monai-Ops is using the Pytorch Mixed-Precision acceleration. A side-effect \n \
           of this acceleration could be certain combination of hyperparameters can make trials fail. \n"
        )

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
            
            if accel == True:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
            if accel == True:
                with torch.cuda.amp.autocast():
                    outputs = sliding_window_inference(inputs=inputs, roi_size=optuna_config['model']['validation_roi'], sw_batch_size=4, predictor=model)
                    _loss = loss_function(outputs, labels)
                _scaler.scale(_loss).backward()
            else:
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
                    if accel == True:
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_inputs, optuna_config['model']['validation_roi'], sw_batch_size, model)
                            output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()
                    else:
                        val_outputs = sliding_window_inference(val_inputs, optuna_config['model']['validation_roi'], sw_batch_size, model)
                        # tracking input, label and output images with Aim
                        output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()

                    aim_run.track(
                            aim.Image(val_inputs[0, 0, :, :, slice_to_track], caption=f"Input Image: {index}"),
                            name="validation",
                            context={"type": "input", 'trial': trial.number},
                        )
                    aim_run.track(
                            aim.Image(val_labels[0, 0, :, :, slice_to_track], caption=f"Label Image: {index}"),
                            name="validation",
                            context={"type": "label", 'trial': trial.number},
                        )
                    aim_run.track(
                            aim.Image(output, caption=f"Predicted Label: {index}"),
                            name="predictions",
                            context={"type": "labels", 'trial': trial.number},
                        )
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
        '-optuna', '--optuna',
        dest='config_file',
        type=str)
    g.add_argument(
        '-data', '--data',
        dest='data_dir',
        type=str)
    g.add_argument(
        '-output', '--output',
        dest='output_dir',
        type=str)
    g.add_argument(
        '-seed', '--seed',
        dest='seed',
        type=int, 
        default=0)
    g.add_argument(
        '-save-best-model', '--save-best-model',
        dest='save_best_model',
        type=str,
        default=None)
    g.add_argument(
        '-group-similar', '--group-similar',
        dest='group_similar',
        action="count", help="group together similar images. The training/validation split will not split grouped images. Images are to be grouped by a shared unique ID")
    g.add_argument(
        '-show-config','--show-config',
        dest='show_config',
        action="count")
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
    if args.save_best_model is not None:
        save_best_model = args.save_best_model
    else:
        save_best_model = None

    if args.show_config:
        print_config()
        exit()
    return [optuna_config, data_dir, output_dir, seed, save_best_model, args.group_similar]


if __name__ == "__main__":
    execute()
