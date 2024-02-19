<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">MONAI-TRAIN</h1>
</p>
<p align="center">
    <em><code>► MONAI & Pytorch Training Pipeline with AimStack</code></em>
</p>

![GitHub Issues](https://img.shields.io/github/issues/adnan-umich/monai-train.svg) [![Python application](https://github.com/adnan-umich/monai-train/actions/workflows/python-app.yml/badge.svg)](https://github.com/adnan-umich/monai-train/actions/workflows/python-app.yml)
<img width="1478" alt="Screenshot 2024-02-07 at 11 37 39 PM" src="https://github.com/adnan-umich/monai-train/assets/124732717/6eefafb1-3af7-4e2c-a871-336097cd7b4f">

<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=flat&logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/MONAI-weekly-blue.svg?logo=MONAI-WEEKLY" alt="MONAI-Weekly">
	<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat&logo=YAML&logoColor=white" alt="YAML">
	<img src="https://img.shields.io/badge/Poetry-60A5FA.svg?style=flat&logo=Poetry&logoColor=white" alt="Poetry">
	<br>
	<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=flat&logo=Plotly&logoColor=white" alt="Plotly">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=flat&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Uvicorn-009688.svg?style=flat&logo=uvicorn&logoColor=white" alt="Uvicorn">
    <img src="https://img.shields.io/badge/CUDA-009688.svg?style=flat&logo=cuda&logoColor=white" alt="CUDA">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [Running monai-train](#-running-monai-train)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>► MONAI is an open-source framework based on PyTorch for deep learning in healthcare imaging. MONAI-Train is a pipeline developed on top of the MONAI ecosystem to facilitate the training and validation of deep learning models for researchers.</code>

---

##  Features

<code>► Image TRANSFORMATIONS (random crop, intensity normalization, pixel density, pixel dimensioning). K-fold cross validation. Experiment tracking with AimStack. YAML-file based model architecture configurations. </code>

---

##  Repository Structure

```sh
└── monai-train/
    ├── .github
    │   └── workflows
    │       └── python-app.yml
    ├── README.md
    ├── example
    │   ├── model_unet.yaml
    │   ├── model_unetr.yaml
    │   └── transformer.yaml
    ├── monai-train
    │   ├── __init__.py
    │   ├── __main__.py
    │   └── transformer.py
    ├── poetry.lock
    ├── pyproject.toml
    └── requirements.txt
```

---

<details closed><summary>monai-train</summary>

| File                                                                                                | Summary                         |
| ---                                                                                                 | ---                             |
| [__main__.py](https://github.com/adnan-umich/monai-train/blob/master/monai-train/__main__.py)       | <code>► main entrypoint</code> |
| [transformer.py](https://github.com/adnan-umich/monai-train/blob/master/monai-train/transformer.py) | <code>► image preprocessing</code> |

</details>

<details closed><summary>example</summary>

| File                                                                                                | Summary                         |
| ---                                                                                                 | ---                             |
| [model_unetr.yaml](https://github.com/adnan-umich/monai-train/blob/master/example/model_unetr.yaml) | <code>► example UNETr model</code> |
| [model_unet.yaml](https://github.com/adnan-umich/monai-train/blob/master/example/model_unet.yaml)   | <code>► example UNet model</code> |
| [transformer.yaml](https://github.com/adnan-umich/monai-train/blob/master/example/transformer.yaml) | <code>► example image processings</code> |

</details>

<details closed><summary>.github.workflows</summary>

| File                                                                                                      | Summary                         |
| ---                                                                                                       | ---                             |
| [python-app.yml](https://github.com/adnan-umich/monai-train/blob/master/.github/workflows/python-app.yml) | <code>► build tests</code> |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version >= 3.9, < 3.12`
* **CUDA**; `version 12.x.y`
* **Poetry**;

###  Installation

1. Clone the monai-train repository:

```sh
git clone https://github.com/adnan-umich/monai-train
```

2. Change to the project directory:

```sh
cd monai-train
```

3. Install the dependencies:

```sh
> poetry install .
```

###  Running `monai-train`

Use the following command to run monai-train:

```sh
> module load python cuda poetry
> poetry shell
> (monai-train-py3.11) python -m monai-train --model   = ./example/model_unet.yaml \
                                             --data    = /path/to/NIfTI/dataset \
                                             --output  = /output \
                                             --split   = 0.8 \
                                             --lr      = 0.0001 \
                                             --epochs  = 20 \
                                             --batch   = 5 \
                                             --seed    = 12345 \
					     --kfold   = 5 \
					     --savemodel = False
```

###  Configurationd

```
--model : CNN model yaml file. 
--data: Location of the NIfTI dataset. It must contain a folder named imagesTr,
        which holds the training images that have been labeled. Additionally,
        there should be a folder named labelsTr containing the corresponding labels for the training sets.
        Furthermore, there should be a folder named imagesTs that contains the test images, which are unlabeled.
--output: Location to save output data, such as trained model, and weights.
--split : percentage of training set, remainder is validation set. Default (80/20).
--lr : learning rate
--epochs : maximum number of epochs
--batch : maximum batch size for the training set. Validation set to 1.
--transfer (optional) : full path of pretrained model pickle file (.pth) for transfer learning.
--seed : Sets the seed for generating random numbers. Default value 0. Can control reproducibility. 
--kfold : (default 0, no cross validation. > 0 will trigger kfold cross validation based training.)
--savemodel : (Default False), used with kfold cross validation. Indicates whether to run the final training with 100% training data for model saving. 
```

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/adnan-umich/monai-train/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/adnan-umich/monai-train/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/adnan-umich/monai-train/issues)**: Submit bugs found or log feature requests for the `monai-train` project.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/adnan-umich/monai-train
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-quick-links)

---
