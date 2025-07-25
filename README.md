# Puncta-Tracking

This repository contains a pipeline for tracking the movement of biological condensates and clusters (puncta) using Fiji, ilastik, and Python. It enables segmentation, tracking, and quantitative analysis of time-lapse microscopy data. The pipeline employs the Mass-Balance Imaging (MBI) technique employed in Yan et al., 2022, used to study the compositional dynamics associated with multichannel movies obtained from fluorescent microscopy of condensates with multiple compartments.

## **Features:**

Segmentation via ilastik pixel classification

ROI extraction and tracking using Fiji (TrackMate)

Post-processing and track analysis in Python

Customizable for multi-channel data

Outputs include trajectories, intensity profiles, and kymographs

## Getting Started

### Dependencies

The package requires both Fiji (or atleast ImageJ with trackmate plugin) and ilastik be installed. The paths to these programs can be specified at runtime. The package has been tested with a fresh install of Fiji version 2.15.1 (trackmate version 7.13.2) and ilastik version 1.4.0.post1 .

This package requires python version >= 3.10 . Following packages are required for this packages:

* lxml
* pyyaml
* numpy
* scipy
* pandas
* networkx
* scikit-image
* nd2
* tifffile
* xarray

Following are only required for running the example jupyter notebook:

* jupyter
* matplotlib
* ffmpeg

### Installing

Installing the package in its own separate environment is highly recommended. We recommend using [conda](https://docs.conda.io/en/latest/#) for the same. The following steps will assume conda is used for creating a new environment.

To install the package, follow these steps:

1. If not already installed, install Fiji and ilastik using the recommended methods listed in their documentations.
2. Either clone the repository or download a zip archive of the repository (click on green "Code" button, select "Download ZIP").
3. Open Terminal (macOS and Linux) or Anaconda cmd/powershell (Windows). Create a new environment and activate it.

```console
conda create -n puncta-tracking python=3.10
conda activate puncta-tracking
```

4. \[Optional\] You may also install additional packages required for running the example notebook:

```console
conda install matplotlib jupyter ffmpeg
```

5. If repository was cloned, navigate to where the cloned repo is located on your system. Use pip to install the package:

```console
cd \path\to\your\cloned\repo
pip install .
```

If instead a ZIP archive was installed, navigate to where the zip archive is located on your system. Use pip to install the package:

```console
pip install \path\to\downloaded\archive
```

### How to use the package

See [example.ipynb](./example/example.ipynb).
