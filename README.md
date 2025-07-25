# Puncta-Tracking

This repository contains a pipeline for tracking the movement of biological condensates and clusters (puncta) using Fiji, ilastik, and Python. It enables segmentation, tracking, and quantitative analysis of time-lapse microscopy data.

## **Features:**

Segmentation via ilastik pixel classification

ROI extraction and tracking using Fiji (TrackMate)

Post-processing and track analysis in Python

Customizable for multi-channel data

Outputs include trajectories, intensity profiles, and kymographs

## **Dependencies:**

Python 3.9

numpy, pandas, matplotlib, scikit-image, tifffile, trackpy

Fiji (ImageJ) with TrackMate plugin

ilastik (>=1.4)

## **Usage:**

Run segmentation in ilastik and export probability maps.

Load results in Fiji and run the provided macro for tracking.

Use Python scripts to analyze and visualize track data.
