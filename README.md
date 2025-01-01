# CVIS Course Assignment - Visual Place Recognition and Localisation

## Project Overview  
This repository contains the implementation and results of the assignment for the **CVIS (Computer Vision for Intelligent Systems)** course, as part of the MRGCV master's program. The selected use case for this project is **buildings**, focusing on visual place recognition and localisation.

Visual place recognition involves determining whether two images capture the same 3D scene, even if taken under different conditions, such as varying illumination or perspectives. Visual localisation focuses on estimating the camera pose for a given image, enabling a reconstruction of the photographer's position at the time the photo was taken.  

The project includes the analysis of widely separated images in time, specifically leveraging a historical image of a building and new photographs of the same scene. 

## Objectives  
1. **Camera Localisation**: Precisely localise the camera that took the historical photo and describe the world frame used for this localisation.  
2. **Change Detection**: Identify and delineate regions in the historical image that correspond to modified or unchanged areas in the scene.  

## Methodology  

### Image Acquisition  
- **Historical Image**: Select an image taken before the year 2000.  
- **New Images**: Capture 3–10 new images of the same scene with a mobile phone, including at least one taken from a similar viewpoint to the historical photo.  

### Camera Calibration  
- **New Camera**: Perform camera calibration using images of a planar pattern.  
- **Historical Camera**: Estimate the calibration parameters from the pose of the historical camera.  

### Software Tools  
- **Self-Developed SfM Libraries**: Utilised for computing camera poses.  
- **COLMAP**: Used as a reference implementation for structure-from-motion (SfM) tasks, providing comparative insights.  

### Feature Matching  
Test various point detection, description, and matching methods, including the lightglue matcher for enhanced results.  

### Change Detection  
Exploit geometrical relationships between the historical photo and new images to identify changes, leveraging SfM software or COLMAP.  

## Project Structure

```
cv_assignment/
│
├── data/
│   ├── raw/               # Original images (old and new) organized by type
│   │   ├── buildings/
│   │   └── colonoscopy/
│   ├── processed/         # Preprocessed images (after calibration, feature detection, etc.)
│   └── calibration/       # Calibration patterns and parameters
│
├── scripts/               # Main Python scripts
│   ├── localisation.py    # Precise camera localisation
│   ├── change_detection.py # Detecting changes in image regions
│   ├── sfm_library/       # Structure-from-Motion utilities
│   │   ├── __init__.py
│   │   ├── feature_detection.py
│   │   ├── camera_pose_estimation.py
│   │   └── utils.py       # Common utility functions
│   └── calibration.py     # Camera calibration
│
├── results/               # Results of processing
│   ├── localisation/      # Camera pose estimation results
│   ├── change_detection/  # Results of change detection
│   └── metrics/           # Performance metrics such as RMSE, execution time, etc.
│
├── slides/                # Presentation and other deliverables
│   └── slides.pdf
│
├── docs/                  # Documentation
│   └── report.md          # Summary of methodologies and results
│
├── requirements.txt       # Project dependencies
├── README.md              # Project explanation (this file)
└── .gitignore             # Files/folders to exclude from the repository
```

---

## Project Workflow

### 1. Camera Calibration
The first step is to calibrate the camera using calibration patterns and generate the intrinsic parameters. This allows for rectification and distortion correction in subsequent steps.

Run the calibration script:
```bash
python scripts/calibration.py
```
This will save the calibration parameters in `data/calibration/` for reuse.

---

### 2. Camera Localisation
Estimate the precise localisation of the camera that took the old photo by detecting and matching features between the old and new images. Use these correspondences to estimate the camera pose in the specified world coordinate system.

Run the localisation script:
```bash
python scripts/localisation.py --old_photo path/to/old.jpg --new_photo path/to/new.jpg
```
The output will be saved in `results/localisation/`.

---

### 3. Change Detection
Identify regions of the old image that correspond to unchanged or modified areas in the scene. This involves comparing regions in the old and new photos and generating a binary mask for changed areas.

Run the change detection script:
```bash
python scripts/change_detection.py --old_photo path/to/old.jpg --new_photo path/to/new.jpg
```
The generated masks will be saved in `results/change_detection/`.

---

## Technical Details

### `localisation.py`
Handles the following:
- Loads old and new images.
- Detects features and matches them between images (e.g., using SIFT or SuperGlue).
- Estimates the camera pose using RANSAC and PnP algorithms.
- Outputs the localisation results.

### `change_detection.py`
Performs:
- Feature matching and region comparison between old and new images.
- Identifies modified and unchanged areas using similarity metrics or other techniques.
- Generates visual outputs and binary masks.

### `sfm_library/`
Contains utilities for structure-from-motion:
- **`feature_detection.py`**: Detects keypoints, extracts descriptors, and matches features.
- **`camera_pose_estimation.py`**: Estimates camera poses based on matched features.
- **`utils.py`**: Provides auxiliary functions such as image loading, preprocessing, and transformation.

### `calibration.py`
Performs:
- Camera calibration using chessboard patterns or other calibration targets.
- Saves intrinsic parameters for use in localisation and change detection tasks.

---

## Dependencies

Install the required packages by running:
```bash
pip install -r requirements.txt

## Deliverables  
1. **Slides**: Present the methodology, results, and insights in a 10-slide presentation.  
2. **Oral Presentation**: Deliver a 15-minute talk summarising the findings.  
3. **Experimental Results**: Submit quantitative information, including RMSE errors, computational times, and the number of matches.  

## Assessment Criteria  
1. **Slides & Oral Talk**: 10%  
2. **Quantitative Analysis**: 10%  
3. **Camera Localisation**: 50%  
   - Methodology: 10%  
   - Experimental Performance: 15%  
   - Comparison with COLMAP: 15%  
   - Utilising 10+ New Images with COLMAP: 10%  
4. **Change Detection**: 30%  
   - Methodology: 15%  
   - Experimental Performance: 15%  

## Software Requirements  
- Python (for scripts such as calibration and custom feature matching).  
- COLMAP (https://colmap.github.io/).  
- SfM libraries developed during course labs.  

## Authors  
- **Juan Lorente** (Project Contributor)  
- Based on guidelines by Javier Morlana, Jesús Bermúdez, Tomás Berriel, and J.M.M Montiel.  