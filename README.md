# PENGWIN Segmentation Challenge - MICCAI 2024

This repository contains our work for the MICCAI 2024 PENGWIN challenge. The challenge aimed to advance automated pelvic fracture segmentation techniques in both 3D CT scans and 2D X-ray images.

## Task 1: 3D CT Pelvic Fragment Segmentation
We used 150 CT scans from diverse patient cohorts for pelvic fragment segmentation, including sacrum and hipbone fragments. The ground truth was annotated and validated by medical experts.

## Task 2: 2D X-ray Pelvic Fragment Segmentation
For this task, we generated realistic X-ray images and 2D labels from the CT data using DeepDRR. A variety of virtual C-arm camera positions and surgical tools were incorporated for realism.

## Dataset
The dataset includes:
- **CT Scans:** 150 patients, multiple institutions, different scanning equipment.
- **X-ray Images:** Generated from CT using DeepDRR, with realistic 2D labels.

## Results
- Task 1: Achieved an average Dice score of X on pelvic fragment segmentation.
- Task 2: Achieved an average Dice score of Y on X-ray segmentation.

## How to Use

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PENGWIN_Segmentation_Challenge.git
cd PENGWIN_Segmentation_Challenge
