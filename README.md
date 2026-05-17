# Brain Tumor MRI Classification — PyTorch CNN

## Problem Description
Multi-class classification of brain MRI scans into 4 categories:
glioma, meningioma, no tumor, pituitary — using a custom CNN in PyTorch (nn.Module).

## Dataset
[Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- 4 classes · ~7,000 images
- Split: 70% train / 15% val / 15% test

## Results

| Model   | Optimizer | Batch | LR    | Test Accuracy | Test Loss |
|---------|-----------|-------|-------|---------------|-----------|
| Model A | Adam      | 32    | 0.001 | 85.24%        | 0.3965    |
| Model B | SGD       | 64    | 0.01  | 84.13%        | 0.4225    |

## Architecture Highlights
- 3 × (Conv2d → BatchNorm → ReLU → MaxPool + Dropout2d) blocks
- AdaptiveAvgPool → Flatten → Linear(512) → Dropout → Linear(128) → Linear(4)
- Enhancements: BatchNorm, Dropout, Data Augmentation

## How to Run
1. Open `brain_tumor_cnn.ipynb` in Google Colab
2. Run Cell 1 to download the dataset via Kaggle API
3. Run all cells in order
4. Results saved to `/results/`

## Requirements
torch · torchvision · sklearn · matplotlib · seaborn
