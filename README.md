# Enhancing Glaucoma Detection with Deep Learning

**Automated segmentation of optic nerve features using U-Net and Medical SAM-2**

## Authors
Anjali Goel\
Daniel Luettgen\
Trenton Slocum\
Hildegard Younce 

Sponsor: Dr. Arjun Dirghangi\
Faculty Mentor: Dr. Aiying Zhang 

## Overview

Glaucoma is a major cause of irreversible blindness, and current screening methods are often inaccessible in under-resourced areas. This project introduces deep learning-based segmentation pipelines for identifying optic disc and optic cup features in both high-resolution fundus images and low-quality clinical video frames.

## Objective

- Segment optic nerve head features under glare, blur, and low contrast
- Enable real-time scoring using clinical metrics like DDLS
- Improve accessibility to glaucoma screening tools

## Datasets

- **ORIGA**: 650 high-resolution fundus images with expert annotations
- **UVA Clinical Dataset**: De-identified fundus video frames from high-risk patients using a custom-built ophthalmoscope camera

## Preprocessing

- Images resized to 256×256 (U-Net) or 512×512 (Medical SAM-2)
- Grayscale masks binarized for optic disc (OD) and optic cup (OC)
- Masks resized using nearest-neighbor interpolation

## Models

### U-Net

- Encoder–decoder structure with skip connections
- Trained with binary cross-entropy (BCE) and hybrid (BCE + Dice) loss

### Medical SAM-2

- Adapted from Meta’s Segment Anything Model 2
- Uses bounding box prompts for one-shot segmentation
- Fine-tuned on ORIGA with dual-mask output

## Results

| Metric            | Model       | Training Loss | Test Loss | Dice Coefficient |
|------------------|-------------|---------------|-----------|------------------|
| BCE-only         | U-Net       | 0.0043        | 0.0046    | 0.8745           |
| Hybrid (BCE+Dice)| U-Net       | 0.0621        | 0.0547    | 0.8989           |
| Dual-Mask BCE    | SAM-2       | —             | 0.012     | 0.8219           |

- Hybrid loss improved accuracy for finer structures (optic cup)
- U-Net showed more consistent, anatomically accurate masks than SAM-2
- SAM-2 was more robust in noisy clinical frames with prompt guidance

## Future Work

- Use bounding box prompts with U-Net to reduce glare-related errors
- Explore hybrid pipelines combining U-Net and SAM-2 outputs
- Automate DDLS score computation from segmented masks
- Build a mobile app for on-device inference and glaucoma screening

## Acknowledgments

Thanks to Dr. Arjun Dirghangi and Dr. Aiying Zhang for their support and guidance.

## File Structure

```
.
├── unet
│   ├── GlaucomaDataset.py           # Code for creating a PyTorch DataSet for fundus images
│   ├── unet.py                      # Code for U-Net model architecture
|   ├── unet_segmentation.ipynb      # Python notebook for training and evaluating the U-Net model
|   ├── model_hybrid_state_dict.pth
|   ├── model_refuge_state_dict.pth
│   └── model_state_dict.pth        
└── Medical-SAM2                     # git submodule forked from https://github.com/SuperMedIntel/Medical-SAM2
    └── medsam2-fine-tune.ipynb      # Python notebook for training and evaluating the Medical-SAM2 architecture
```
