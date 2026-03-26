# CORA


## Overview

CORA is pretrained by detecting synthetically generated coronary lesions inserted into real CCTA volumes. This biases the encoder toward vascular pathology rather than dominant background anatomy. The framework consists of three components:

- **Synthesis Engine** (`dataset.py`): Generates realistic calcified and non-calcified plaques with controlled HU distributions and irregular morphologies within coronary artery regions.
- **Multi-Window Input**: Converts raw HU volumes into 4-channel inputs (fat, soft tissue, angiographic, calcification windows) to capture complementary attenuation characteristics.
- **3D Residual U-Net** (`model.py`): Encoder-decoder architecture pretrained on abnormality segmentation, with downstream task heads for classification and multimodal MACE prediction.

## Repository Structure

```
├── models/model.py        # Model architectures (pretraining + downstream)
├── dataset.py      # Pretraining dataset with lesion synthesis engine
├── pretrain_CORA.py     # Pretraining script
└── README.md
```

## Requirements

```
torch >= 2.0
numpy
scipy
pandas
scikit-learn
tqdm
matplotlib
transformers
dynamic-network-architectures
```

Install dependencies:
```bash
pip install torch numpy scipy pandas scikit-learn tqdm matplotlib transformers dynamic-network-architectures
```

## Data Preparation

Each CCTA volume should be preprocessed into an NPZ file containing:
- `CTA_HU`: Raw CCTA volume in Hounsfield Units (D × H × W).
- `CA`: Binary coronary artery mask (D × H × W), extracted using a pretrained segmentation model (e.g., nnU-Net trained on ImageCAS).

All volumes should be resampled to 0.5 × 0.5 × 0.5 mm³ voxel spacing.

Expected directory structure:
```
npz_root/
├── patient_001/
│   └── CTA_patient_001.npz
├── patient_002/
│   └── CTA_patient_002.npz
└── ...
```

A patient index file (Excel) listing all patient identifiers is required.

## Pretraining

1. Update paths in `pretrain.py`:
   ```python
   EXCEL_FILE = "data/CTA_all_list.xlsx"
   NPZ_ROOT = "/path/to/preprocessed/npz"
   CHECKPOINT_DIR = "checkpoints/cora_pretrain"
   ```

2. Run pretraining:
   ```bash
   python pretrain.py
   ```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 50 | Number of pretraining epochs |
| `BATCH_SIZE` | 56 | Batch size |
| `LEARNING_RATE` | 1e-4 | Peak learning rate (AdamW) |
| `PATCH_SHAPE` | (96, 96, 96) | Patch size for artery-centric sampling |
| `WARMUP_EPOCHS` | 3 | Linear warmup epochs |
| Tversky β | 0.9 | FN penalty weight (prioritizes recall) |
| Focal γ | 4.0 | Focusing parameter for hard examples |

### Loss Function

The pretraining objective combines:
- **Tversky Loss** (α=0.1, β=0.9): Prioritizes recall for sparse lesion detection.
- **Focal Loss** (γ=4.0): Down-weights easy background voxels.

## Downstream Usage

After pretraining, the encoder weights can be loaded for downstream tasks:

```python
from model import CORAClassifier

# Initialize classifier
classifier = CORAClassifier(num_input_channels=4)

# Load pretrained encoder weights
pretrained = torch.load("checkpoints/cora_pretrain/cora_pretrained.pth")
encoder_weights = {k.replace("unet.encoder.", ""): v 
                   for k, v in pretrained.items() 
                   if k.startswith("unet.encoder.")}
classifier.encoder.load_state_dict(encoder_weights)
```

Supported downstream tasks:
- **Plaque characterization**: Volume-level multi-label classification (calcified / non-calcified).
- **Stenosis detection**: Lesion-level segmentation using full encoder-decoder.
- **Coronary artery segmentation**: Dense 3D segmentation with Dice loss fine-tuning.
- **MACE risk stratification**: Multimodal fusion with clinical text via `CORAMultimodalMACE`.

