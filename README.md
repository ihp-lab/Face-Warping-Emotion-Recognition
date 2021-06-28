# Face-Warping-Emotion-Recognition
Source codes for our ACII'21 paper: Contrastive Learning for Domain Transfer in Cross-Corpus Emotion Recognition.

# Getting Started
## Installation
- This code is tested with PyTorch 1.2.0 and Python 3.7.6

## Datasets
IDs for face warping with Aff-Wild2 and SEWA are provided in Aff-Wild2-train-id.txt, Aff-Wild2-val-id.txt, and SEWA-id.txt.

Aff-Wild2 and SEWA. All the frames should be cropped and aligned.

Put these datasets into the folder "data". The directory structure should be modified to match:
```
├── code
|	├── ..
├── contrastive
|	├── ..
├── mini_datasets
|	├── Aff-Wild2
|	|	├── train.csv
|	|	├── val.csv
|	|	├── test.csv
|	├── SEWA
|	|	├── train.csv
|	|	├── val.csv
|	|	├── test.csv
├── data
|	├── Aff-Wild2
|	|	├── cropped_aligned
|	|	|	├── ..
|	├── SEWA
|	|	├── porep_SEWA
|	|	|	├── ..
|	├── Real
|	|	├── Aff-Wild2_v3
|	|	|	├── ..
|	├── Fake
|	|	├── Aff-Wild2_v3
|	|	|	├── ..
```

## Training
Train base model with Aff-Wild2/SEWA:
```
python train_bl.py --source Aff-Wild2/SEWA --label arousal/valence
```

Train domain adaptation models with Aff-Wild2 and SEWA:
```
python train_dann.py/train_dan.py/train_adda.py --source Aff-Wild2 --target SEWA --label arousal/valence
```

Pre-train FATE with Aff-Wild2 and SEWA (in `/contrastive` folder):
```
python pretrain.py
```

Fine-tune FATE with SEWA (in `/code` folder):
```
python fine-tune.py --label arousal/valence
```

## Test
Test on SEWA (base model and domain adaptation baselines):
```
python test_models.py --source Aff-Wild2 --target SEWA --model bl/dann/dan/adda
```
Test on SEWA (FATE):
```
python test_FATE.py --source Aff-Wild2 --target SEWA
```

Please put `/checkpoints` folder in `/code` folder and `/checkpoints_v3` folder in `/contrastive` folder when loading weights. All the weights are available on [Google Drive](https://drive.google.com/drive/folders/1RGfjAVR3tbycXtC-vYBklg_qwyr_V616?usp=sharing).

In `/checkpoints` folder, `/model` folder saves the weights for FATE.
