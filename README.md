# DistillBLIP

This project implements a knowledge distillation pipeline for the Salesforce BLIP image captioning model. The goal is to create a smaller, more efficient model while maintaining comparable performance to the original.

## Project Structure

```
distillBLIP/
├── data/               # Data loaders and processing
├── models/             # Model architectures
├── training/           # Training scripts and utilities
├── evaluation/         # Evaluation metrics and scripts
├── utils/              # Utility functions
├── configs/            # Configuration files
├── notebooks/          # Example notebooks
└── scripts/            # Helper scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download the pretrained BLIP model:

```bash
python scripts/download_teacher_model.py
```

2. Prepare your dataset:

```bash
python scripts/prepare_dataset.py --dataset coco
```

3. Train the distilled model:

```bash
python training/train.py --config configs/distill_config.yaml
```

4. Evaluate the distilled model:

```bash
python evaluation/evaluate.py --config configs/eval_config.yaml --student_model checkpoints/best_model.pth
```

## Automated Pipeline

For convenience, we provide a script that automates the entire process of downloading the COCO dataset and training the distilled model:

### Option 1: Using the Batch File (Easiest)

1. Simply double-click on `run_distillation.bat` in your project directory.

This batch file will:

- Install required dependencies
- Download the COCO dataset
- Create a smaller subset of COCO (5,000 images) for faster training
- Run the distillation process for 5 epochs
- Evaluate the distilled model

### Option 2: Using the Python Script Directly

For more control over the process, you can run the Python script directly:

```bash
# Basic usage (full dataset, 10 epochs)
python scripts/download_and_train.py

# Using a subset for faster training
python scripts/download_and_train.py --subset --subset_size 5000 --num_epochs 5

# Skip download if you already have COCO
python scripts/download_and_train.py --skip_download --num_epochs 10

# Run evaluation after training
python scripts/download_and_train.py --eval_after_train
```

## Publishing to Hugging Face

To share your distilled model with the community, you can publish it to the Hugging Face Model Hub:

1. Create an account on [Hugging Face](https://huggingface.co/) if you don't have one
2. Install the Hugging Face Hub library: `pip install huggingface_hub`
3. Login to Hugging Face: `huggingface-cli login`
4. Use our provided script to upload your model:

```bash
python scripts/publish_to_hf.py --model_path checkpoints/best_model.pth --model_name your-username/distilled-blip
```

This will convert your PyTorch model to the Hugging Face Transformers format and upload it to the Model Hub.

## Citation

If you use this code for your research, please cite the original BLIP paper, and DLIP paper:

```
@misc{li2022blipbootstrappinglanguageimagepretraining,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      eprint={2201.12086},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2201.12086},
}
@misc{kuang2023dlipdistillinglanguageimagepretraining,
      title={DLIP: Distilling Language-Image Pre-training},
      author={Huafeng Kuang and Jie Wu and Xiawu Zheng and Ming Li and Xuefeng Xiao and Rui Wang and Min Zheng and Rongrong Ji},
      year={2023},
      eprint={2308.12956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.12956},
}
```
