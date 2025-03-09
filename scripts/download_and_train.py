#!/usr/bin/env python
"""
Script to download COCO dataset and run BLIP distillation.
"""

import os
import sys
import argparse
import subprocess
import zipfile
import time
import shutil
import requests
from tqdm import tqdm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Download COCO dataset and run BLIP distillation")
    
    # Download options
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store datasets")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading COCO if already downloaded")
    
    # Training options
    parser.add_argument("--config", type=str, default="configs/distill_config.yaml", help="Path to training config")
    parser.add_argument("--teacher_model", type=str, default="Salesforce/blip-image-captioning-large", 
                        help="Teacher model name or path")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save training logs")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_after_train", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval_config", type=str, default="configs/eval_config.yaml", 
                        help="Path to evaluation config")
    parser.add_argument("--subset", action="store_true", help="Use a subset of COCO for faster training")
    parser.add_argument("--subset_size", type=int, default=5000, help="Number of images to use in subset")
    
    return parser.parse_args()


def download_file(url, filename, chunk_size=1024):
    """
    Download a file with progress bar and resume capability.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        chunk_size: Size of chunks to download
        
    Returns:
        Path to the downloaded file
    """
    # Check if file exists and get its size
    file_size = 0
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"File {os.path.basename(filename)} exists, size: {file_size / (1024*1024):.2f} MB")
    
    # Set up headers for resuming download
    headers = {}
    if file_size > 0:
        headers['Range'] = f'bytes={file_size}-'
        print(f"Resuming download from byte {file_size}")
    
    # Make request with headers
    try:
        response = requests.get(url, stream=True, headers=headers)
        
        # Check if the server supports resuming
        if file_size > 0 and response.status_code == 200:
            # Server doesn't support resume, start from beginning
            print("Server doesn't support resuming downloads. Starting from beginning.")
            file_size = 0
            response = requests.get(url, stream=True)
        
        # Get total size
        total_size = int(response.headers.get('content-length', 0))
        
        # If resuming, add the existing file size to get the total
        if file_size > 0 and response.status_code == 206:  # Partial Content
            total_size += file_size
            print(f"Total file size: {total_size / (1024*1024):.2f} MB")
        
        # Open file in append mode if resuming, otherwise write mode
        mode = 'ab' if file_size > 0 and response.status_code == 206 else 'wb'
        
        with open(filename, mode) as f, tqdm(
                desc=os.path.basename(filename),
                initial=file_size,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                bar.update(size)
                
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(filename) and file_size > 0:
            print(f"You can try resuming the download later.")
        raise


def extract_zip(zip_path, extract_path):
    """Extract a zip file with progress."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        print(f"Extracting {zip_path} to {extract_path} ({total_files} files)")
        
        for i, file in enumerate(zip_ref.infolist()):
            if i % 100 == 0 or i == total_files - 1:
                print(f"Extracting file {i+1}/{total_files}", end='\r')
            zip_ref.extract(file, extract_path)
    
    print(f"\nExtraction of {zip_path} completed.")
    

def download_coco(data_dir, skip_if_exists=False):
    """Download COCO 2017 dataset using DatasetNinja for faster downloads."""
    coco_dir = os.path.join(data_dir, "coco")
    os.makedirs(coco_dir, exist_ok=True)
    
    # Create subdirectories
    train_dir = os.path.join(coco_dir, "train2017")
    val_dir = os.path.join(coco_dir, "val2017")
    annotations_dir = os.path.join(coco_dir, "annotations")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Check if already downloaded
    train_files = os.listdir(train_dir) if os.path.exists(train_dir) else []
    val_files = os.listdir(val_dir) if os.path.exists(val_dir) else []
    annotation_files = os.listdir(annotations_dir) if os.path.exists(annotations_dir) else []
    
    if skip_if_exists and len(train_files) > 0 and len(val_files) > 0 and len(annotation_files) > 0:
        print("COCO dataset seems to be already downloaded and extracted. Skipping download.")
        return
    
    # DatasetNinja download URLs (faster than the official ones)
    urls = {
        "train_images": "https://datasetninja.com/get-download-link?id=5f0cbcd7-f497-4578-a627-a2fbc5c0a153",
        "val_images": "https://datasetninja.com/get-download-link?id=8c5e1f2c-5cbc-4f41-8c64-1a8ed0293971",
        "annotations": "https://datasetninja.com/get-download-link?id=7fcb7fc7-3b94-4a06-adb2-731d1ad3c067"
    }
    
    downloads_dir = os.path.join(coco_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Download files
    download_paths = {}
    for name, url in urls.items():
        print(f"Getting download link for {name} from DatasetNinja...")
        # Get the actual download link from DatasetNinja
        try:
            response = requests.get(url)
            if response.status_code == 200:
                download_url = response.json().get("url")
                if download_url:
                    print(f"Downloading {name} from {download_url}")
                    filename = os.path.join(downloads_dir, os.path.basename(download_url.split("?")[0]))
                    download_paths[name] = download_file(download_url, filename)
                else:
                    print(f"Failed to get download URL for {name}. Falling back to official source.")
                    # Fallback to official URLs
                    fallback_urls = {
                        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
                        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                    }
                    fallback_url = fallback_urls.get(name)
                    print(f"Downloading {name} from {fallback_url}")
                    filename = os.path.join(downloads_dir, os.path.basename(fallback_url))
                    download_paths[name] = download_file(fallback_url, filename)
            else:
                print(f"Failed to get download link for {name}. Falling back to official source.")
                # Fallback to official URLs
                fallback_urls = {
                    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
                    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                }
                fallback_url = fallback_urls.get(name)
                print(f"Downloading {name} from {fallback_url}")
                filename = os.path.join(downloads_dir, os.path.basename(fallback_url))
                download_paths[name] = download_file(fallback_url, filename)
        except Exception as e:
            print(f"Error getting download link: {e}. Falling back to official source.")
            # Fallback to official URLs
            fallback_urls = {
                "train_images": "http://images.cocodataset.org/zips/train2017.zip",
                "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            }
            fallback_url = fallback_urls.get(name)
            print(f"Downloading {name} from {fallback_url}")
            filename = os.path.join(downloads_dir, os.path.basename(fallback_url))
            download_paths[name] = download_file(fallback_url, filename)
    
    # Extract files
    for name, path in download_paths.items():
        print(f"Extracting {name} from {path}")
        extract_zip(path, coco_dir)
    
    print("COCO dataset download and extraction completed.")


def create_subset(coco_dir, subset_size):
    """Create a subset of COCO for faster training."""
    import json
    import random
    import shutil
    
    print(f"Creating a subset of COCO with {subset_size} images...")
    
    # Create subset directories
    subset_dir = os.path.join(coco_dir, "subset")
    subset_train_dir = os.path.join(subset_dir, "train2017")
    subset_val_dir = os.path.join(subset_dir, "val2017")
    subset_annotations_dir = os.path.join(subset_dir, "annotations")
    
    os.makedirs(subset_dir, exist_ok=True)
    os.makedirs(subset_train_dir, exist_ok=True)
    os.makedirs(subset_val_dir, exist_ok=True)
    os.makedirs(subset_annotations_dir, exist_ok=True)
    
    # Load train annotations
    train_annotations_path = os.path.join(coco_dir, "annotations", "captions_train2017.json")
    with open(train_annotations_path, 'r') as f:
        train_data = json.load(f)
    
    # Get unique image IDs
    train_image_ids = list(set(ann['image_id'] for ann in train_data['annotations']))
    
    # Randomly sample subset_size images
    train_subset_size = int(subset_size * 0.8)  # 80% for training
    val_subset_size = subset_size - train_subset_size  # 20% for validation
    
    random.shuffle(train_image_ids)
    train_subset_ids = set(train_image_ids[:train_subset_size])
    
    # Load validation annotations
    val_annotations_path = os.path.join(coco_dir, "annotations", "captions_val2017.json")
    with open(val_annotations_path, 'r') as f:
        val_data = json.load(f)
    
    # Get unique validation image IDs
    val_image_ids = list(set(ann['image_id'] for ann in val_data['annotations']))
    random.shuffle(val_image_ids)
    val_subset_ids = set(val_image_ids[:val_subset_size])
    
    # Create subset annotations
    train_subset_annotations = {
        'info': train_data['info'],
        'licenses': train_data['licenses'],
        'images': [img for img in train_data['images'] if img['id'] in train_subset_ids],
        'annotations': [ann for ann in train_data['annotations'] if ann['image_id'] in train_subset_ids]
    }
    
    val_subset_annotations = {
        'info': val_data['info'],
        'licenses': val_data['licenses'],
        'images': [img for img in val_data['images'] if img['id'] in val_subset_ids],
        'annotations': [ann for ann in val_data['annotations'] if ann['image_id'] in val_subset_ids]
    }
    
    # Save subset annotations
    train_subset_path = os.path.join(subset_annotations_dir, "captions_train2017.json")
    with open(train_subset_path, 'w') as f:
        json.dump(train_subset_annotations, f)
    
    val_subset_path = os.path.join(subset_annotations_dir, "captions_val2017.json")
    with open(val_subset_path, 'w') as f:
        json.dump(val_subset_annotations, f)
    
    # Copy subset images
    print("Copying train subset images...")
    for img in tqdm(train_subset_annotations['images']):
        filename = img['file_name']
        src_path = os.path.join(coco_dir, "train2017", filename)
        dst_path = os.path.join(subset_train_dir, filename)
        shutil.copy(src_path, dst_path)
    
    print("Copying validation subset images...")
    for img in tqdm(val_subset_annotations['images']):
        filename = img['file_name']
        src_path = os.path.join(coco_dir, "val2017", filename)
        dst_path = os.path.join(subset_val_dir, filename)
        shutil.copy(src_path, dst_path)
    
    print(f"COCO subset created with {len(train_subset_annotations['images'])} train and {len(val_subset_annotations['images'])} validation images.")
    
    # Update config
    return {
        'train_image_dir': subset_train_dir,
        'train_annotations': train_subset_path,
        'val_image_dir': subset_val_dir,
        'val_annotations': val_subset_path
    }


def update_config_for_subset(config_path, subset_paths):
    """Update the configuration file to use the subset paths."""
    import yaml
    
    print(f"Updating config file {config_path} to use COCO subset...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data paths
    if 'data' not in config:
        config['data'] = {}
    
    config['data']['train_image_dir'] = subset_paths['train_image_dir']
    config['data']['train_annotations'] = subset_paths['train_annotations']
    config['data']['val_image_dir'] = subset_paths['val_image_dir']
    config['data']['val_annotations'] = subset_paths['val_annotations']
    
    # Save the updated config
    updated_config_path = config_path.replace('.yaml', '_subset.yaml')
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Updated config saved to {updated_config_path}")
    return updated_config_path


def verify_dependencies():
    """Verify that all required dependencies are installed."""
    try:
        import torch
        import torchvision
        import transformers
        import yaml
        import tqdm
        import PIL
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("All dependencies appear to be installed correctly.")
        
        # Check for CUDA
        if torch.cuda.is_available():
            print(f"CUDA is available. Detected {torch.cuda.device_count()} device(s).")
            print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available. Training will proceed on CPU, which may be slow.")
        
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required dependencies with: pip install -r requirements.txt")
        return False


def run_training(args, config_path):
    """Run the BLIP distillation training."""
    print("\n" + "="*80)
    print("Starting BLIP distillation training...")
    print("="*80)
    
    # Verify that the training script exists
    train_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "train.py")
    if not os.path.exists(train_script):
        print(f"Training script not found at {train_script}")
        return False
    
    # Prepare command
    cmd = [
        sys.executable, train_script,
        "--config", config_path,
        "--teacher_model", args.teacher_model,
        "--output_dir", args.output_dir,
        "--log_dir", args.log_dir,
        "--num_epochs", str(args.num_epochs)
    ]
    
    # Run the training process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    return_code = process.wait()
    
    if return_code == 0:
        print("\nTraining completed successfully!")
        return True
    else:
        print(f"\nTraining failed with return code {return_code}")
        return False


def run_evaluation(args, best_model_path=None):
    """Run evaluation on the distilled model."""
    print("\n" + "="*80)
    print("Starting evaluation of distilled model...")
    print("="*80)
    
    # Find the best model if not provided
    if best_model_path is None:
        if os.path.exists(os.path.join(args.output_dir, "best_model.pth")):
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
        else:
            # Find the latest checkpoint
            checkpoints = [f for f in os.listdir(args.output_dir) if f.endswith(".pth")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(args.output_dir, x)))
                best_model_path = os.path.join(args.output_dir, latest_checkpoint)
            else:
                print("No model checkpoints found for evaluation.")
                return False
    
    print(f"Using model checkpoint: {best_model_path}")
    
    # Verify that the evaluation script exists
    eval_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "evaluate.py")
    if not os.path.exists(eval_script):
        print(f"Evaluation script not found at {eval_script}")
        return False
    
    # Prepare command
    results_dir = os.path.join("evaluation_results", f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    
    cmd = [
        sys.executable, eval_script,
        "--config", args.eval_config,
        "--teacher_model", args.teacher_model,
        "--student_model", best_model_path,
        "--output_dir", results_dir,
        "--compare_with_teacher"
    ]
    
    # Run the evaluation process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    return_code = process.wait()
    
    if return_code == 0:
        print(f"\nEvaluation completed successfully! Results saved to {results_dir}")
        return True
    else:
        print(f"\nEvaluation failed with return code {return_code}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("BLIP Distillation Pipeline")
    print("="*80)
    
    # Verify dependencies
    if not verify_dependencies():
        return
    
    # Create required directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Download COCO dataset
    if not args.skip_download:
        download_coco(args.data_dir, skip_if_exists=True)
    
    # Create a subset of COCO if requested
    config_path = args.config
    if args.subset:
        coco_dir = os.path.join(args.data_dir, "coco")
        subset_paths = create_subset(coco_dir, args.subset_size)
        config_path = update_config_for_subset(args.config, subset_paths)
    
    # Run training
    training_success = run_training(args, config_path)
    
    # Run evaluation if requested and training was successful
    if args.eval_after_train and training_success:
        run_evaluation(args)
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    main()
