import os
import json
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import BlipProcessor


class CaptioningDataset(Dataset):
    """
    Dataset for image captioning with knowledge distillation.
    """
    def __init__(
        self,
        image_dir: str,
        annotations_file: Optional[str] = None,
        processor: Optional[BlipProcessor] = None,
        processor_name: str = "Salesforce/blip-image-captioning-large",
        max_length: int = 30,
        split: str = "train",
        transform=None,
        preprocess_images: bool = True,
    ):
        """
        Initialize the captioning dataset.
        
        Args:
            image_dir (str): Directory containing the images.
            annotations_file (str, optional): Path to annotations file with captions.
            processor (BlipProcessor, optional): Processor for tokenizing text and preprocessing images.
            processor_name (str): Name of the processor to load if not provided.
            max_length (int): Maximum length for captions.
            split (str): Dataset split ('train', 'val', or 'test').
            transform: Optional transform to apply to images.
            preprocess_images (bool): Whether to preprocess images during loading.
        """
        self.image_dir = image_dir
        self.annotations_file = annotations_file
        self.max_length = max_length
        self.split = split
        self.transform = transform
        self.preprocess_images = preprocess_images
        
        # Load processor for text tokenization and image preprocessing
        self.processor = processor if processor is not None else BlipProcessor.from_pretrained(processor_name)
        
        # Load annotations if available
        self.samples = []
        if annotations_file is not None and os.path.exists(annotations_file):
            self._load_annotations()
        else:
            self._load_from_directory()
    
    def _load_annotations(self):
        """Load image-caption pairs from annotations file."""
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        # Handle different annotation formats
        if isinstance(data, dict):
            # Handle COCO-style annotations
            if 'annotations' in data:
                annotations = data['annotations']
                images = {img['id']: img['file_name'] for img in data['images']}
                
                for ann in annotations:
                    image_id = ann['image_id']
                    caption = ann['caption']
                    if image_id in images:
                        self.samples.append({
                            'image_path': os.path.join(self.image_dir, images[image_id]),
                            'caption': caption
                        })
            # Handle simpler format
            elif 'images' in data:
                for item in data['images']:
                    # Skip if not in the correct split
                    if 'split' in item and item['split'] != self.split:
                        continue
                    
                    image_path = os.path.join(self.image_dir, item['file_name'])
                    captions = item['captions'] if isinstance(item.get('captions', []), list) else [item.get('caption', '')]
                    
                    for caption in captions:
                        self.samples.append({
                            'image_path': image_path,
                            'caption': caption
                        })
        elif isinstance(data, list):
            # Handle list of annotations
            for item in data:
                image_path = os.path.join(self.image_dir, item['file_name'])
                captions = item['captions'] if isinstance(item.get('captions', []), list) else [item.get('caption', '')]
                
                for caption in captions:
                    self.samples.append({
                        'image_path': image_path,
                        'caption': caption
                    })
    
    def _load_from_directory(self):
        """Load images from directory without annotations."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_path = os.path.join(root, file)
                    self.samples.append({
                        'image_path': image_path,
                        'caption': ''  # No caption available
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get an item from the dataset."""
        item = self.samples[idx]
        image_path = item['image_path']
        caption = item['caption']
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Process image and caption with the processor
        if self.preprocess_images:
            if caption:
                inputs = self.processor(image, caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                return inputs
            else:
                inputs = self.processor(image, return_tensors="pt")
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                return inputs
        else:
            # Return the PIL image and caption for later processing
            if caption:
                return {"image": image, "caption": caption, "image_path": image_path}
            else:
                return {"image": image, "image_path": image_path}


def get_captioning_dataloader(
    image_dir: Optional[str] = None,
    annotations_file: Optional[str] = None,
    processor: Optional[BlipProcessor] = None,
    processor_name: str = "Salesforce/blip-image-captioning-large",
    batch_size: int = 32,
    max_length: int = 30,
    split: str = "train",
    shuffle: bool = None,
    num_workers: int = 4,
    preprocess_images: bool = True,
    dataset_name: Optional[str] = None,
    image_size: int = 384,
    subset_size: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    """
    Create a DataLoader for image captioning.
    
    Args:
        image_dir (str, optional): Directory containing the images.
        annotations_file (str, optional): Path to annotations file with captions.
        processor (BlipProcessor, optional): Processor for tokenizing text and preprocessing images.
        processor_name (str): Name of the processor to load if not provided.
        batch_size (int): Batch size for the dataloader.
        max_length (int): Maximum sequence length for captions.
        split (str): Dataset split ('train', 'val', or 'test').
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True for train, False otherwise.
        num_workers (int): Number of workers for data loading.
        preprocess_images (bool): Whether to preprocess images during loading.
        dataset_name (str, optional): Name of the dataset to use (e.g., 'coco'). If provided, will use predefined paths.
        image_size (int): Size to resize images to.
        subset_size (int, optional): If provided, use only a subset of this size from the dataset.
        pin_memory (bool): Whether to pin memory in the dataloader.
        prefetch_factor (int): Number of batches to prefetch.
        persistent_workers (bool): Whether to keep worker processes alive after dataset iteration.
        
    Returns:
        DataLoader: DataLoader for the captioning dataset.
    """
    # Set default shuffle based on split if not explicitly provided
    if shuffle is None:
        shuffle = (split == "train")
    
    # Handle dataset_name to set default paths
    if dataset_name is not None:
        if dataset_name.lower() == "coco":
            # Set default COCO paths if not provided
            if image_dir is None:
                image_dir = f"data/coco/{split}2017"
            if annotations_file is None:
                annotations_file = f"data/coco/annotations/captions_{split}2017.json"
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    # Ensure image_dir is provided
    if image_dir is None:
        raise ValueError("Either image_dir or dataset_name must be provided")
    
    dataset = CaptioningDataset(
        image_dir=image_dir,
        annotations_file=annotations_file,
        processor=processor,
        processor_name=processor_name,
        max_length=max_length,
        split=split,
        preprocess_images=preprocess_images
    )
    
    # Apply subset if specified
    if subset_size is not None and subset_size > 0 and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn if not preprocess_images else None,
    }
    
    # Add additional kwargs only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs.update({
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers
        })
    
    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs
    )
    
    return dataloader


def collate_fn(batch):
    """
    Custom collate function for batching items that have not been preprocessed.
    
    Args:
        batch: List of dictionaries from the dataset.
        
    Returns:
        dict: Batched data.
    """
    # Separate images and captions
    images = [item["image"] for item in batch]
    captions = [item.get("caption", "") for item in batch]
    image_paths = [item["image_path"] for item in batch]
    
    return {
        "images": images,
        "captions": captions,
        "image_paths": image_paths
    }
