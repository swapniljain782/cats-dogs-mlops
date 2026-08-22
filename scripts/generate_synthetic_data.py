#!/usr/bin/env python3
"""Generate synthetic Cats vs Dogs dataset for demo purposes."""
import os
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm


def generate_cat_image(size=(224, 224), seed=None):
    """Generate a synthetic cat-like image."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base color (gray/brown tones for cats)
    base_color = np.random.randint(100, 180, 3)
    img_array = np.full((*size, 3), base_color, dtype=np.uint8)
    
    # Add some noise/texture
    noise = np.random.randint(-30, 30, (*size, 3))
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add "ears" - two triangles at top
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # Left ear
    ear_color = tuple(np.clip(base_color + np.random.randint(-20, 20, 3), 0, 255).tolist())
    draw.polygon([
        (size[0]//4 - 20, size[1]//4),
        (size[0]//4, size[1]//4 - 40),
        (size[0]//4 + 20, size[1]//4)
    ], fill=ear_color)
    
    # Right ear
    draw.polygon([
        (3*size[0]//4 - 20, size[1]//4),
        (3*size[0]//4, size[1]//4 - 40),
        (3*size[0]//4 + 20, size[1]//4)
    ], fill=ear_color)
    
    # Add "eyes"
    eye_color = (0, 0, 0)
    draw.ellipse([size[0]//3 - 10, size[1]//3 - 10, size[0]//3 + 10, size[1]//3 + 10], fill=eye_color)
    draw.ellipse([2*size[0]//3 - 10, size[1]//3 - 10, 2*size[0]//3 + 10, size[1]//3 + 10], fill=eye_color)
    
    return img


def generate_dog_image(size=(224, 224), seed=None):
    """Generate a synthetic dog-like image."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base color (golden/black/white tones for dogs)
    base_color = np.random.randint(120, 200, 3)
    img_array = np.full((*size, 3), base_color, dtype=np.uint8)
    
    # Add some noise/texture
    noise = np.random.randint(-30, 30, (*size, 3))
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add "ears" - floppy ears on sides
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    ear_color = tuple(np.clip(base_color + np.random.randint(-30, 10, 3), 0, 255).tolist())
    
    # Left floppy ear
    draw.ellipse([size[0]//6 - 25, size[1]//5 - 10, size[0]//6 + 15, size[1]//5 + 40], fill=ear_color)
    
    # Right floppy ear
    draw.ellipse([5*size[0]//6 - 15, size[1]//5 - 10, 5*size[0]//6 + 25, size[1]//5 + 40], fill=ear_color)
    
    # Add "snout"
    snout_color = tuple(np.clip(base_color + np.random.randint(-40, -20, 3), 0, 255).tolist())
    draw.ellipse([size[0]//2 - 30, 2*size[1]//3 - 20, size[0]//2 + 30, 2*size[1]//3 + 20], fill=snout_color)
    
    # Add "nose"
    draw.ellipse([size[0]//2 - 8, 2*size[1]//3 - 5, size[0]//2 + 8, 2*size[1]//3 + 10], fill=(0, 0, 0))
    
    return img


def generate_dataset(output_dir: str, num_samples: int = 1000):
    """Generate synthetic dataset."""
    output_path = Path(output_dir)
    cats_dir = output_path / "cats"
    dogs_dir = output_path / "dogs"
    
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic images per class...")
    
    # Generate cat images
    for i in tqdm(range(num_samples), desc="Generating cats"):
        img = generate_cat_image(seed=i)
        img.save(cats_dir / f"cat_{i:04d}.jpg", quality=90)
    
    # Generate dog images
    for i in tqdm(range(num_samples), desc="Generating dogs"):
        img = generate_dog_image(seed=i)
        img.save(dogs_dir / f"dog_{i:04d}.jpg", quality=90)
    
    print(f"Dataset generated at {output_path}")
    print(f"  Cats: {len(list(cats_dir.glob('*.jpg')))}")
    print(f"  Dogs: {len(list(dogs_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
    
    generate_dataset(output_dir, num_samples)