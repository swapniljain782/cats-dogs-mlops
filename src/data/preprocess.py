"""Image preprocessing and data augmentation for Cats vs Dogs classification."""
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import tensorflow as tf
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def create_augmentation_pipeline(
    rotation_range: int = 20,
    zoom_range: float = 0.2,
    horizontal_flip: bool = True,
    brightness_range: Tuple[float, float] = (0.8, 1.2)
) -> tf.keras.Sequential:
    """Create data augmentation pipeline."""
    layers = []
    
    if rotation_range > 0:
        layers.append(tf.keras.layers.RandomRotation(rotation_range / 360.0))
    
    if zoom_range > 0:
        layers.append(tf.keras.layers.RandomZoom(zoom_range))
    
    if horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    
    if brightness_range:
        layers.append(tf.keras.layers.RandomBrightness(
            factor=(brightness_range[0] - 1.0, brightness_range[1] - 1.0),
            value_range=(0.0, 1.0)
        ))
    
    return tf.keras.Sequential(layers) if layers else None


def get_image_paths_and_labels(data_dir: Path) -> Tuple[List[str], List[int]]:
    """Get image paths and labels from directory structure."""
    image_paths = []
    labels = []
    
    # Expect structure: data_dir/class_name/*.jpg
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    for class_idx, class_dir in enumerate(class_dirs):
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append(str(img_path))
                labels.append(class_idx)
    
    logger.info(f"Found {len(image_paths)} images in {len(class_dirs)} classes")
    for i, class_dir in enumerate(class_dirs):
        count = sum(1 for _ in class_dir.glob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        logger.info(f"  Class {i} ({class_dir.name}): {count} images")
    
    return image_paths, labels


def create_tf_dataset(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    augment: bool = False,
    augmentation_pipeline: tf.keras.Sequential = None
) -> tf.data.Dataset:
    """Create TensorFlow dataset from image paths and labels."""
    
    valid_paths = []
    valid_labels = []
    for p, l in zip(image_paths, labels):
        try:
            with Image.open(p) as img:
                img.load()
                img.convert("RGB")
            valid_paths.append(p)
            valid_labels.append(l)
        except Exception:
            logger.warning(f"Skipping corrupted image: {p}")
    
    image_paths = valid_paths
    labels = valid_labels
    
    def load_image(path, label):
        def _load(p):
            p = p.numpy().decode()
            img = Image.open(p).convert("RGB")
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.float32) / 255.0
        img = tf.py_function(_load, [path], tf.float32)
        img.set_shape((*image_size, 3))
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    if augment and augmentation_pipeline:
        # CPU augmentation (Apple Silicon / no GPU) - avoids GPU memory contention
        def augment_fn(img, label):
            with tf.device('/CPU:0'):
                return augmentation_pipeline(img, training=True), label
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        # GPU augmentation (NVIDIA GPU) - uncomment below and comment CPU version above
        # def augment_fn(img, label):
        #     return augmentation_pipeline(img, training=True), label
        # dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def preprocess_dataset(raw_data_dir: Path, output_dir: Path) -> None:
    """Preprocess raw dataset and split directly into train/val/test TFRecords."""
    config = get_config()
    
    image_size = (config.data.image_size, config.data.image_size)
    batch_size = config.data.batch_size
    
    # Get all image paths and labels
    image_paths, labels = get_image_paths_and_labels(raw_data_dir)
    num_samples = len(image_paths)
    
    # Shuffle and split indices
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_end = int(num_samples * config.data.train_split)
    val_end = train_end + int(num_samples * config.data.val_split)
    
    split_indices = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    
    # Ensure output dirs exist
    for split_name in split_indices:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)
    
    writers = {
        name: tf.io.TFRecordWriter(str(output_dir / name / "dataset.tfrecord"))
        for name in split_indices
    }
    
    counts = {name: 0 for name in split_indices}
    
    def serialize_example(image, label):
        raw_bytes = image.numpy().astype(np.float32).tobytes()
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_bytes])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label.numpy())])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    
    # Process images without augmentation for efficiency
    for i, (idx, path, label) in enumerate(zip(indices, [image_paths[j] for j in indices], [labels[j] for j in indices])):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            split_name = "train" if i < train_end else ("val" if i < val_end else "test")
            
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_array.tobytes()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feature))
            writers[split_name].write(ex.SerializeToString())
            counts[split_name] += 1
        except Exception as e:
            logger.warning(f"Skipping corrupted image {path}: {e}")
            continue
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Progress: {i + 1}/{num_samples}")
    
    for w in writers.values():
        w.close()
    
    # Save metadata for each split
    class_names = sorted([d.name for d in raw_data_dir.iterdir() if d.is_dir()])
    for split_name in split_indices:
        metadata = {
            "num_samples": counts[split_name],
            "image_size": list(image_size),
            "class_names": class_names,
        }
        with open(output_dir / split_name / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Done - Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
    
    logger.info(f"Preprocessing complete. Saved {len(image_paths)} samples.")


def main():
    """Main entry point for DVC stage."""
    config = get_config()
    raw_dir = Path("data/raw/PetImages")
    output_dir = Path("data")
    preprocess_dataset(raw_dir, output_dir)


if __name__ == "__main__":
    main()