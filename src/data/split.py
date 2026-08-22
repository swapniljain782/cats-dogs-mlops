"""Split dataset into train/validation/test sets."""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_tfrecord(example_proto, image_size):
    """Parse a single TFRecord example."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.parse_tensor(example["image"], out_type=tf.float32)
    image = tf.reshape(image, (*image_size, 3))
    label = example["label"]
    return image, label


def load_dataset_from_tfrecord(tfrecord_path: Path, image_size: tuple) -> tf.data.Dataset:
    """Load dataset from TFRecord file."""
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(
        lambda x: parse_tfrecord(x, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def split_dataset():
    """Split processed dataset into train/val/test."""
    config = get_config()
    
    processed_dir = Path("data/processed")
    tfrecord_path = processed_dir / "dataset.tfrecord"
    metadata_path = processed_dir / "metadata.json"
    
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    num_samples = metadata["num_samples"]
    image_size = tuple(metadata["image_size"])
    
    # Load all data to split
    dataset = load_dataset_from_tfrecord(tfrecord_path, image_size)
    
    # Convert to numpy arrays for splitting
    images = []
    labels = []
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Loaded {len(images)} samples for splitting")
    
    # Split ratios
    train_split = config.data.train_split
    val_split = config.data.val_split
    test_split = config.data.test_split
    
    # First split: train vs (val + test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels,
        train_size=train_split,
        stratify=labels,
        random_state=42
    )
    
    # Second split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=42
    )
    
    logger.info(f"Split sizes - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Save splits as TFRecords
    output_dirs = {
        "train": Path("data/train"),
        "val": Path("data/val"),
        "test": Path("data/test")
    }
    
    splits = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels)
    }
    
    for split_name, (split_images, split_labels) in splits.items():
        output_dir = output_dirs[split_name]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tfrecord_out = output_dir / "dataset.tfrecord"
        
        def serialize_example(image, label):
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(tf.convert_to_tensor(image)).numpy()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            }
            return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
        
        with tf.io.TFRecordWriter(str(tfrecord_out)) as writer:
            for img, lbl in zip(split_images, split_labels):
                writer.write(serialize_example(img, lbl))
        
        # Save split metadata
        split_metadata = {
            "num_samples": len(split_images),
            "image_size": list(image_size),
            "class_names": metadata["class_names"]
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(split_metadata, f, indent=2)
        
        logger.info(f"Saved {split_name} split to {output_dir} ({len(split_images)} samples)")


def get_split_dataset(split: str, batch_size: int = None, shuffle: bool = True) -> tf.data.Dataset:
    """Get a TensorFlow dataset for a specific split."""
    config = get_config()
    batch_size = batch_size or config.data.batch_size
    image_size = (config.data.image_size, config.data.image_size)
    
    split_dir = Path(f"data/{split}")
    tfrecord_path = split_dir / "dataset.tfrecord"
    
    dataset = load_dataset_from_tfrecord(tfrecord_path, image_size)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def main():
    """Main entry point for DVC stage."""
    split_dataset()


if __name__ == "__main__":
    main()