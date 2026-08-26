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
    image = tf.io.decode_raw(example["image"], tf.float32)
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
    """Split processed dataset into train/val/test (memory-efficient)."""
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
    
    train_split = config.data.train_split
    val_split = config.data.val_split
    
    output_dirs = {
        "train": Path("data/train"),
        "val": Path("data/val"),
        "test": Path("data/test")
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    writers = {
        "train": tf.io.TFRecordWriter(str(output_dirs["train"] / "dataset.tfrecord")),
        "val": tf.io.TFRecordWriter(str(output_dirs["val"] / "dataset.tfrecord")),
        "test": tf.io.TFRecordWriter(str(output_dirs["test"] / "dataset.tfrecord")),
    }
    
    counts = {"train": 0, "val": 0, "test": 0}
    
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    
    for i, raw_record in enumerate(raw_dataset):
        r = hash(str(i)) % 10000 / 10000.0
        if r < train_split:
            split = "train"
        elif r < train_split + val_split:
            split = "val"
        else:
            split = "test"
        writers[split].write(raw_record.numpy())
        counts[split] += 1
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Split progress: {i + 1}/{num_samples}")
    
    for w in writers.values():
        w.close()
    
    logger.info(f"Split complete - Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
    
    for split_name in ["train", "val", "test"]:
        split_metadata = {
            "num_samples": counts[split_name],
            "image_size": list(image_size),
            "class_names": metadata["class_names"]
        }
        with open(output_dirs[split_name] / "metadata.json", "w") as f:
            json.dump(split_metadata, f, indent=2)


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