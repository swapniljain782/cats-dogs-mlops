"""Download dataset from Kaggle using kagglehub."""
import os
import shutil
from pathlib import Path
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def download_dataset() -> Path:
    """Download the Cats and Dogs dataset from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Install with: pip install kagglehub")
        raise
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading dataset from Kaggle...")
    
    # Download dataset
    path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
    
    source_path = Path(path)
    logger.info(f"Dataset downloaded to: {source_path}")
    
    # Copy to our data/raw directory
    if source_path.exists():
        # The dataset might have subdirectories, copy everything
        for item in source_path.iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        logger.info(f"Dataset copied to: {output_dir}")
    else:
        logger.error(f"Downloaded path does not exist: {source_path}")
        raise FileNotFoundError(f"Dataset not found at {source_path}")
    
    return output_dir


def main():
    """Main entry point for DVC stage."""
    download_dataset()


if __name__ == "__main__":
    main()