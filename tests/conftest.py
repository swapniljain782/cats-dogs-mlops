"""Pytest configuration and shared fixtures."""
import sys
import os
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """Return the project root directory."""
    return project_root


@pytest.fixture(scope="session")
def sample_image_bytes():
    """Create a sample JPEG image as bytes."""
    from PIL import Image
    import io
    
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_image_file(tmp_path_factory):
    """Create a sample JPEG image file."""
    from PIL import Image
    
    tmp_dir = tmp_path_factory.mktemp("images")
    img_path = tmp_dir / "test_image.jpg"
    img = Image.new("RGB", (224, 224), color="red")
    img.save(str(img_path))
    return img_path
