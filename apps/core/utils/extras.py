import logging
import base64
import requests
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for fingerprint augmentation parameters."""

    rotation_angles: List[float] = None
    scales: List[float] = None
    noise_vars: List[float] = None
    blur_kernels: List[int] = None
    perspective_deltas: List[int] = None
    brightness_contrast: List[Tuple[int, int]] = None
    elastic_params: List[Tuple[float, float]] = None

    def __post_init__(self):
        """Initialize default augmentation parameters if not provided."""
        if self.rotation_angles is None:
            self.rotation_angles = list(range(5, 46, 5))  # 5, 10, 15, ..., 45° (9 angles)
        if self.scales is None:
            self.scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # 7 scales
        if self.noise_vars is None:
            self.noise_vars = list(range(10, 51, 10))  # 10, 20, 30, 40, 50 (5 noise levels)
        if self.blur_kernels is None:
            self.blur_kernels = [3, 5, 7, 9, 11]  # 5 blur levels
        if self.perspective_deltas is None:
            self.perspective_deltas = list(
                range(10, 41, 10)
            )  # 10, 20, 30, 40 (4 perspective levels)
        if self.brightness_contrast is None:
            self.brightness_contrast = [(i, i) for i in range(20, 61, 20)]  # 20, 40, 60 (3 levels)
        if self.elastic_params is None:
            self.elastic_params = [
                (i, j) for i in range(10, 31, 10) for j in range(2, 7, 2)
            ]  # 6 combinations


def _load_and_validate_image(image_path: str) -> np.ndarray:
    """Load and validate fingerprint image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    if len(image.shape) != 2:
        raise ValueError(f"Expected grayscale image, got shape: {image.shape}")

    return image


def image_to_base64(image: np.ndarray) -> str:
    """Convert a numpy image to base64 string."""
    _, buffer = cv2.imencode(".bmp", image)
    img_bytes = buffer.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64


def get_embedding_from_server(image: np.ndarray) -> np.ndarray:
    """
    Send the image as base64 to 26.120.105.190:8088/embed to get the embedding.
    """
    url = "http://26.120.105.190:8088/embed"
    img_b64 = image_to_base64(image)
    payload = {"image_base64": img_b64}
    try:
        # Increased timeout to 45 seconds to add buffer
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        if "embedding" in data:
            return np.array(data["embedding"])
        else:
            raise ValueError(f"No embedding in response: {data}")
    except Exception as e:
        logger.error(f"Failed to get embedding from server: {str(e)}")
        raise


def get_distorted_embeddings_from_server(image: np.ndarray) -> Dict[str, list]:
    """
    Send the image as base64 to 26.120.105.190:8088/embed_distorted to get embeddings for all distorted versions.

    Returns:
        Dictionary mapping distorted image filenames to their embedding vectors (as lists).
    """
    url = "http://26.120.105.190:8088/embed_distorted"
    img_b64 = image_to_base64(image)
    payload = {"image_base64": img_b64}
    try:
        # Increased timeout to 90 seconds to add buffer
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data:
            # Each value is a list (embedding vector)
            return data["embeddings"]
        else:
            raise ValueError(f"No 'embeddings' in response: {data}")
    except Exception as e:
        logger.error(f"Failed to get distorted embeddings from server: {str(e)}")
        raise


def _generate_original_embedding(image: np.ndarray):
    """Generate original embedding using the remote server (CNN-based)."""
    try:
        original_embedding = get_embedding_from_server(image)
        results = {"original": original_embedding.tolist()}
        logger.info("Successfully generated original embedding from server")
        return results, image
    except Exception as e:
        logger.error(f"Failed to generate original embedding: {str(e)}")
        raise ValueError(f"Failed to generate original embedding: {str(e)}")

def restore_fingerprint(image: np.ndarray):
    url = "http://26.120.105.190:8088/restore_fingerprint"
    img_b64 = image_to_base64(image)
    payload = {"image_base64": img_b64}
    try:
        # Increased timeout to 90 seconds to add buffer
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        if "restored" in data:
            # Each value is a list (embedding vector)
            return data["restored"]
        else:
            raise ValueError(f"No 'restored' in response: {data}")
    except Exception as e:
        logger.error(f"Failed to get distorted restored from server: {str(e)}")
        raise




def process_fingerprint_for_django(
    image_path: str,
    config: Optional[AugmentationConfig] = None,
    use_multiprocessing: bool = True,
    max_workers: Optional[int] = None,
    skip_complex_augmentations: bool = False,
) -> Dict[str, list]:
    """
    Process fingerprint image and get embedding from remote server (CNN-based).

    Args:
        image_path: Path to the fingerprint image
        config: AugmentationConfig object with parameters (unused here)
        use_multiprocessing: Ignored
        max_workers: Ignored
        skip_complex_augmentations: Ignored

    Returns:
        Dictionary containing the embedding
    """
    # Load and validate image
    image = _load_and_validate_image(image_path)

    # Generate original embedding from server
    results, _ = _generate_original_embedding(image)

    logger.info(f"Generated {len(results)} embeddings total (from server)")
    return results


if __name__ == "__main__":
    # Example usage
    image_path = r"C:\Users\Karar\Desktop\Programming\HurryApp 3\Main project\Ai\fingerprints\karar\20250912_155635.bmp"

    print("Getting embedding from remote server:")
    results = process_fingerprint_for_django(
        image_path,
        skip_complex_augmentations=False,
    )
    print(f"Generated {len(results)} embeddings from server")
    print(f"Embedding: {results['original']}")
