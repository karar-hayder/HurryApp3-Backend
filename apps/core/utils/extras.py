import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import from __init__ to ensure correct import context for multiprocessing
from . import *

# Explicit import for use in subprocesses (fixes multiprocessing import issues)
from .img_processing import preprocess_image

# Import get_embedding and assess_image_quality directly to avoid NameError
from .embeddings import get_embedding, assess_image_quality

# If _process_augmentation_batch or _process_complex_augmentations are needed, import them explicitly
try:
    from .img_processing import (
        _process_augmentation_batch,
        _process_complex_augmentations,
    )
except ImportError:
    # fallback for legacy code or if these are defined elsewhere
    pass

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
            self.rotation_angles = list(
                range(5, 46, 5)
            )  # 5, 10, 15, ..., 45° (9 angles)
        if self.scales is None:
            self.scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # 7 scales
        if self.noise_vars is None:
            self.noise_vars = list(
                range(10, 51, 10)
            )  # 10, 20, 30, 40, 50 (5 noise levels)
        if self.blur_kernels is None:
            self.blur_kernels = [3, 5, 7, 9, 11]  # 5 blur levels
        if self.perspective_deltas is None:
            self.perspective_deltas = list(
                range(10, 41, 10)
            )  # 10, 20, 30, 40 (4 perspective levels)
        if self.brightness_contrast is None:
            self.brightness_contrast = [
                (i, i) for i in range(20, 61, 20)
            ]  # 20, 40, 60 (3 levels)
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


def _generate_original_embedding(image: np.ndarray, method: str):
    """Generate original embedding and validate image quality."""
    quality_metrics = assess_image_quality(image, strict_mode=True)
    logger.info(f"Image quality metrics: {quality_metrics}")

    if not getattr(quality_metrics, "is_valid", False):
        logger.warning("Image quality is poor, applying preprocessing")
        # Use the already-imported preprocess_image from the module scope
        image = preprocess_image(image)

    try:
        original_embedding = get_embedding(image, method)
        results = {"original": original_embedding.tolist()}
        logger.info("Successfully generated original embedding")
        return results, image
    except Exception as e:
        logger.error(f"Failed to generate original embedding: {str(e)}")
        raise ValueError(f"Failed to generate original embedding: {str(e)}")


def _process_parallel_augmentations(
    image: np.ndarray, config: AugmentationConfig, method: str, max_workers: int
) -> Dict[str, list]:
    """Process augmentations using multiprocessing."""
    augmentation_tasks = [
        ("rotation", config.rotation_angles),
        ("scale", config.scales),
        ("noise", config.noise_vars),
        ("blur", config.blur_kernels),
    ]

    results = {}
    logger.info(f"Using multiprocessing with {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_augmentation_batch, (image, aug_type, params, method)
            ): aug_type
            for aug_type, params in augmentation_tasks
        }

        for future in as_completed(future_to_task):
            aug_type = future_to_task[future]
            try:
                batch_results = future.result()
                results.update(batch_results)
                logger.info(f"Completed {aug_type} augmentation batch")
            except Exception as e:
                logger.error(f"Error in {aug_type} augmentation batch: {str(e)}")

    return results


def _process_sequential_augmentations(
    image: np.ndarray, config: AugmentationConfig, method: str
) -> Dict[str, list]:
    """Process augmentations sequentially."""
    augmentation_tasks = [
        ("rotation", config.rotation_angles),
        ("scale", config.scales),
        ("noise", config.noise_vars),
        ("blur", config.blur_kernels),
    ]

    results = {}
    logger.info("Using sequential processing")

    for aug_type, params in augmentation_tasks:
        try:
            batch_results = _process_augmentation_batch(
                (image, aug_type, params, method)
            )
            results.update(batch_results)
            logger.info(f"Completed {aug_type} augmentation batch")
        except Exception as e:
            logger.error(f"Error in {aug_type} augmentation batch: {str(e)}")

    return results


def process_fingerprint_for_django(
    image_path: str,
    method: str = "minutiae",
    config: Optional[AugmentationConfig] = None,
    use_multiprocessing: bool = True,
    max_workers: Optional[int] = None,
    skip_complex_augmentations: bool = False,
) -> Dict[str, list]:
    """
    Process fingerprint image with comprehensive augmentation techniques.

    Args:
        image_path: Path to the fingerprint image
        method: "minutiae", "ridge_orientation", "gabor", or "combined"
        config: AugmentationConfig object with parameters
        use_multiprocessing: Whether to use multiprocessing for parallel augmentation
        max_workers: Maximum number of worker processes
        skip_complex_augmentations: Skip slow complex augmentations (perspective, elastic, etc.)

    Returns:
        Dictionary containing all generated embeddings
    """
    # Initialize configuration
    if config is None:
        config = AugmentationConfig()

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers max

    # Load and validate image
    image = _load_and_validate_image(image_path)

    # Generate original embedding
    results, processed_image = _generate_original_embedding(image, method)

    # Process augmentations
    if use_multiprocessing and max_workers > 1:
        augmentation_results = _process_parallel_augmentations(
            processed_image, config, method, max_workers
        )
    else:
        augmentation_results = _process_sequential_augmentations(
            processed_image, config, method
        )

    results.update(augmentation_results)

    # Process complex augmentations if requested
    if not skip_complex_augmentations:
        logger.info(
            "Starting complex augmentations (perspective, brightness/contrast, elastic)"
        )
        _process_complex_augmentations(processed_image, results, config, method)
        logger.info("Completed complex augmentations")
    else:
        logger.info("Skipping complex augmentations for faster processing")

    logger.info(f"Generated {len(results)} embeddings total")
    return results


if __name__ == "__main__":
    # Example usage
    image_path = r"C:\Users\Karar\Desktop\Programming\HurryApp 3\Main project\Ai\fingerprints\karar\20250912_155635.bmp"

    print("Training mode with fast augmentations:")
    training_results = process_fingerprint_for_django(
        image_path,
        "minutiae",
        skip_complex_augmentations=False,
    )
    print(f"Generated {len(training_results)} embeddings for training")

    print("\nInference mode (single embedding):")
    # Import get_fingerprint_embedding_for_inference if not already imported
    try:
        from .embeddings import get_fingerprint_embedding_for_inference
    except ImportError:
        pass
    inference_embedding = get_fingerprint_embedding_for_inference(
        image_path, "combined"
    )
    print(f"Generated inference embedding with shape: {inference_embedding.shape}")
    print(f"Generated inference embedding with data: {inference_embedding}")
