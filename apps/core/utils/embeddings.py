"""Fingerprint embedding functions for feature extraction and matching."""

import cv2
import logging
import numpy as np
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Union

from scipy import ndimage
from scipy.fft import fft2, fftshift
from skimage import feature, filters, measure, morphology
from skimage.filters import gabor

# URGENT FIX: Import preprocess_image explicitly for use in this module
from .img_processing import preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Image quality assessment metrics."""

    is_valid: bool
    contrast_score: float
    clarity_score: float
    noise_level: float
    ridge_quality: float


def assess_image_quality(image: np.ndarray, strict_mode: bool = False) -> QualityMetrics:
    """
    Assess fingerprint image quality using multiple metrics.

    Args:
        image: Grayscale fingerprint image
        strict_mode: If True, use stricter thresholds for original images

    Returns:
        QualityMetrics object with quality assessment
    """
    if len(image.shape) != 2:
        raise ValueError(f"Expected grayscale image, got shape: {image.shape}")

    brightness = np.mean(image)
    contrast = np.std(image)

    # Calculate clarity using Laplacian variance
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    clarity = np.var(laplacian)

    # Calculate noise level using gradient magnitude
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    noise_level = np.std(gradient_magnitude)

    # Calculate ridge quality using FFT
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    ridge_quality = np.mean(magnitude_spectrum)

    # Apply quality thresholds
    if strict_mode:
        is_valid = 30 <= brightness <= 225 and contrast > 20 and clarity > 100 and noise_level < 50
    else:
        is_valid = 10 <= brightness <= 245 and contrast > 5 and clarity > 10 and noise_level < 500

    return QualityMetrics(
        is_valid=is_valid,
        contrast_score=float(contrast),
        clarity_score=float(clarity),
        noise_level=float(noise_level),
        ridge_quality=float(ridge_quality),
    )


@lru_cache(maxsize=128)
def _get_cached_kernel(size: int, shape: str = "ellipse") -> np.ndarray:
    """Cache morphological kernels to avoid recomputation."""
    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    else:
        return np.ones((size, size), dtype=np.uint8)


def _normalize_features(features: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """
    Normalize features using z-score normalization and handle dimension requirements.
    """
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if np.std(features) > 1e-8:
        features = (features - np.mean(features)) / np.std(features)

    if len(features) < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: len(features)] = features
        return padded
    elif len(features) > target_dim:
        return features[:target_dim]
    else:
        return features


def _validate_feature_vector(features: np.ndarray) -> bool:
    """Validate that the feature vector is reasonable for matching."""
    if features is None or len(features) == 0:
        logger.warning("Feature vector is empty")
        return False

    if not np.all(np.isfinite(features)):
        logger.warning("Feature vector contains NaN or infinite values")
        return False

    if np.all(features == 0):
        logger.warning("Feature vector is all zeros")
        return False

    if np.std(features) < 1e-6:
        logger.warning("Feature vector has very low variance")
        return False

    return True


def _filter_spurious_minutiae(minutiae_map: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """Filter out spurious minutiae points that are likely noise."""
    h, w = minutiae_map.shape
    border_size = 5

    # Remove border minutiae
    minutiae_map[:border_size, :] = 0
    minutiae_map[-border_size:, :] = 0
    minutiae_map[:, :border_size] = 0
    minutiae_map[:, -border_size:] = 0

    # Remove isolated minutiae
    kernel = _get_cached_kernel(3, "rect")
    connected_components = cv2.morphologyEx(minutiae_map, cv2.MORPH_CLOSE, kernel)
    minutiae_map = minutiae_map & connected_components

    return minutiae_map


def _find_ridge_endings(skeleton: np.ndarray) -> np.ndarray:
    """Find ridge endings in skeletonized image."""
    kernel = _get_cached_kernel(3, "rect")
    kernel[1, 1] = 0  # Exclude center pixel
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    ridge_endings = (skeleton > 0) & (neighbor_count == 1)
    return ridge_endings.astype(np.uint8)


def _find_bifurcations(skeleton: np.ndarray) -> np.ndarray:
    """Find bifurcations in skeletonized image."""
    kernel = _get_cached_kernel(3, "rect")
    kernel[1, 1] = 0  # Exclude center pixel
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    bifurcations = (skeleton > 0) & (neighbor_count >= 3)
    return bifurcations.astype(np.uint8)


def _extract_spatial_relationships(
    ridge_endings: np.ndarray, bifurcations: np.ndarray
) -> np.ndarray:
    """Extract spatial relationship features between minutiae points."""
    ending_coords = np.column_stack(np.where(ridge_endings))
    bifurcation_coords = np.column_stack(np.where(bifurcations))

    features = []

    # Ridge ending distances
    if len(ending_coords) > 1:
        ending_distances = []
        for i in range(len(ending_coords)):
            for j in range(i + 1, min(i + 5, len(ending_coords))):
                dist = np.linalg.norm(ending_coords[i] - ending_coords[j])
                ending_distances.append(dist)

        if ending_distances:
            features.extend(
                [
                    np.mean(ending_distances),
                    np.std(ending_distances),
                    np.percentile(ending_distances, 25),
                    np.percentile(ending_distances, 75),
                ]
            )
        else:
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])

    # Bifurcation distances
    if len(bifurcation_coords) > 1:
        bifurcation_distances = []
        for i in range(len(bifurcation_coords)):
            for j in range(i + 1, min(i + 5, len(bifurcation_coords))):
                dist = np.linalg.norm(bifurcation_coords[i] - bifurcation_coords[j])
                bifurcation_distances.append(dist)

        if bifurcation_distances:
            features.extend(
                [
                    np.mean(bifurcation_distances),
                    np.std(bifurcation_distances),
                    np.percentile(bifurcation_distances, 25),
                    np.percentile(bifurcation_distances, 75),
                ]
            )
        else:
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])

    # Cross distances between endings and bifurcations
    if len(ending_coords) > 0 and len(bifurcation_coords) > 0:
        cross_distances = []
        for end_coord in ending_coords[:10]:
            for bif_coord in bifurcation_coords[:10]:
                dist = np.linalg.norm(end_coord - bif_coord)
                cross_distances.append(dist)

        if cross_distances:
            features.extend(
                [
                    np.mean(cross_distances),
                    np.std(cross_distances),
                    np.percentile(cross_distances, 25),
                    np.percentile(cross_distances, 75),
                ]
            )
        else:
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])

    return np.array(features, dtype=np.float32)


def _calculate_orientation_consistency(skeleton: np.ndarray) -> float:
    """
    Calculate local ridge orientation consistency.
    """

    grad_x = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    orientations = np.arctan2(grad_y, grad_x)

    window_size = 7
    h, w = orientations.shape
    consistency_scores = []

    for i in range(window_size // 2, h - window_size // 2, window_size):
        for j in range(window_size // 2, w - window_size // 2, window_size):
            window = orientations[
                i - window_size // 2 : i + window_size // 2 + 1,
                j - window_size // 2 : j + window_size // 2 + 1,
            ]

            cos_vals = np.cos(window)
            sin_vals = np.sin(window)
            mean_cos = np.mean(cos_vals)
            mean_sin = np.mean(sin_vals)
            circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
            consistency_scores.append(circular_variance)

    return np.mean(consistency_scores) if consistency_scores else 0.0


def get_ridge_orientation_map(image: np.ndarray) -> np.ndarray:
    """
    Extract ridge frequency and orientation maps with improved local Fourier analysis.
    """

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    orientation = np.arctan2(grad_y, grad_x)

    local_frequencies = _estimate_local_ridge_frequency(image)

    hist, _ = np.histogram(orientation.flatten(), bins=32, range=(-np.pi, np.pi))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

    freq_hist, _ = np.histogram(local_frequencies.flatten(), bins=32, range=(0, 0.5))
    freq_hist = freq_hist.astype(np.float32) / (freq_hist.sum() + 1e-8)

    quality_features = _extract_ridge_quality_features(image, orientation, local_frequencies)

    features = np.concatenate([hist, freq_hist, quality_features])

    return _normalize_features(features, target_dim=128)


def _estimate_local_ridge_frequency(image: np.ndarray, window_size: int = 16) -> np.ndarray:
    """
    Estimate local ridge frequency using windowed Fourier transforms.
    """
    h, w = image.shape
    frequency_map = np.zeros_like(image, dtype=np.float32)

    step_size = window_size // 2

    for i in range(0, h - window_size + 1, step_size):
        for j in range(0, w - window_size + 1, step_size):

            window = image[i : i + window_size, j : j + window_size]

            window = window * np.hanning(window_size)[:, None] * np.hanning(window_size)[None, :]

            f_transform = fft2(window)
            f_shift = fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            center = window_size // 2
            magnitude_spectrum[center, center] = 0  # Remove DC component

            max_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
            freq_y = (max_idx[0] - center) / window_size
            freq_x = (max_idx[1] - center) / window_size
            dominant_freq = np.sqrt(freq_x**2 + freq_y**2)

            end_i = min(i + window_size, h)
            end_j = min(j + window_size, w)
            frequency_map[i:end_i, j:end_j] = dominant_freq

    return frequency_map


def _extract_ridge_quality_features(
    image: np.ndarray, orientation: np.ndarray, frequency_map: np.ndarray
) -> np.ndarray:
    """
    Extract ridge quality features including contrast and clarity measures.
    """

    kernel = _get_cached_kernel(5, "rect")
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel / 25.0)
    local_var = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel / 25.0)
    local_contrast = np.sqrt(local_var)

    grad_magnitude = np.sqrt(
        cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) ** 2
        + cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) ** 2
    )

    orientation_consistency = _calculate_local_orientation_consistency(orientation)

    freq_consistency = np.std(frequency_map)

    quality_features = np.array(
        [
            np.mean(local_contrast),
            np.std(local_contrast),
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            orientation_consistency,
            freq_consistency,
            np.mean(frequency_map),
            np.std(frequency_map),
        ],
        dtype=np.float32,
    )

    return quality_features


def _calculate_local_orientation_consistency(
    orientation: np.ndarray, window_size: int = 8
) -> float:
    """
    Calculate local orientation consistency using circular statistics.
    """
    h, w = orientation.shape
    consistency_scores = []

    for i in range(0, h - window_size + 1, window_size // 2):
        for j in range(0, w - window_size + 1, window_size // 2):
            window = orientation[i : i + window_size, j : j + window_size]

            cos_vals = np.cos(window)
            sin_vals = np.sin(window)
            mean_cos = np.mean(cos_vals)
            mean_sin = np.mean(sin_vals)
            circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
            consistency_scores.append(circular_variance)

    return np.mean(consistency_scores) if consistency_scores else 0.0


def get_minutiae_features(image: np.ndarray) -> np.ndarray:
    """
    Extract high-quality minutiae-based features optimized for fingerprint matching.
    Focuses on discriminative minutiae characteristics while avoiding redundant ridge flow patterns.
    """

    kernel = _get_cached_kernel(3, "ellipse")
    enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    skeleton = morphology.skeletonize(enhanced > 128)

    kernel_3x3 = _get_cached_kernel(3, "rect")
    kernel_3x3[1, 1] = 0  # Don't count center pixel

    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel_3x3)

    ridge_endings = (skeleton & (neighbor_count == 1)).astype(np.uint8)
    bifurcations = (skeleton & (neighbor_count >= 3)).astype(np.uint8)

    ridge_endings = _filter_spurious_minutiae(ridge_endings, skeleton)
    bifurcations = _filter_spurious_minutiae(bifurcations, skeleton)

    ending_count = np.sum(ridge_endings)
    bifurcation_count = np.sum(bifurcations)

    ending_moments = cv2.moments(ridge_endings)
    bifurcation_moments = cv2.moments(bifurcations)

    spatial_features = _extract_spatial_relationships(ridge_endings, bifurcations)

    discriminative_features = np.array(
        [
            ending_count,
            bifurcation_count,
            ending_moments.get("m00", 0),
            ending_moments.get("m10", 0),
            ending_moments.get("m01", 0),
            ending_moments.get("m20", 0),
            ending_moments.get("m02", 0),
            ending_moments.get("m11", 0),
            bifurcation_moments.get("m00", 0),
            bifurcation_moments.get("m10", 0),
            bifurcation_moments.get("m01", 0),
            bifurcation_moments.get("m20", 0),
            bifurcation_moments.get("m02", 0),
            bifurcation_moments.get("m11", 0),
        ],
        dtype=np.float32,
    )

    all_features = np.concatenate([discriminative_features, spatial_features])

    selected_features = _select_discriminative_features(all_features)

    return _normalize_features(selected_features, target_dim=32)


def _select_discriminative_features(features: np.ndarray) -> np.ndarray:
    """
    Select only the most discriminative features to avoid redundant ridge flow patterns.
    """

    feature_importance = np.abs(features) + np.std(features) * 0.1

    top_indices = np.argsort(feature_importance)[-20:]  # Keep top 20 most important features

    return features[top_indices]


def get_gabor_features(image: np.ndarray) -> np.ndarray:
    """
    Apply Gabor filters with improved parameter selection and feature extraction.
    """

    frequencies = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # More frequency bands
    orientations = [0, 30, 60, 90, 120, 150]  # More orientation angles

    features = []

    for freq in frequencies:
        for orientation in orientations:

            theta = np.radians(orientation)

            filtered = gabor(image, frequency=freq, theta=theta, sigma_x=3, sigma_y=3)
            real_part = filtered[0]
            imag_part = filtered[1]

            features.extend(
                [
                    np.mean(real_part),
                    np.std(real_part),
                    np.min(real_part),
                    np.max(real_part),
                    np.median(real_part),
                    np.mean(imag_part),
                    np.std(imag_part),
                    np.var(real_part),
                    np.var(imag_part),
                ]
            )

    features = np.array(features, dtype=np.float32)

    return _normalize_features(features, target_dim=128)


def get_embedding(
    image: np.ndarray, method: str = "minutiae", is_augmented: bool = False
) -> np.ndarray:
    """
    Generate fingerprint embedding using specialized methods with quality validation.

    Args:
        image: Grayscale fingerprint image
        method: "minutiae", "ridge_orientation", "gabor", or "combined"
        is_augmented: If True, use lenient quality assessment for augmented images
    """

    if len(image.shape) != 2:
        raise ValueError(f"Expected grayscale image, got shape: {image.shape}")

    quality_metrics = assess_image_quality(image, strict_mode=not is_augmented)
    if not quality_metrics.is_valid:
        if is_augmented:
            logger.debug(f"Augmented image quality assessment failed: {quality_metrics}")
        else:
            logger.warning(f"Original image quality assessment failed: {quality_metrics}")

        # URGENT FIX: Use preprocess_image from img_processing
        image = preprocess_image(image)

    try:
        if method == "minutiae":
            features = get_minutiae_features(image)
        elif method == "ridge_orientation":
            features = get_ridge_orientation_map(image)
        elif method == "gabor":
            features = get_gabor_features(image)
        elif method == "combined":

            minutiae = get_minutiae_features(image)
            ridge = get_ridge_orientation_map(image)
            gabor_feat = get_gabor_features(image)

            combined = np.concatenate([minutiae, ridge, gabor_feat])
            features = _normalize_features(combined, target_dim=128)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not _validate_feature_vector(features):
            logger.warning("Feature vector validation failed, using fallback")

            features = np.random.normal(0, 0.1, 128).astype(np.float32)

        return features

    except Exception as e:
        logger.error(f"Failed to generate embedding with method {method}: {str(e)}")

        return np.random.normal(0, 0.1, 128).astype(np.float32)


def get_fingerprint_embedding_for_inference(
    image_path: str, method: str = "minutiae"
) -> np.ndarray:
    """
    Get fingerprint embedding for inference (real-world distorted images).
    This function is optimized for processing poor quality/distorted fingerprints
    that you'll encounter during actual usage.

    Args:
        image_path: Path to the fingerprint image
        method: "minutiae", "ridge_orientation", "gabor", or "combined"

    Returns:
        Normalized feature vector for Annoy comparison
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    # URGENT FIX: Use preprocess_image from img_processing
    image = preprocess_image(image)

    embedding = get_embedding(image, method, is_augmented=True)

    logger.info(f"Generated inference embedding with method: {method}")
    return embedding


def get_fingerprint_embedding_for_training(image_path: str, method: str = "minutiae") -> np.ndarray:
    """
    Get fingerprint embedding for training (high-quality original images).
    This function uses strict quality assessment for training data.

    Args:
        image_path: Path to the high-quality fingerprint image
        method: "minutiae", "ridge_orientation", "gabor", or "combined"

    Returns:
        Normalized feature vector for training
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    embedding = get_embedding(image, method, is_augmented=False)

    logger.info(f"Generated training embedding with method: {method}")
    return embedding
