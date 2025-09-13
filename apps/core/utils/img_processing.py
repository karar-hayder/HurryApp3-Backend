"""Only functions for processing/distorting images"""

import logging
import random
from typing import List, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Avoid circular import: import get_embedding only inside functions that need it


def _rotate_image_vectorized(image: np.ndarray, angles: np.ndarray) -> List[np.ndarray]:
    """Vectorized rotation for multiple angles."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    results = []

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        results.append(rotated)

    return results


def _scale_image_vectorized(image: np.ndarray, scales: np.ndarray) -> List[np.ndarray]:
    """Vectorized scaling for multiple scale factors."""
    h, w = image.shape[:2]
    results = []

    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        final = cv2.resize(resized, (w, h))
        results.append(final)

    return results


def _add_noise_vectorized(
    image: np.ndarray, noise_vars: np.ndarray
) -> List[np.ndarray]:
    """Vectorized noise addition."""
    h, w = image.shape[:2]
    results = []

    for var in noise_vars:
        noise = np.random.normal(0, np.sqrt(var), (h, w)).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        results.append(noisy)

    return results


def _blur_image_vectorized(image: np.ndarray, kernels: np.ndarray) -> List[np.ndarray]:
    """Vectorized blurring with different kernel sizes."""
    results = []

    for ksize in kernels:
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        results.append(blurred)

    return results


def perspective_warp(image: np.ndarray, delta: int) -> np.ndarray:
    """Apply perspective transformation to image."""
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32(
        [
            [random.randint(0, delta), random.randint(0, delta)],
            [w - random.randint(0, delta), random.randint(0, delta)],
            [random.randint(0, delta), h - random.randint(0, delta)],
            [w - random.randint(0, delta), h - random.randint(0, delta)],
        ]
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))


def adjust_brightness_contrast(
    image: np.ndarray, brightness: int, contrast: int
) -> np.ndarray:
    """Adjust brightness and contrast of image."""
    return cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)


def elastic_transform(image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Apply elastic transformation to image."""
    kernel_size = min(17, max(3, int(sigma * 3)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    dx = cv2.GaussianBlur(dx, (kernel_size, kernel_size), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (kernel_size, kernel_size), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to handle edge cases and improve quality.

    Args:
        image: Input grayscale image

    Returns:
        Preprocessed image
    """
    # Adjust brightness if too dark or too bright
    if np.mean(image) < 30:
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=50)
    elif np.mean(image) > 225:
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=-30)

    # Enhance contrast if too low
    if np.std(image) < 20:
        image = cv2.equalizeHist(image)

    # Apply median filter to reduce noise
    image = cv2.medianBlur(image, 3)

    return image


def _process_augmentation_batch(args: Tuple) -> Dict[str, list]:
    """Process a batch of augmentations in parallel."""
    # Import here to avoid circular import at module level
    from apps.core.utils.embeddings import get_embedding

    image, augmentation_type, params, method = args
    results = {}

    try:
        if augmentation_type == "rotation":
            angles = np.array(params)
            augmented_images = _rotate_image_vectorized(image, angles)
            for i, (angle, aug_img) in enumerate(zip(angles, augmented_images)):
                embedding = get_embedding(aug_img, method, is_augmented=True)
                results[f"rotated_{angle}_v{i+1}"] = embedding.tolist()

        elif augmentation_type == "scale":
            scales = np.array(params)
            augmented_images = _scale_image_vectorized(image, scales)
            for i, (scale, aug_img) in enumerate(zip(scales, augmented_images)):
                embedding = get_embedding(aug_img, method, is_augmented=True)
                results[f"scaled_{scale}_v{i+1}"] = embedding.tolist()

        elif augmentation_type == "noise":
            noise_vars = np.array(params)
            augmented_images = _add_noise_vectorized(image, noise_vars)
            for i, (var, aug_img) in enumerate(zip(noise_vars, augmented_images)):
                embedding = get_embedding(aug_img, method, is_augmented=True)
                results[f"gaussian_noise_{var}_v{i+1}"] = embedding.tolist()

        elif augmentation_type == "blur":
            kernels = np.array(params)
            augmented_images = _blur_image_vectorized(image, kernels)
            for i, (k, aug_img) in enumerate(zip(kernels, augmented_images)):
                embedding = get_embedding(aug_img, method, is_augmented=True)
                results[f"blurred_k{k}_v{i+1}"] = embedding.tolist()

    except Exception as e:
        logger.error(f"Error in augmentation batch {augmentation_type}: {str(e)}")

    return results


def _process_complex_augmentations(
    image: np.ndarray, results: Dict[str, list], config, method: str
):
    """Process complex augmentations that can't be easily vectorized."""

    # Import here to avoid circular import at module level
    from apps.core.utils.embeddings import get_embedding

    def perspective_warp(image, delta):
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32(
            [
                [random.randint(0, delta), random.randint(0, delta)],
                [w - random.randint(0, delta), random.randint(0, delta)],
                [random.randint(0, delta), h - random.randint(0, delta)],
                [w - random.randint(0, delta), h - random.randint(0, delta)],
            ]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (w, h))

    def adjust_brightness_contrast(image, brightness, contrast):
        return cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)

    def elastic_transform(image, alpha, sigma):
        kernel_size = min(17, max(3, int(sigma * 3)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        dx = cv2.GaussianBlur(dx, (kernel_size, kernel_size), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (kernel_size, kernel_size), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    logger.info(
        f"Processing {len(config.perspective_deltas)} perspective transformations"
    )
    for i, delta in enumerate(config.perspective_deltas):
        try:
            img = perspective_warp(image, delta)
            embedding = get_embedding(img, method, is_augmented=True)
            results[f"perspective_{delta}_v{i+1}"] = embedding.tolist()
            if (i + 1) % 5 == 0:  # Log progress every 5 items
                logger.info(
                    f"Completed {i+1}/{len(config.perspective_deltas)} perspective transformations"
                )
        except Exception as e:
            logger.warning(f"Failed to generate perspective_{delta}_v{i+1}: {str(e)}")

    logger.info(
        f"Processing {len(config.brightness_contrast)} brightness/contrast adjustments"
    )
    for i, (b, c) in enumerate(config.brightness_contrast):
        try:
            img = adjust_brightness_contrast(image, b, c)
            embedding = get_embedding(img, method, is_augmented=True)
            results[f"brightness_contrast_{b}_{c}_v{i+1}"] = embedding.tolist()
            if (i + 1) % 5 == 0:  # Log progress every 5 items
                logger.info(
                    f"Completed {i+1}/{len(config.brightness_contrast)} brightness/contrast adjustments"
                )
        except Exception as e:
            logger.warning(
                f"Failed to generate brightness_contrast_{b}_{c}_v{i+1}: {str(e)}"
            )

    logger.info(f"Processing {len(config.elastic_params)} elastic transformations")
    for i, (alpha, sigma) in enumerate(config.elastic_params):
        try:
            img = elastic_transform(image, alpha, sigma)
            embedding = get_embedding(img, method, is_augmented=True)
            results[f"elastic_{alpha}_{sigma}_v{i+1}"] = embedding.tolist()
            if (i + 1) % 10 == 0:  # Log progress every 10 items (elastic is slower)
                logger.info(
                    f"Completed {i+1}/{len(config.elastic_params)} elastic transformations"
                )
        except Exception as e:
            logger.warning(
                f"Failed to generate elastic_{alpha}_{sigma}_v{i+1}: {str(e)}"
            )
