import cv2
import numpy as np
import random
from typing import Dict



def get_embedding(image: np.ndarray, use_orb: bool = False) -> np.ndarray:
    if use_orb:
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros((128,), dtype=np.float32)
        
        # Ensure descriptors is a numpy array and has the right shape
        if not isinstance(descriptors, np.ndarray):
            return np.zeros((128,), dtype=np.float32)
        
        # Flatten and ensure we have at least 128 elements
        flattened = descriptors.flatten()
        if len(flattened) < 128:
            # Pad with zeros if we don't have enough features
            padded = np.zeros((128,), dtype=np.float32)
            padded[:len(flattened)] = flattened.astype(np.float32)
            return padded
        else:
            return flattened[:128].astype(np.float32)
    else:
        # For non-ORB, ensure we return a consistent size
        flattened = image.flatten().astype(np.float32) / 255.0
        if len(flattened) < 128:
            # Pad with zeros if image is too small
            padded = np.zeros((128,), dtype=np.float32)
            padded[:len(flattened)] = flattened
            return padded
        else:
            return flattened[:128]
def process_fingerprint_for_django(image_path: str, use_orb: bool = False) -> Dict[str, list]:
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def scale_image(image, scale_factor):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
        return cv2.resize(resized, (w, h))  # back to original size

    def add_gaussian_noise(image, var):
        row, col = image.shape[:2]
        mean = 0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col)).astype('uint8')
        noisy = cv2.add(image, gauss)
        return noisy

    def blur_image(image, ksize):
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def perspective_warp(image, delta):
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.randint(0, delta), random.randint(0, delta)],
            [w - random.randint(0, delta), random.randint(0, delta)],
            [random.randint(0, delta), h - random.randint(0, delta)],
            [w - random.randint(0, delta), h - random.randint(0, delta)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (w, h))

    def adjust_brightness_contrast(image, brightness, contrast):
        return cv2.convertScaleAbs(image, alpha=1 + contrast/100.0, beta=brightness)

    def elastic_transform(image, alpha, sigma):
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = (random_state.rand(*shape) * 2 - 1)
        dy = (random_state.rand(*shape) * 2 - 1)
        dx = cv2.GaussianBlur(dx, (17, 17), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (17, 17), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    
    # Ensure image is valid
    if len(image.shape) != 2:
        raise ValueError(f"Expected grayscale image, got shape: {image.shape}")

    # Base output
    try:
        original_embedding = get_embedding(image, use_orb)
        results = {
            "original": original_embedding.tolist()
        }
    except Exception as e:
        raise ValueError(f"Failed to generate original embedding: {str(e)}")

    # Enhanced variants with more comprehensive coverage
    # Rotation angles: 1-45° in smaller increments for better coverage
    rotation_angles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45]
    
    # Scale variations: 0.5-1.5 with more granular steps
    scales = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]
    
    # More noise levels for better robustness
    noise_vars = [2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60]
    
    # More blur variations including different kernel sizes
    blur_kernels = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    
    # Perspective deltas remain similar but with more variations
    perspective_deltas = [5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50]
    
    # More brightness/contrast combinations
    brightness_contrast = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (35, 35), (40, 40), (45, 45), (50, 50), (55, 55), (60, 60), (65, 65), (70, 70), (75, 75), (80, 80)]
    
    # More elastic transform parameters
    elastic_params = [(5, 1), (8, 2), (10, 2), (12, 3), (15, 3), (18, 4), (20, 4), (22, 5), (25, 5), (28, 6), (30, 6), (32, 7), (35, 7), (38, 8), (40, 8), (42, 9), (45, 9), (48, 10), (50, 10)]

    # Generate embeddings for all variants with error handling
    for i, angle in enumerate(rotation_angles):
        try:
            img = rotate_image(image, angle)
            results[f"rotated_{angle}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate rotated_{angle}_v{i+1}: {str(e)}")
            continue

    for i, scale in enumerate(scales):
        try:
            img = scale_image(image, scale)
            results[f"scaled_{scale}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate scaled_{scale}_v{i+1}: {str(e)}")
            continue

    for i, var in enumerate(noise_vars):
        try:
            img = add_gaussian_noise(image, var)
            results[f"gaussian_noise_{var}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate gaussian_noise_{var}_v{i+1}: {str(e)}")
            continue

    for i, k in enumerate(blur_kernels):
        try:
            img = blur_image(image, k)
            results[f"blurred_k{k}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate blurred_k{k}_v{i+1}: {str(e)}")
            continue

    for i, delta in enumerate(perspective_deltas):
        try:
            img = perspective_warp(image, delta)
            results[f"perspective_{delta}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate perspective_{delta}_v{i+1}: {str(e)}")
            continue

    for i, (b, c) in enumerate(brightness_contrast):
        try:
            img = adjust_brightness_contrast(image, b, c)
            results[f"brightness_contrast_{b}_{c}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate brightness_contrast_{b}_{c}_v{i+1}: {str(e)}")
            continue

    for i, (alpha, sigma) in enumerate(elastic_params):
        try:
            img = elastic_transform(image, alpha, sigma)
            results[f"elastic_{alpha}_{sigma}_v{i+1}"] = get_embedding(img, use_orb).tolist()
        except Exception as e:
            print(f"Warning: Failed to generate elastic_{alpha}_{sigma}_v{i+1}: {str(e)}")
            continue

    return results


if __name__ == "__main__":
    image_path = r"C:\Users\hp\Pictures\Capture.PNG"
    print(process_fingerprint_for_django(image_path,True))