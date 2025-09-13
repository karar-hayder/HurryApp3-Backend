from django.core.management.base import BaseCommand
import numpy as np
import cv2
from apps.core.extras import get_embedding, process_fingerprint_for_django
import tempfile
import os


class Command(BaseCommand):
    help = "Test the embedding generation fix"

    def handle(self, *args, **options):
        self.stdout.write("Testing embedding generation fix...")

        # Test 1: Normal image with minutiae features
        self.stdout.write("\n1. Testing normal image with minutiae features...")
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        try:
            embedding = get_embedding(test_image, method="minutiae", is_augmented=True)
            self.stdout.write(
                self.style.SUCCESS(f"   ✓ Success: Generated embedding of length {len(embedding)}")
            )
            self.stdout.write(f"   ✓ Embedding type: {type(embedding)}")
            self.stdout.write(f"   ✓ Embedding dtype: {embedding.dtype}")
            assert len(embedding) == 128, f"Expected length 128, got {len(embedding)}"
            assert isinstance(embedding, np.ndarray), f"Expected numpy array, got {type(embedding)}"
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Error: {str(e)}"))
            return

        # Test 2: Image with no features (uniform color)
        self.stdout.write("\n2. Testing uniform color image (no features)...")
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        try:
            embedding = get_embedding(uniform_image, method="minutiae", is_augmented=True)
            self.stdout.write(
                self.style.SUCCESS(f"   ✓ Success: Generated embedding of length {len(embedding)}")
            )
            assert len(embedding) == 128, f"Expected length 128, got {len(embedding)}"
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Error: {str(e)}"))
            return

        # Test 3: Very small image
        self.stdout.write("\n3. Testing very small image...")
        small_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        try:
            embedding = get_embedding(small_image, method="minutiae", is_augmented=True)
            self.stdout.write(
                self.style.SUCCESS(f"   ✓ Success: Generated embedding of length {len(embedding)}")
            )
            assert len(embedding) == 128, f"Expected length 128, got {len(embedding)}"
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Error: {str(e)}"))
            return

        # Test 4: Full process_fingerprint_for_django
        self.stdout.write("\n4. Testing full process_fingerprint_for_django...")
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                cv2.imwrite(tmp_file_path, test_image)

            embeddings = process_fingerprint_for_django(
                tmp_file_path, method="minutiae", skip_complex_augmentations=True
            )
            self.stdout.write(
                self.style.SUCCESS(f"   ✓ Success: Generated {len(embeddings)} embedding variants")
            )

            # Check a few embeddings
            for i, (name, vector) in enumerate(embeddings.items()):
                if i < 3:  # Show first 3
                    self.stdout.write(f"   {name}: length {len(vector)}, type {type(vector)}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Error: {str(e)}"))
            return
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except PermissionError:
                    # File might still be in use, ignore the error
                    pass

        self.stdout.write(
            self.style.SUCCESS(
                "\n✓ All tests passed! The embedding generation fix is working correctly."
            )
        )
