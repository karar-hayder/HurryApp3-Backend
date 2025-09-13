from celery import shared_task

@shared_task
def generate_distorted_embeddings_for_fingerprint(fingerprint_id):
    """
    Hotfix: Try alternate image path if not found at default location.
    Only store the embeddings of distorted images, not the images themselves.
    """
    import os
    import logging

    from .models import FingerPrint, FingerPrintEmbed
    from .utils import extras

    logger = logging.getLogger(__name__)

    try:
        fingerprint = FingerPrint.objects.get(id=fingerprint_id)
    except FingerPrint.DoesNotExist:
        logger.error(f"Fingerprint with id {fingerprint_id} does not exist.")
        return

    image_path = fingerprint.img.path if fingerprint.img else None

    # HOTFIX: If not found, try to look for the image in a nested 'fingerprints/fingerprints/' directory
    if not image_path or not os.path.exists(image_path):
        logger.warning(f"Image file for fingerprint {fingerprint_id} not found at {image_path}. Trying hotfix path...")
        if image_path:
            # Try to replace the first occurrence of 'fingerprints/' with 'fingerprints/fingerprints/'
            # Only if not already double
            if "fingerprints/" in image_path and "fingerprints/fingerprints/" not in image_path:
                alt_path = image_path.replace("fingerprints/", "fingerprints/fingerprints/", 1)
                if os.path.exists(alt_path):
                    logger.info(f"Hotfix: Found image for fingerprint {fingerprint_id} at {alt_path}")
                    image_path = alt_path
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Image file for fingerprint {fingerprint_id} not found (even after hotfix).")
            return

    try:
        # --- Enhance the image and replace the original file with the enhanced version ---
        import cv2
        import numpy as np

        # Load image as grayscale
        img_np = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            logger.error(f"Failed to load image for fingerprint {fingerprint_id} at {image_path}")
            return

        enhanced = extras.restore_fingerprint(img_np)
        if isinstance(enhanced, list):
            enhanced_np = np.array(enhanced, dtype=np.uint8)
        elif isinstance(enhanced, str):
            import base64 as _base64
            img_bytes = _base64.b64decode(enhanced)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            enhanced_np = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        else:
            enhanced_np = enhanced

        # Save the enhanced image, replacing the original file
        success, buffer = cv2.imencode(".bmp", enhanced_np)
        if not success:
            logger.error(f"Failed to encode enhanced image for fingerprint {fingerprint_id}")
            return
        with open(image_path, "wb") as f:
            f.write(buffer.tobytes())
        logger.info(f"Enhanced image saved for fingerprint {fingerprint_id} at {image_path}")

        # Generate and save the "original" embedding (now from enhanced image)
        image = extras._load_and_validate_image(image_path)
        embedding_result, _ = extras._generate_original_embedding(image)
        for variant_name, vector in embedding_result.items():
            try:
                FingerPrintEmbed.objects.create(
                    fingerprint_id=fingerprint.id,
                    variant_name=variant_name,
                    embedding=vector,
                )
                logger.info(
                    f"Saved embedding '{variant_name}' for fingerprint {fingerprint.id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save embedding {variant_name} for fingerprint {fingerprint.id}: {str(e)}"
                )
                continue

        # Now generate and save distorted embeddings (variants) -- only store their embeddings, not the images
        # try:
        #     embeddings_dict = extras.get_distorted_embeddings_from_server(image)
        # except Exception as e:
        #     logger.error(f"Failed to get distorted embeddings from server: {e}")
        #     return

        # if not isinstance(embeddings_dict, dict):
        #     logger.error(f"Distorted embeddings response is not a dict: {type(embeddings_dict)}")
        #     return

        # for variant_name, vector in embeddings_dict.items():
        #     if not isinstance(vector, list) or not vector:
        #         logger.warning(f"Invalid embedding vector for {variant_name}: {type(vector)}")
        #         continue
        #     try:
        #         # Only store the embedding, not the distorted image itself
        #         FingerPrintEmbed.objects.create(
        #             fingerprint_id=fingerprint.id,
        #             variant_name=variant_name,
        #             embedding=vector,
        #         )
        #         logger.info(
        #             f"Saved distorted embedding '{variant_name}' for fingerprint {fingerprint.id}"
        #         )
        #     except Exception as e:
        #         logger.error(
        #             f"Failed to save distorted embedding {variant_name} for fingerprint {fingerprint.id}: {str(e)}"
        #         )
        #         continue

        # --- ADDITION: Rebuild Annoy index after embeddings are saved ---
        try:
            from .annoy_index import build_annoy_index
            build_annoy_index()
            logger.info(f"Annoy index rebuilt after embeddings for fingerprint {fingerprint.id}")
        except Exception as e:
            logger.error(f"Failed to rebuild Annoy index after embeddings for fingerprint {fingerprint.id}: {str(e)}")

    except Exception as e:
        logger.error(
            f"Failed to generate embeddings for fingerprint {fingerprint_id}: {str(e)}"
        )
