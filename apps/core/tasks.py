from celery import shared_task


@shared_task
def generate_distorted_embeddings_for_fingerprint(fingerprint_id):
    """
    Celery task to generate distorted embeddings for a given fingerprint.
    Loads the image, sends it to the remote server for distorted embeddings,
    and saves each embedding variant in the database.
    The server returns a dict: {filename: embedding_list, ...}
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

    if not fingerprint.img or not os.path.exists(fingerprint.img.path):
        logger.error(f"Image file for fingerprint {fingerprint_id} not found.")
        return

    try:
        # Load and validate image
        image = extras._load_and_validate_image(fingerprint.img.path)

        # Get distorted embeddings from remote server
        # The server returns: {"embeddings": {filename: embedding_list, ...}}
        embeddings_dict = extras.get_distorted_embeddings_from_server(image)
        if not isinstance(embeddings_dict, dict):
            logger.error(f"Distorted embeddings response is not a dict: {type(embeddings_dict)}")
            return

        # Save each embedding variant (key is filename, value is embedding list)
        saved_count = 0
        for fname, vector in embeddings_dict.items():
            # The server returns the filename as the key (e.g., "rotated_15.png")
            variant_name = os.path.splitext(fname)[0]  # Remove file extension for variant name
            if not isinstance(vector, list) or not vector:
                logger.warning(f"Invalid embedding vector for {variant_name}: {type(vector)}")
                continue
            try:
                FingerPrintEmbed.objects.create(
                    fingerprint_id=fingerprint.id,
                    variant_name=variant_name,
                    embedding=vector,
                )
                saved_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to save embedding {variant_name} for fingerprint {fingerprint.id}: {str(e)}"
                )
                continue

        logger.info(
            f"Generated and saved {saved_count} distorted embeddings for fingerprint {fingerprint.id}"
        )

    except Exception as e:
        logger.error(
            f"Failed to generate distorted embeddings for fingerprint {fingerprint_id}: {str(e)}"
        )
