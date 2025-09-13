import os
import datetime
from django.db import models
from mongoengine import (
    Document,
    StringField,
    IntField,
    FloatField,
    ListField,
    DateTimeField,
)
from django.db.models.signals import post_save
from django.dispatch import receiver

from .utils import extras


class FingerPrint(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    img = models.ImageField(upload_to="fingerprints/", null=True, blank=True)
    is_known = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        if self.is_known and self.name:
            return f"Known: {self.name}"
        return f"Unknown Person (ID: {self.id})"


class FingerPrintEmbed(Document):
    fingerprint_id = IntField(required=True)  # Link to Django model ID
    variant_name = StringField(required=True)  # e.g., 'rotated_15_v1'
    embedding = ListField(FloatField(), required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    meta = {"collection": "fingerprint_embeddings", "indexes": ["fingerprint_id", "variant_name"]}

    def __str__(self):
        return f"Embedding {self.variant_name} for FingerPrint ID {self.fingerprint_id}"


@receiver(post_save, sender=FingerPrint)
def generate_and_save_embeddings(sender, instance, created, **kwargs):
    if not created or not instance.img:
        return

    image_path = instance.img.path
    if not os.path.exists(image_path):
        return

    try:
        # Only generate the "original" embedding using the direct function
        image = extras._load_and_validate_image(image_path)
        embedding_result, _ = extras._generate_original_embedding(image)
        # embedding_result is a dict like {"original": [vector]}
        for variant_name, vector in embedding_result.items():
            FingerPrintEmbed.objects.create(
                fingerprint_id=instance.id,
                variant_name=variant_name,
                embedding=vector,
            )

        # Rebuild index only if this is a known fingerprint
        if instance.is_known:
            from apps.core.tasks import generate_distorted_embeddings_for_fingerprint
            generate_distorted_embeddings_for_fingerprint.delay(instance.id)
            from .annoy_index import build_annoy_index

            build_annoy_index()

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to generate embeddings for fingerprint {instance.id}: {str(e)}")
