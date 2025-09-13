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
    id = models.AutoField(primary_key=True, db_index=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=32, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    source = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="Context/source, e.g. security, healthcare, visitor, etc.",
    )

    notes = models.TextField(null=True, blank=True)
    img = models.ImageField(upload_to="fingerprints/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # Remove is_known logic since it may not exist
        return f"Person (ID: {self.id})"


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
    # if not created or not instance.img:
    #     return

    image_path = instance.img.path
    if not os.path.exists(image_path):
        return
    try:
        # Always generate distorted embeddings and rebuild index for any new fingerprint
        from apps.core.tasks import generate_distorted_embeddings_for_fingerprint
        generate_distorted_embeddings_for_fingerprint.delay(instance.id)
        from .annoy_index import build_annoy_index

        build_annoy_index()

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to generate embeddings for fingerprint {instance.id}: {str(e)}")
