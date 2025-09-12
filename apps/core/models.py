# models.py
import os
from django.db import models
from mongoengine import Document, StringField, IntField, FloatField, ListField, DateTimeField, ReferenceField
import datetime
from django.db.models.signals import post_save
from django.dispatch import receiver
from .extras import process_fingerprint_for_django  # Your function

class FingerPrint(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    img = models.ImageField(upload_to='fingerprints/', null=True, blank=True)
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

    meta = {
        'collection': 'fingerprint_embeddings',
        'indexes': [
            'fingerprint_id',
            'variant_name'
        ]
    }

    def __str__(self):
        return f"Embedding {self.variant_name} for FingerPrint ID {self.fingerprint_id}"


@receiver(post_save, sender=FingerPrint)
def generate_and_save_embeddings(sender, instance, created, **kwargs):
    if not created or not instance.img:
        return

    # Absolute file path to image
    image_path = instance.img.path
    if not os.path.exists(image_path):
        return

    embeddings = process_fingerprint_for_django(image_path, use_orb=True)
    for variant_name, vector in embeddings.items():
        FingerPrintEmbed.objects.create(
            fingerprint_id=instance.id,
            variant_name=variant_name,
            embedding=vector
        )

    # Rebuild index only if this is a known fingerprint
    if instance.is_known:
        from .annoy_index import build_annoy_index
        build_annoy_index()


def regenerate_embeddings_for_fingerprint(fingerprint_id, use_orb=True, rebuild_index=True):
    """
    Utility function to regenerate embeddings for a specific fingerprint.
    
    Args:
        fingerprint_id (int): ID of the fingerprint to regenerate embeddings for
        use_orb (bool): Whether to use ORB features (default: True)
        rebuild_index (bool): Whether to rebuild the Annoy index after regeneration (default: True)
    
    Returns:
        dict: Results of the regeneration process
    """
    from .annoy_index import build_annoy_index
    
    try:
        fingerprint = FingerPrint.objects.get(id=fingerprint_id)
        
        # Delete old embeddings
        deleted_count = FingerPrintEmbed.objects.filter(
            fingerprint_id=fingerprint_id
        ).delete()[0]
        
        # Check if image exists
        if not fingerprint.img or not os.path.exists(fingerprint.img.path):
            return {
                "success": False,
                "error": "Image file not found",
                "fingerprint_id": fingerprint_id
            }
        
        # Generate new embeddings
        embeddings = process_fingerprint_for_django(
            fingerprint.img.path, 
            use_orb=use_orb
        )
        
        # Save new embeddings
        new_embeddings_count = 0
        for variant_name, vector in embeddings.items():
            FingerPrintEmbed.objects.create(
                fingerprint_id=fingerprint_id,
                variant_name=variant_name,
                embedding=vector
            )
            new_embeddings_count += 1
        
        # Rebuild index if requested
        index_rebuilt = False
        if rebuild_index:
            try:
                build_annoy_index()
                index_rebuilt = True
            except Exception as e:
                return {
                    "success": True,
                    "warning": f"Embeddings regenerated but index rebuild failed: {str(e)}",
                    "fingerprint_id": fingerprint_id,
                    "deleted_embeddings": deleted_count,
                    "new_embeddings": new_embeddings_count,
                    "index_rebuilt": False
                }
        
        return {
            "success": True,
            "fingerprint_id": fingerprint_id,
            "deleted_embeddings": deleted_count,
            "new_embeddings": new_embeddings_count,
            "index_rebuilt": index_rebuilt
        }
        
    except FingerPrint.DoesNotExist:
        return {
            "success": False,
            "error": f"Fingerprint with ID {fingerprint_id} not found",
            "fingerprint_id": fingerprint_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to regenerate embeddings: {str(e)}",
            "fingerprint_id": fingerprint_id
        }


def regenerate_all_embeddings(use_orb=True, rebuild_index=True):
    """
    Utility function to regenerate embeddings for all fingerprints.
    
    Args:
        use_orb (bool): Whether to use ORB features (default: True)
        rebuild_index (bool): Whether to rebuild the Annoy index after regeneration (default: True)
    
    Returns:
        dict: Results of the regeneration process
    """
    fingerprints = FingerPrint.objects.filter(img__isnull=False).exclude(img='')
    
    results = {
        "processed": [],
        "errors": [],
        "total_processed": 0,
        "total_errors": 0
    }
    
    for fingerprint in fingerprints:
        result = regenerate_embeddings_for_fingerprint(
            fingerprint.id, 
            use_orb=use_orb, 
            rebuild_index=False  # We'll rebuild once at the end
        )
        
        if result["success"]:
            results["processed"].append(result)
            results["total_processed"] += 1
        else:
            results["errors"].append(result)
            results["total_errors"] += 1
    
    # Rebuild index once at the end if requested
    if rebuild_index and results["total_processed"] > 0:
        try:
            from .annoy_index import build_annoy_index
            build_annoy_index()
            results["index_rebuilt"] = True
        except Exception as e:
            results["index_rebuild_error"] = str(e)
            results["index_rebuilt"] = False
    
    return results