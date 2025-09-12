from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
import cv2 as cv
from .models import FingerPrint, FingerPrintEmbed
from .extras import get_embedding, process_fingerprint_for_django
from .annoy_index import search_annoy_index, build_annoy_index
import base64
from django.core.files.base import ContentFile
import tempfile
import os
import json
import logging

logger = logging.getLogger(__name__)


class FingerPrintRegisterView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request):
        img_base64 = request.data.get('img')
        name = request.data.get('name')
        is_known = request.data.get('is_known', 'false').lower() == 'true'

        if not img_base64:
            return Response({"error": "Fingerprint image is required."}, status=400)

        # Convert base64 to image file
        try:
            # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            if ',' in img_base64:
                img_base64 = img_base64.split(',')[1]
            
            # Decode base64 string
            img_data = base64.b64decode(img_base64)
            img_file = ContentFile(img_data, name='fingerprint.pmb')
            
        except Exception as e:
            return Response({"error": "Invalid base64 image data."}, status=400)

        fingerprint = FingerPrint.objects.create(
            name=name,
            img=img_file,
            is_known=is_known
        )

        return Response({
            "message": "Fingerprint registered successfully.",
            "id": fingerprint.id,
            "name": fingerprint.name,
            "is_known": fingerprint.is_known
        }, status=201)

class FingerPrintDetectView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request):
        img = request.FILES.get('img')
        if not img:
            return Response({"error": "Fingerprint image is required."}, status=400)

        print(f"DEBUG: Received image file: {img.name}, size: {img.size} bytes")

        # Save temporarily to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in img.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        print(f"DEBUG: Saved temporary image to: {temp_path}")

        try:
            # Read image using OpenCV
            image = cv.imread(temp_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                print(f"DEBUG: Failed to read image from {temp_path}")
                return Response({"error": "Invalid image."}, status=400)

            print(f"DEBUG: Successfully loaded image with shape: {image.shape}")

            # Generate a single embedding vector
            embedding = get_embedding(image, use_orb=True)
            print(f"DEBUG: Generated embedding with length: {len(embedding) if embedding is not None else 'None'}")
            
            if embedding is None:
                print("DEBUG: Failed to generate embedding")
                return Response({"error": "Failed to generate fingerprint embedding."}, status=400)
            
            query_vectors = [embedding]  # Annoy expects a list of vectors
            print(f"DEBUG: Query vectors prepared: {len(query_vectors)} vector(s)")

        finally:
            os.remove(temp_path)
            print(f"DEBUG: Cleaned up temporary file: {temp_path}")

        # Search using Annoy
        print("DEBUG: Starting Annoy index search...")
        match_id, certainty = search_annoy_index(query_vectors, top_k=5)
        print(f"DEBUG: Search result - Match ID: {match_id}, Certainty: {certainty}")

        # Check if we have any fingerprints in the database
        total_fingerprints = FingerPrint.objects.count()
        total_embeddings = FingerPrintEmbed.objects.count()
        print(f"DEBUG: Database contains {total_fingerprints} fingerprints and {total_embeddings} embeddings")

        if match_id and certainty > 0.7:
            print(f"DEBUG: Found potential match with ID {match_id} and certainty {certainty}")
            try:
                matched_fp = FingerPrint.objects.get(id=match_id)
                print(f"DEBUG: Successfully retrieved matched fingerprint: {matched_fp}")
                return Response({
                    "match": True,
                    "certainty": certainty,
                    "name": matched_fp.name,
                    "id": matched_fp.id,
                    "debug_info": {
                        "total_fingerprints": total_fingerprints,
                        "total_embeddings": total_embeddings,
                        "embedding_length": len(embedding)
                    }
                })
            except FingerPrint.DoesNotExist:
                print(f"DEBUG: Fingerprint with ID {match_id} not found in database")
                return Response({
                    "match": False,
                    "certainty": certainty,
                    "error": "Matched fingerprint not found in database",
                    "debug_info": {
                        "total_fingerprints": total_fingerprints,
                        "total_embeddings": total_embeddings,
                        "embedding_length": len(embedding)
                    }
                }, status=400)  # Changed from 200 to 400 for proper error handling

        print(f"DEBUG: No match found or certainty too low. Match ID: {match_id}, Certainty: {certainty}")
        return Response({
            "match": False,
            "certainty": certainty,
            "debug_info": {
                "total_fingerprints": total_fingerprints,
                "total_embeddings": total_embeddings,
                "embedding_length": len(embedding),
                "threshold": 0.7,
                "query_vectors_count": len(query_vectors),
                "vector_dimensions": len(embedding) if embedding is not None else 0
            }
        }, status=200)

class FingerPrintListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        """
        Return all saved fingerprints with their embeddings
        """
        try:
            # Get all fingerprints
            fingerprints = FingerPrint.objects.all()
            
            result = []
            for fp in fingerprints:
                # Get all embeddings for this fingerprint
                embeddings = FingerPrintEmbed.objects.filter(fingerprint_id=fp.id)
                
                fingerprint_data = {
                    "id": fp.id,
                    "name": fp.name,
                    "is_known": fp.is_known,
                    "created_at": fp.created_at,
                    "updated_at": fp.updated_at,
                    "image_url": fp.img.url if fp.img else None,
                    "embeddings": []
                }
                
                # Add embedding data
                for embedding in embeddings:
                    fingerprint_data["embeddings"].append({
                        "variant_name": embedding.variant_name,
                        "embedding_vector": embedding.embedding,
                        "created_at": embedding.created_at
                    })
                
                result.append(fingerprint_data)
            
            return Response({
                "fingerprints": result,
                "total_count": len(result)
            }, status=200)
            
        except Exception as e:
            return Response({
                "error": f"Failed to retrieve fingerprints: {str(e)}"
            }, status=500)

class FingerPrintDetailView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, fingerprint_id):
        """
        Return a specific fingerprint with its embeddings
        """
        try:
            # Get the specific fingerprint
            fingerprint = FingerPrint.objects.get(id=fingerprint_id)
            
            # Get all embeddings for this fingerprint
            embeddings = FingerPrintEmbed.objects.filter(fingerprint_id=fingerprint_id)
            
            fingerprint_data = {
                "id": fingerprint.id,
                "name": fingerprint.name,
                "is_known": fingerprint.is_known,
                "created_at": fingerprint.created_at,
                "updated_at": fingerprint.updated_at,
                "image_url": fingerprint.img.url if fingerprint.img else None,
                "embeddings": []
            }
            
            # Add embedding data
            for embedding in embeddings:
                fingerprint_data["embeddings"].append({
                    "variant_name": embedding.variant_name,
                    "embedding_vector": embedding.embedding,
                    "created_at": embedding.created_at
                })
            
            return Response(fingerprint_data, status=200)
            
        except FingerPrint.DoesNotExist:
            return Response({
                "error": "Fingerprint not found"
            }, status=404)
        except Exception as e:
            return Response({
                "error": f"Failed to retrieve fingerprint: {str(e)}"
            }, status=500)


class FingerPrintRegenerateEmbeddingsView(APIView):
    """
    API endpoint to delete old embeddings and regenerate new ones for fingerprints.
    Can regenerate embeddings for all fingerprints or specific ones.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        """
        Regenerate embeddings for fingerprints.
        
        Request body options:
        - fingerprint_ids: List of specific fingerprint IDs to regenerate (optional)
        - rebuild_index: Boolean to rebuild Annoy index after regeneration (default: True)
        - use_orb: Boolean to use ORB features (default: True)
        """
        try:
            # Handle both JSON and form data
            if hasattr(request, 'data') and request.data:
                data = request.data
            else:
                # Try to parse JSON from request body
                try:
                    import json
                    data = json.loads(request.body.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = {}
            
            fingerprint_ids = data.get('fingerprint_ids', [])
            rebuild_index = data.get('rebuild_index', True)
            use_orb = data.get('use_orb', True)
            
            # If no specific IDs provided, regenerate all fingerprints
            if not fingerprint_ids:
                fingerprints = FingerPrint.objects.filter(img__isnull=False).exclude(img='')
            else:
                fingerprints = FingerPrint.objects.filter(
                    id__in=fingerprint_ids, 
                    img__isnull=False
                ).exclude(img='')
            
            if not fingerprints.exists():
                return Response({
                    "error": "No fingerprints found with valid images to regenerate"
                }, status=404)
            
            results = {
                "processed": [],
                "errors": [],
                "total_processed": 0,
                "total_errors": 0
            }
            
            for fingerprint in fingerprints:
                try:
                    # Delete old embeddings for this fingerprint
                    deleted_count = FingerPrintEmbed.objects.filter(
                        fingerprint_id=fingerprint.id
                    ).delete()[0]
                    
                    logger.info(f"Deleted {deleted_count} old embeddings for fingerprint {fingerprint.id}")
                    
                    # Check if image file exists
                    if not fingerprint.img or not os.path.exists(fingerprint.img.path):
                        results["errors"].append({
                            "fingerprint_id": fingerprint.id,
                            "error": "Image file not found"
                        })
                        results["total_errors"] += 1
                        continue
                    
                    # Generate new embeddings
                    logger.info(f"Generating embeddings for fingerprint {fingerprint.id} at path: {fingerprint.img.path}")
                    embeddings = process_fingerprint_for_django(
                        fingerprint.img.path, 
                        use_orb=use_orb
                    )
                    logger.info(f"Generated {len(embeddings)} embedding variants for fingerprint {fingerprint.id}")
                    
                    # Save new embeddings
                    new_embeddings_count = 0
                    for variant_name, vector in embeddings.items():
                        # Validate embedding vector
                        if not isinstance(vector, list) or len(vector) == 0:
                            logger.warning(f"Invalid embedding vector for {variant_name}: {type(vector)}")
                            continue
                        
                        try:
                            FingerPrintEmbed.objects.create(
                                fingerprint_id=fingerprint.id,
                                variant_name=variant_name,
                                embedding=vector
                            )
                            new_embeddings_count += 1
                        except Exception as e:
                            logger.error(f"Failed to save embedding {variant_name} for fingerprint {fingerprint.id}: {str(e)}")
                            continue
                    
                    results["processed"].append({
                        "fingerprint_id": fingerprint.id,
                        "name": fingerprint.name,
                        "is_known": fingerprint.is_known,
                        "deleted_embeddings": deleted_count,
                        "new_embeddings": new_embeddings_count
                    })
                    results["total_processed"] += 1
                    
                    logger.info(f"Successfully regenerated {new_embeddings_count} embeddings for fingerprint {fingerprint.id}")
                    
                except Exception as e:
                    error_msg = f"Failed to regenerate embeddings for fingerprint {fingerprint.id}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append({
                        "fingerprint_id": fingerprint.id,
                        "error": error_msg
                    })
                    results["total_errors"] += 1
            
            # Rebuild Annoy index if requested
            if rebuild_index and results["total_processed"] > 0:
                try:
                    build_annoy_index()
                    results["index_rebuilt"] = True
                    logger.info("Annoy index rebuilt successfully")
                except Exception as e:
                    error_msg = f"Failed to rebuild Annoy index: {str(e)}"
                    logger.error(error_msg)
                    results["index_rebuild_error"] = error_msg
                    results["index_rebuilt"] = False
            
            return Response({
                "message": f"Embedding regeneration completed. Processed: {results['total_processed']}, Errors: {results['total_errors']}",
                "results": results
            }, status=200)
            
        except Exception as e:
            logger.error(f"Failed to regenerate embeddings: {str(e)}")
            return Response({
                "error": f"Failed to regenerate embeddings: {str(e)}"
            }, status=500)

    def delete(self, request):
        """
        Delete all embeddings for specific fingerprints or all fingerprints.
        
        Request body options:
        - fingerprint_ids: List of specific fingerprint IDs to delete embeddings for (optional)
        - delete_all: Boolean to delete all embeddings (default: False)
        """
        try:
            # Handle both JSON and form data
            if hasattr(request, 'data') and request.data:
                data = request.data
            else:
                # Try to parse JSON from request body
                try:
                    import json
                    data = json.loads(request.body.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = {}
            
            fingerprint_ids = data.get('fingerprint_ids', [])
            delete_all = data.get('delete_all', False)
            
            if delete_all:
                # Delete all embeddings
                deleted_count = FingerPrintEmbed.objects.all().delete()[0]
                message = f"Deleted all {deleted_count} embeddings"
            elif fingerprint_ids:
                # Delete embeddings for specific fingerprints
                deleted_count = FingerPrintEmbed.objects.filter(
                    fingerprint_id__in=fingerprint_ids
                ).delete()[0]
                message = f"Deleted {deleted_count} embeddings for {len(fingerprint_ids)} fingerprints"
            else:
                return Response({
                    "error": "Either provide fingerprint_ids or set delete_all=True"
                }, status=400)
            
            # Rebuild index after deletion
            try:
                build_annoy_index()
                index_rebuilt = True
            except Exception as e:
                logger.error(f"Failed to rebuild index after deletion: {str(e)}")
                index_rebuilt = False
            
            return Response({
                "message": message,
                "deleted_count": deleted_count,
                "index_rebuilt": index_rebuilt
            }, status=200)
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {str(e)}")
            return Response({
                "error": f"Failed to delete embeddings: {str(e)}"
            }, status=500)