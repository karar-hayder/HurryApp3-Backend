from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import AllowAny
import cv2 as cv
from .models import FingerPrint, FingerPrintEmbed

from .annoy_index import search_annoy_index, build_annoy_index
import base64
from django.core.files.base import ContentFile
import tempfile
import os
import json

import pprint

class FingerPrintRegisterView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        fingerprint_fields = [
            "name", "img", "age", "gender", "notes", "country", "city", "address", "phone", "email"
        ]
        fingerprint_data = {}
        for field in fingerprint_fields:
            if field in request.data:
                fingerprint_data[field] = request.data.get(field)

        img_file = None

        if "img" in request.FILES:
            uploaded_file = request.FILES["img"]
            allowed_extensions = [".bmp", ".jpg", ".jpeg", ".png"]
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension not in allowed_extensions:
                return Response(
                    {
                        "error": f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
                    },
                    status=400,
                )

            if uploaded_file.size > 10 * 1024 * 1024:
                return Response({"error": "File size too large. Maximum size is 10MB."}, status=400)

            # Just save the uploaded file as is, let background task handle enhancement/embedding
            fingerprint_data["img"] = uploaded_file

        elif "img" in request.data:
            img_base64 = request.data.get("img")

            if not img_base64:
                return Response({"error": "Fingerprint image is required."}, status=400)

            try:
                if "," in img_base64:
                    img_base64 = img_base64.split(",")[1]
                img_data = base64.b64decode(img_base64)
                img_file = ContentFile(img_data, name="fingerprint.bmp")
                fingerprint_data["img"] = img_file
            except Exception as e:
                return Response({"error": f"Invalid base64 image data: {str(e)}"}, status=400)
        else:
            return Response(
                {"error": "Fingerprint image is required (as file upload or base64 data)."},
                status=400,
            )

        try:
            fingerprint = FingerPrint.objects.create(**fingerprint_data)
        except Exception as e:
            return Response({"error": f"Failed to register fingerprint: {str(e)}"}, status=500)

        # Return all fields of the FingerPrint model
        response_data = {}
        for field in fingerprint._meta.fields:
            value = getattr(fingerprint, field.name)
            if field.name == "img":
                value = fingerprint.img.url if fingerprint.img else None
            response_data[field.name] = value

        response_data["message"] = "Fingerprint registered successfully."

        return Response(
            response_data,
            status=201,
        )

from .utils import extras

import time

class FingerPrintDetectView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        import time
        start_time = time.perf_counter()
        timings = {}

        print(f"Request to {request.path} with method {request.method}")
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request data: {pprint.pformat(request.data)}")
        print(f"Request FILES: {request.FILES}")

        data = request.data if hasattr(request, "data") else {}

        img_file = None
        t0 = time.perf_counter()
        if "img" in request.FILES:
            uploaded_file = request.FILES["img"]
            allowed_extensions = [".bmp", ".jpg", ".jpeg", ".png"]
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            print(f"Uploaded file extension: {file_extension}")
            if file_extension not in allowed_extensions:
                print("Rejected file due to unsupported extension")
                return Response(
                    {
                        "error": f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
                    },
                    status=400,
                )
            if uploaded_file.size > 10 * 1024 * 1024:
                print("Rejected file due to size > 10MB")
                return Response({"error": "File size too large. Maximum size is 10MB."}, status=400)
            img_file = uploaded_file
        elif "img" in data:
            img_base64 = data.get("img")
            print(f"Received base64 image: {str(img_base64)[:50]}...")
            if not img_base64:
                print("No base64 image data provided")
                return Response({"error": "Fingerprint image is required."}, status=400)
            try:
                if "," in img_base64:
                    img_base64 = img_base64.split(",", 1)[1]
                import base64 as _base64
                try:
                    img_data = _base64.b64decode(img_base64)
                except (_base64.binascii.Error, ValueError) as e:
                    print(f"Invalid base64 image data: {e}")
                    raise ValueError("Invalid base64 image data") from e
                img_file = ContentFile(img_data, name="fingerprint.bmp")
            except Exception as e:
                print(f"Invalid base64 image data: {e}")
                return Response({"error": "Invalid base64 image data."}, status=400)
        else:
            print("No image provided in request")
            return Response(
                {"error": "Fingerprint image is required (as file upload or base64 data)."},
                status=400,
            )
        t1 = time.perf_counter()
        timings["input_parsing"] = t1 - t0

        file_extension = os.path.splitext(img_file.name)[1].lower()
        t2 = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            for chunk in img_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        t3 = time.perf_counter()
        timings["tempfile_write"] = t3 - t2

        print(f"Temporary file saved at: {temp_path}")

        try:
            t4 = time.perf_counter()
            try:
                raw_image = extras._load_and_validate_image(temp_path)
                print(f"Loaded raw image from {temp_path} with shape: {getattr(raw_image, 'shape', None)}")
            except Exception as e:
                print(f"Invalid image: {e}")
                return Response({"error": f"Invalid image: {str(e)}"}, status=400)
            t5 = time.perf_counter()
            timings["load_and_validate_image"] = t5 - t4

            t6 = time.perf_counter()
            try:
                raw_embedding_result, _ = extras._generate_original_embedding(raw_image)
                raw_embedding = raw_embedding_result.get("original")
                print(f"Generated raw embedding: {raw_embedding[:10]}... (length: {len(raw_embedding) if raw_embedding is not None else 0})")
            except Exception as e:
                print(f"Failed to generate raw fingerprint embedding: {e}")
                return Response(
                    {"error": f"Failed to generate fingerprint embedding (raw): {str(e)}"}, status=400
                )
            t7 = time.perf_counter()
            timings["raw_embedding"] = t7 - t6

            t8 = time.perf_counter()
            try:
                restored_image = extras.restore_fingerprint(raw_image)
                print(f"Restored image type: {type(restored_image)}")
                if isinstance(restored_image, list):
                    import numpy as np
                    enhanced_image = np.array(restored_image, dtype=np.uint8)
                    print(f"Restored image converted from list to np.ndarray with shape: {enhanced_image.shape}")
                elif isinstance(restored_image, str):
                    import base64 as _base64
                    import cv2
                    import numpy as np
                    img_bytes = _base64.b64decode(restored_image)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    enhanced_image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    print(f"Restored image decoded from base64 to np.ndarray with shape: {enhanced_image.shape}")
                else:
                    enhanced_image = restored_image
                    print(f"Restored image used as is, type: {type(enhanced_image)}")
            except Exception as e:
                print(f"Failed to restore fingerprint image: {e}")
                return Response({"error": f"Failed to restore fingerprint image: {str(e)}"}, status=400)
            t9 = time.perf_counter()
            timings["restore_fingerprint"] = t9 - t8

            t10 = time.perf_counter()
            try:
                enhanced_embedding_result, _ = extras._generate_original_embedding(enhanced_image)
                enhanced_embedding = enhanced_embedding_result.get("original")
                print(f"Generated enhanced embedding: {enhanced_embedding[:10]}... (length: {len(enhanced_embedding) if enhanced_embedding is not None else 0})")
            except Exception as e:
                print(f"Failed to generate enhanced fingerprint embedding: {e}")
                return Response(
                    {"error": f"Failed to generate fingerprint embedding (enhanced): {str(e)}"}, status=400
                )
            t11 = time.perf_counter()
            timings["enhanced_embedding"] = t11 - t10

            if raw_embedding is None and enhanced_embedding is None:
                print("Both raw and enhanced embeddings are None after generation")
                return Response({"error": "Failed to generate fingerprint embeddings."}, status=400)

            query_vectors = []
            if raw_embedding is not None:
                query_vectors.append(raw_embedding)
            if enhanced_embedding is not None:
                query_vectors.append(enhanced_embedding)

        finally:
            t12 = time.perf_counter()
            try:
                os.remove(temp_path)
                print(f"Temporary file {temp_path} deleted")
            except PermissionError as e:
                if hasattr(e, "winerror") and e.winerror == 32:
                    print(f"PermissionError (WinError 32) when deleting temp file {temp_path}: {e}. Skipping.")
                else:
                    print(f"Error deleting temp file {temp_path}: {e}")
            t13 = time.perf_counter()
            timings["tempfile_delete"] = t13 - t12

        from collections import defaultdict

        t14 = time.perf_counter()
        match_candidates = []
        for idx, embedding in enumerate(query_vectors):
            from .annoy_index import annoy_indexes, annoy_id_to_fingerprint_id

            for metric_name, index in annoy_indexes.items():
                try:
                    idxs, dists = index.get_nns_by_vector(
                        embedding, 3, include_distances=True
                    )
                    for i, annoy_id in enumerate(idxs):
                        fp_id = annoy_id_to_fingerprint_id.get(annoy_id)
                        if fp_id is None:
                            continue
                        if metric_name == 'angular':
                            similarity = 1 - dists[i] / 2
                        elif metric_name == 'euclidean':
                            similarity = 1 / (1 + dists[i])
                        else:
                            similarity = 1 / (1 + dists[i])
                        match_candidates.append({
                            "fingerprint_id": fp_id,
                            "certainty": similarity,
                            "matched_on": "raw" if idx == 0 else "enhanced",
                            "metric": metric_name,
                        })
                except Exception as e:
                    print(f"Error searching {metric_name} index: {e}")
                    continue
        t15 = time.perf_counter()
        timings["annoy_search"] = t15 - t14

        t16 = time.perf_counter()
        best_by_id = {}
        for cand in match_candidates:
            fp_id = cand["fingerprint_id"]
            if fp_id not in best_by_id or cand["certainty"] > best_by_id[fp_id]["certainty"]:
                best_by_id[fp_id] = cand

        person_best = {}
        for fp_id, cand in best_by_id.items():
            try:
                fp = FingerPrint.objects.get(id=fp_id)
                person_key = fp.name if hasattr(fp, "name") and fp.name is not None else f"__fpid_{fp_id}"
            except FingerPrint.DoesNotExist:
                person_key = f"__fpid_{fp_id}"
            if person_key not in person_best or cand["certainty"] > person_best[person_key]["certainty"]:
                person_best[person_key] = dict(cand)
                person_best[person_key]["_fp_id"] = fp_id

        best_matches = sorted(person_best.values(), key=lambda x: x["certainty"], reverse=True)[:3]
        t17 = time.perf_counter()
        timings["aggregate_and_sort"] = t17 - t16

        total_fingerprints = FingerPrint.objects.count()
        total_embeddings = FingerPrintEmbed.objects.count()
        print(f"Total fingerprints: {total_fingerprints}, total embeddings: {total_embeddings}")

        t18 = time.perf_counter()
        response_matches = []
        best_match_image_url = None

        for idx, match in enumerate(best_matches):
            fp_id = match.get("_fp_id", match.get("fingerprint_id"))
            try:
                fp = FingerPrint.objects.get(id=fp_id)
                # Updated: Return all fields of the FingerPrint model for each match
                match_data = {}
                for field in fp._meta.fields:
                    value = getattr(fp, field.name)
                    if field.name == "img":
                        value = fp.img.url if fp.img else None
                    match_data[field.name] = value
                # Add the source of the fingerprint explicitly
                match_data["source"] = getattr(fp, "source", None)
                match_data["certainty"] = match["certainty"]
                match_data["matched_on"] = match["matched_on"]
                match_data["metric"] = match["metric"]
                if idx == 0:
                    best_match_image_url = fp.img.url if fp.img else None
                response_matches.append(match_data)
            except FingerPrint.DoesNotExist:
                response_matches.append({
                    "id": fp_id,
                    "certainty": match["certainty"],
                    "matched_on": match["matched_on"],
                    "metric": match["metric"],
                    "error": "Matched fingerprint not found in database"
                })
        t19 = time.perf_counter()
        timings["prepare_response_matches"] = t19 - t18

        threshold = 0.3  # Increased threshold for >50% accuracy
        has_good_match = any(m.get("certainty", 0) > threshold for m in response_matches)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        timings["total"] = total_time

        response_data = {
            "match": has_good_match,
            "top_matches": response_matches,
            "threshold": threshold,
            "debug_info": {
                "total_fingerprints": total_fingerprints,
                "total_embeddings": total_embeddings,
                "embedding_length_raw": len(raw_embedding) if 'raw_embedding' in locals() and raw_embedding is not None else 0,
                "embedding_length_enhanced": len(enhanced_embedding) if 'enhanced_embedding' in locals() and enhanced_embedding is not None else 0,
                "query_vectors_count": len(query_vectors) if 'query_vectors' in locals() else 0,
                "vector_dimensions_raw": len(raw_embedding) if 'raw_embedding' in locals() and raw_embedding is not None else 0,
                "vector_dimensions_enhanced": len(enhanced_embedding) if 'enhanced_embedding' in locals() and enhanced_embedding is not None else 0,
                "timings_seconds": timings,
            },
        }
        if best_match_image_url:
            response_data["best_match_image_url"] = best_match_image_url

        print(f"Timings (seconds): {timings}")

        return Response(
            response_data,
            status=200,
        )

class FingerPrintListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        """
        Return all saved fingerprints with their embeddings
        """
        try:
            fingerprints = FingerPrint.objects.all()
            result = []
            for fp in fingerprints:
                embeddings = FingerPrintEmbed.objects.filter(fingerprint_id=fp.id)
                # Updated: Return all fields of the FingerPrint model
                fingerprint_data = {}
                for field in fp._meta.fields:
                    value = getattr(fp, field.name)
                    if field.name == "img":
                        value = fp.img.url if fp.img else None
                    fingerprint_data[field.name] = value
                fingerprint_data["embeddings"] = []
                for embedding in embeddings:
                    fingerprint_data["embeddings"].append(
                        {
                            "variant_name": embedding.variant_name,
                            "embedding_vector": embedding.embedding,
                            "created_at": embedding.created_at,
                        }
                    )
                result.append(fingerprint_data)
            return Response({"fingerprints": result, "total_count": len(result)}, status=200)
        except Exception as e:
            return Response({"error": f"Failed to retrieve fingerprints: {str(e)}"}, status=500)

class FingerPrintDetailView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, fingerprint_id):
        """
        Return a specific fingerprint with its embeddings
        """
        try:
            fingerprint = FingerPrint.objects.get(id=fingerprint_id)
            embeddings = FingerPrintEmbed.objects.filter(fingerprint_id=fingerprint_id)
            # Updated: Return all fields of the FingerPrint model
            fingerprint_data = {}
            for field in fingerprint._meta.fields:
                value = getattr(fingerprint, field.name)
                if field.name == "img":
                    value = fingerprint.img.url if fingerprint.img else None
                fingerprint_data[field.name] = value
            fingerprint_data["embeddings"] = []
            for embedding in embeddings:
                fingerprint_data["embeddings"].append(
                    {
                        "variant_name": embedding.variant_name,
                        "embedding_vector": embedding.embedding,
                        "created_at": embedding.created_at,
                    }
                )
            return Response(fingerprint_data, status=200)
        except FingerPrint.DoesNotExist:
            return Response({"error": "Fingerprint not found"}, status=404)
        except Exception as e:
            return Response({"error": f"Failed to retrieve fingerprint: {str(e)}"}, status=500)
