# annoy_index.py

from annoy import AnnoyIndex
import numpy as np
from .models import FingerPrintEmbed, FingerPrint
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

VECTOR_SIZE = 128  # Assuming 128-dim vectors from ORB
INDEX_FILE = 'annoy_fingerprint_index.ann'

# Mapping between annoy IDs and actual FingerPrint model IDs
annoy_id_to_fingerprint_id = {}
fingerprint_id_to_annoy_ids = {}

# Use multiple indexes with different configurations for ensemble voting
annoy_indexes = {
    'angular': AnnoyIndex(VECTOR_SIZE, 'angular'),
    'euclidean': AnnoyIndex(VECTOR_SIZE, 'euclidean'),
    'manhattan': AnnoyIndex(VECTOR_SIZE, 'manhattan')
}

def build_annoy_index():
    """
    Load all known embeddings and build multiple Annoy indexes for ensemble voting.
    """
    global annoy_indexes, annoy_id_to_fingerprint_id, fingerprint_id_to_annoy_ids
    annoy_id_to_fingerprint_id = {}
    fingerprint_id_to_annoy_ids = {}
    
    # Clear existing indexes
    for index in annoy_indexes.values():
        index.unbuild()
    
    # Recreate indexes
    annoy_indexes = {
        'angular': AnnoyIndex(VECTOR_SIZE, 'angular'),
        'euclidean': AnnoyIndex(VECTOR_SIZE, 'euclidean'),
        'manhattan': AnnoyIndex(VECTOR_SIZE, 'manhattan')
    }

    all_embeds = FingerPrintEmbed.objects.filter(
        fingerprint_id__in=[f.id for f in FingerPrint.objects.filter(is_known=True)]
    )

    current_annoy_id = 0

    for embed in all_embeds:
        vec = np.array(embed.embedding).astype(np.float32)

        if len(vec) != VECTOR_SIZE:
            continue  # skip malformed vectors

        # Add to all indexes
        for index in annoy_indexes.values():
            index.add_item(current_annoy_id, vec)

        annoy_id_to_fingerprint_id[current_annoy_id] = embed.fingerprint_id
        fingerprint_id_to_annoy_ids.setdefault(embed.fingerprint_id, []).append(current_annoy_id)

        current_annoy_id += 1

    # Build with more trees for better accuracy
    for index in annoy_indexes.values():
        index.build(50)  # Increased from 10 to 50 trees

def search_annoy_index(query_vectors, top_k=20):
    """
    Search multiple Annoy indexes using ensemble voting for improved accuracy.
    Return the best match and its confidence score.
    """
    if not annoy_id_to_fingerprint_id:
        return None, 0.0

    # Ensemble voting with weighted scores
    ensemble_scores = defaultdict(list)
    
    for vec in query_vectors:
        vec = np.array(vec).astype(np.float32)
        
        # Search each index type
        for metric_name, index in annoy_indexes.items():
            try:
                idxs, dists = index.get_nns_by_vector(vec, top_k, include_distances=True)
                
                for i, annoy_id in enumerate(idxs):
                    fp_id = annoy_id_to_fingerprint_id.get(annoy_id)
                    if fp_id is None:
                        continue
                    
                    # Convert distance to similarity based on metric
                    if metric_name == 'angular':
                        similarity = 1 - dists[i] / 2
                    elif metric_name == 'euclidean':
                        # Normalize euclidean distance to [0,1] similarity
                        similarity = 1 / (1 + dists[i])
                    else:  # manhattan
                        # Normalize manhattan distance to [0,1] similarity
                        similarity = 1 / (1 + dists[i])
                    
                    # Weight different metrics
                    weight = {'angular': 0.4, 'euclidean': 0.35, 'manhattan': 0.25}[metric_name]
                    ensemble_scores[fp_id].append(similarity * weight)
                    
            except Exception as e:
                logger.warning(f"Error searching {metric_name} index: {e}")
                continue

    if not ensemble_scores:
        return None, 0.0

    # Calculate weighted average scores
    final_scores = {}
    for fp_id, scores in ensemble_scores.items():
        if scores:
            # Use weighted average with outlier removal
            scores = np.array(scores)
            # Remove outliers (scores more than 2 std devs from mean)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            filtered_scores = scores[np.abs(scores - mean_score) <= 2 * std_score]
            
            if len(filtered_scores) > 0:
                final_scores[fp_id] = np.mean(filtered_scores)
            else:
                final_scores[fp_id] = mean_score

    # Get best match with confidence calculation
    best_match = None
    best_score = -1.0
    
    for fp_id, score in final_scores.items():
        if score > best_score:
            best_match = fp_id
            best_score = score

    # Calculate confidence based on score distribution
    if len(final_scores) > 1:
        sorted_scores = sorted(final_scores.values(), reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if sorted_scores[0] > 0 else 0
        # Boost confidence for high scores
        if best_score > 0.8:
            confidence = min(1.0, confidence * 1.5)
    else:
        confidence = best_score

    return best_match, round(max(best_score, confidence), 4)
