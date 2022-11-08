from typing import Any, Dict, List, Optional

import os
import json
import faiss
import numpy as np
import os.path as osp

def save_json_results(query_results, outpath):
    folder_name = osp.dirname(outpath)
    os.makedirs(folder_name, exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(query_results, f)

    print(f"Save query results to  {outpath}")

class FaissRetrieval:
    """
    Compute the accuracy of the model.
    Expect the model to return a dict with the following keys:
    - "pairs": a tuple of two torch.tensors, each of shape (N, D), 
    where N is the number of pairs and D is the embedding dimension.
    Each pair is a pair of visual and language embeddings. Have a unique id for each pair.
    """

    def __init__(self, dimension=768, **kwargs):
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        self.faiss_pool = faiss.IndexFlatIP(dimension)
        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            self.faiss_pool = faiss.index_cpu_to_all_gpus(self.faiss_pool)
            print(f"Using {ngpus} to retrieve")
        else:
            print("Using CPU to retrieve")
        self.faiss_pool.reset()

    def similarity_search(self, 
        query_embeddings: np.ndarray, 
        gallery_embeddings: np.ndarray, 
        query_ids: List[Any] = None,
        gallery_ids: List[Any] = None,
        target_ids: List[Any] = None,
        top_k: int = 25,
        save_results: str = None):
        """
        Compute the similarity between queries and gallery embeddings.
        """

        self.faiss_pool.reset()
        self.faiss_pool.add(gallery_embeddings)
        top_k_scores_all, top_k_indexes_all = self.faiss_pool.search(
            query_embeddings, k=top_k
        )

        if save_results is not None:
            results_dict = {}

            for idx, (top_k_scores, top_k_indexes) in enumerate(zip(top_k_scores_all, top_k_indexes_all)):
                current_id = query_ids[idx] # current query id
                pred_ids = [gallery_ids[i] for i in top_k_indexes] # retrieved ids from gallery

                results_dict[current_id] = {
                    'pred_ids': pred_ids,
                    'scores': top_k_scores.tolist() 
                }

                if target_ids is not None:
                    tids = target_ids[idx] # target ids
                    if not isinstance(tids, list):
                        tids = [tids]
                    results_dict[current_id].update({
                        'target_ids': tids,
                    })
                    
            save_json_results(results_dict, save_results)

        return top_k_scores_all, top_k_indexes_all
