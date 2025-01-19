from vdb_upsertion import QdrantVectorDB, EmbeddingModel
from utils import load_config
from typing import List
from qdrant_client.models import PointStruct
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
DEBUG: bool = True


class SemanticSearchEngine:
    """Search engine class for the semantic search ebtween the query and the embedding vectors"""

    def __init__(self):
        self.config = load_config()
        self.qdrant_db = QdrantVectorDB(self.config)
        self.embedding_model = EmbeddingModel()

    def search(self, query: str, top_k: int = 5):
        """Apply semantic search by embedding the query and
        using Qdrant built-in search and returns the top k poi"nstructs with their payloads
        """
        query_vector = self.embedding_model.encode_text(query)
        results: List[PointStruct] = self.qdrant_db.search(
            query_vector=query_vector, top_k=top_k
        )
        formatted_results = []
        for pointstruct in results:
            formatted_results.append(
                {
                    "id": pointstruct.id,
                    "score": pointstruct.score,
                    "payload": pointstruct.payload,
                }
            )
        return formatted_results


if __name__ == "__main__":
    # this was part of an abstract that is actually present in the db, this is like a sanity check
    query = "we investigate the ionized gas kinematics and photoionization properties in comparison with AGNs with strong outflows. We find significant differences between the kinematics of ionized gas and stars for two AGNs, which indicates the presence of AGN-driven outflows. Nevertheless, the low velocity and velocity dispersion of ionized gas indicate relatively weak outflows in these AGNs. Our results highlight the importance of spatially-resolved observation in investigating gas kinematics and identifying the signatures of AGN-driven outflows"
    top_k = 5
    search_engine = SemanticSearchEngine()
    results = search_engine.search(query, top_k=top_k)
    logging.info("Top results:\n")
    for idx, res in enumerate(results):
        print(
            f"  |{idx + 1}. Title: {res['payload']['title']}|Score: {res['score']:.4f} | id : {res['id']}"
        )
        print("-" * 100)
