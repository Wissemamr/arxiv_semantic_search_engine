import pandas as pd
from colorama import Fore
from box import Box
from qdrant_client import QdrantClient, models
import numpy as np
import uuid
from utils import load_config
from encoder import EmbeddingModel
from arxiv_data import ArxivData
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
config = load_config()


class QdrantVectorDB:
    def __init__(self, config: Box):
        self.cloud_client = QdrantClient(
            url=config.qdrant.url, api_key=config.qdrant.api_key, timeout=30.0
        )
        self.collection_name = config.qdrant.collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            self.cloud_client.get_collection(self.collection_name)
        except Exception:
            self.cloud_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=config.model.embedding_dim, distance=models.Distance.COSINE
                ),
            )

    def upsert_points(self, embeddings, ids, payloads):
        points = []
        for id_, emb, payload in zip(ids, embeddings, payloads):
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            points.append({"id": id_, "vector": emb, "payload": payload})

        self.cloud_client.upsert(collection_name=self.collection_name, points=points)
        logging.info(f"{Fore.CYAN}Upserted {len(embeddings)} points.")

    def search(self, query_vector, top_k=5):
        search_result = self.cloud_client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_k
        )
        return search_result


def process_batch(
    start_index, end_index, df, embedding_model, qdrant_db, unique_payloads
):
    """Handles embedding the data an dupserting the pointstruvts to Qdrant DB"""
    embeddings = []
    ids = []
    payloads = []
    failed_rows = []
    batch_data = df.iloc[start_index:end_index]
    for idx, row in batch_data.iterrows():
        try:
            payload = {
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "label": str(row["label"]),
            }
            unique_key = (payload["title"], payload["abstract"])
            if unique_key in unique_payloads:
                continue
            unique_payloads.add(unique_key)
            emb = embedding_model.encode_text(row["abstract"])
            embeddings.append(emb)
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            payloads.append(payload)
        except Exception as e:
            failed_rows.append({"row": row.to_dict(), "error": str(e)})
            logging.info(f"{Fore.RED}Error processing row {idx}: {str(e)}{Fore.RESET}")
    if embeddings:
        try:
            qdrant_db.upsert_points(embeddings, ids, payloads)
        except Exception as e:
            logging.info(f"{Fore.RED}Error during batch upsert: {str(e)}{Fore.RESET}")
            failed_rows.extend(
                [
                    {"row": row[1].to_dict(), "error": str(e)}
                    for row in batch_data.iterrows()
                ]
            )
    return failed_rows


def main_upsertion(start_index=0, end_index=None):
    try:
        arxiv_data = ArxivData(config.data.filepath)
        df = arxiv_data.clean_df()
        embedding_model = EmbeddingModel()
        logging.info(
            f"{Fore.GREEN}[+]{Fore.RESET} Embedding model initialized on {embedding_model.device}"
        )
        qdrant_db = QdrantVectorDB(config)
        logging.info(f"{Fore.GREEN}[+]{Fore.RESET} Qdrant connection established")
        if end_index is None:
            end_index = len(df)
        failed_rows = []
        # to avoid duplicates
        unique_payloads = set()
        for batch_start in range(start_index, end_index, config.data.batch_size):
            batch_end = min(batch_start + config.data.batch_size, end_index)
            logging.info(
                f"{Fore.MAGENTA}Processing batch {batch_start} to {batch_end}..."
            )
            batch_failed_rows = process_batch(
                batch_start, batch_end, df, embedding_model, qdrant_db, unique_payloads
            )
            failed_rows.extend(batch_failed_rows)
        if failed_rows:
            failed_df = pd.DataFrame(failed_rows)
            failed_df.to_csv("../data/failed_rows_.csv", index=False)
            logging.info(
                f"{Fore.YELLOW}Failed rows saved to 'failed_rows.csv'{Fore.RESET}"
            )
        logging.info(f"{Fore.BLUE }Processing completed successfully!{Fore.RESET}")
    except Exception as e:
        logging.info(f"{Fore.RED}Critical error: {str(e)}")
        raise


if __name__ == "__main__":
    # start index inclusiev, end index exclsuive
    # main(start_index=0, end_index=500)
    # main(start_index=1000, end_index=1500)
    # main(start_index=2000, end_index=10000)
    # main(start_index=10000, end_index=20000)
    # main(start_index=19900, end_index=20000)
    # main(start_index=20000, end_index=30000)
    # main(start_index=30000, end_index=40000)
    # main(start_index=40000, end_index=50000)
    # main(start_index=50000, end_index=50100)
    # main(start_index=50100, end_index=60000)
    # main(start_index=60000, end_index=70000)
    # main(start_index=70000, end_index=75000)
    # main(start_index=75000, end_index=80000)
    # =>
    # main_upsertion(start_index=80000, end_index=85000)
    # main_upsertion(start_index=84550, end_index=85000)
    # main_upsertion(start_index=85000, end_index=90000)
    # 96200
    # main_upsertion(start_index=90000, end_index=96200)
    main_upsertion(start_index=96200, end_index=100000)
