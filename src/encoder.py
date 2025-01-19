# =============================================
# Encoder model
# =============================================
from typing import Dict, Any
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from icecream import ic
import logging
from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
DEBUG: bool = False
config = load_config()


class EmbeddingModel:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.model.name)
        self.model = DistilBertModel.from_pretrained(config.model.name).to(self.device)
        self.model.eval()

    def get_model_info(self) -> Dict[str, Any]:
        return {"embed_dim": self.model.config.hidden_size}  # changed to use config

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            # output of shape (batch_size, sequence_length, hidden_size)
            outputs = self.model(**inputs)
            if DEBUG:
                logging.info(f"Tokenized sentence: {inputs}")
                logging.info(outputs.last_hidden_state.shape)
        # single vector to represent the entire sentence, not a vector for each token.
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings


if __name__ == "__main__":
    encoder = EmbeddingModel()
    model_info = encoder.get_model_info()
    ic(model_info)

    # if you get this ref, you have a great taste in music :))
    test_sent = "Loathe the way they light candles in Rome, but love the sweet air of the votives, hurt and grieve but don't suffer alone, engage with the pain as a motive"
    embedding = encoder.encode_text(test_sent)
    logging.info(f"Embedding : {embedding}")
