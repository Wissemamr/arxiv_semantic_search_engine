# Semantic Search Engine for Arxiv Research Papers

<br>
This repository contains a semantic search application fro ArXiv research papers that leverages a pre-trained DistilBERT model for generating vector embeddings and Qdrant Vector database to store them and retrieve the most relevant research papers based on user queries and and the results of the search engine. 

## Pipeline

1. **Data acquisition**:
    - We used the [ArXiv-10](https://huggingface.co/datasets/effectiveML/ArXiv-10) dataset from HuggingFace that contains 100k rows of research papers in 10 disciplines : computer science, astrophysics, quantum physics, statistics... featuring the id of teh paper, its abstract and its label(discipline/domain)

1. **Embedding Creation**:
   - The DistilBERT model was used due to limited compute power (no GPU) to encode the dataset, more specifically the abstract column into dense vector representations of dimension 768.

2. **Vector Database**:
   - The [Qdrant Cloud](https://qdrant.tech/documentation/cloud-intro/) was used to instance store the pre-computed embeddings of the research papers.

3. **Semantic Search **:
   - A user enters a query into the Streamlit app and can choose how many search results to display.
   - The query is encoded into a vector using DistilBERT.
   - The vector is sent to the Qdrant Cloud instance, using built-in Qdrant semantic search, taking cosine similarity as a simialrity metric, Qdrant retrieves the top-k most similar poinstructs and their payloads.
   - The payloads of the retrieved pointstruts including the title and the text version of the abstract itself are displayed in streamlit cards within the interface.

## Tech Stack
- **Python + Pytorch** : for the implementation.
- **HugghingFace (Dataset + Transformers)** : For data acquisition and generating vector embeddings
- **Qdrant Cloud / Docker** : As a vector database for storing, indexing the data, searching and retrivieng the search results.
- **Streamlit** : For builduing the user interface.
- **Box / PyYAML** : For managing configuration files.



## Installation and Setup

### Prerequisites
- Python 3.8+
- A Qdrant Cloud account
- Docker

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Wissemamr/arxiv_semantic_search_engine.git
   ```


2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   With docker
   - Pull the latest Qdrant iamge from docker hub 
   ```
   docker pull qdrant/qdrant
   ```

   - Run the service
   ```
   docker run -p 6333:6333 -p 6334:6334 \
      -v $(pwd)/qdrant_storage:/qdrant/storage:z \
      qdrant/qdrant
   ```
4. **Set Up Configuration**:
   Create a `config.yaml` file in the root directory with the following structure:
     ```yaml
     qdrant:
       url: "<your-qdrant-cloud-url>"
       api_key: "<your-api-key>"
       collection_name: "arxiv-collection"
     ```

4. **Run the Application**:
   ```bash
   cd src
   streamlit run app.py
   ```

5. **Access the App**:
   - Open a browser and navigate to `http://localhost:8501`.




## Future Improvements

- Use a richer dataset or add more columns to the current dataset such as publication year, the list of keywords, citation link and the link to access the document directly.
- Enable pagination for large result sets.
- Integrate additional metadata filters (e.g., year, author).
- Deploy the app to a cloud platform for public access.


## Useful Ressources :

- [Hugging Face](https://huggingface.co/)
- [Qdrant](https://qdrant.tech/) 
- [Streamlit](https://streamlit.io/) 
