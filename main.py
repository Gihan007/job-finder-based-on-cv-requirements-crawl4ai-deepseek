import asyncio
import subprocess
import sys
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
import os
import weaviate
import weaviate.classes as wvc
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from weaviate.auth import Auth
from utils.data_utils import save_venues_to_csv
from utils.scraper_utils import fetch_and_process_page, get_browser_config, get_llm_strategy
from models.venue import Venue


# load environment variables
load_dotenv()
CV_PATH = os.getenv("PATH_CV")
TESSERACT_PATH = os.getenv("PATH_TESSERACT")
OCR_TEXTS_DIR = os.getenv("OCR_TEXT_DIR")
BASE_URL = os.getenv("URL_JOB_PORTAL")  # Job Website URL
CSS_SELECTOR = os.getenv("CSS_SELECTOR_CLASS")  # Details about Web page
WEAVIATE_URL = os.getenv("URL_WEAVIATE")
WEAVIATE_API_KEY = os.getenv("API_KEY_WEAVIATE")
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')
EMBEDDING_NORMALIZE = os.getenv('EMBEDDING_NORMALIZE', 'true').lower() == 'true'

EMBEDDING_CONFIG = {
    'model_kwargs': {'device': EMBEDDING_DEVICE},
    'encode_kwargs': {'normalize_embeddings': EMBEDDING_NORMALIZE}
}



def run_ocr_extraction(cv_path, tesseract_path):
    # Executes the ocr script if cv has not been analyzed
    ocr_output_file_path = os.path.join(OCR_TEXTS_DIR, Path(CV_PATH).stem + ".txt")
    if not os.path.exists(ocr_output_file_path):
        try:
            result = subprocess.run([
                sys.executable,
                "ocr.py",
                "--cv", cv_path,
                "--tesseract", tesseract_path
            ], capture_output=True, text=True, check=True)

            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Script failed with return code", e.returncode)
            print("STDERR:\n", e.stderr)
    else:
        print("OCR texts and images already exist!\nSkip OCR Process!")


def establish_weaviate_connection():
    """Establishes connection to Weaviate instance."""
    if WEAVIATE_API_KEY is not None:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
    else:
        client = weaviate.connect_to_local()

    if client.is_ready():
        print("✅ Successfully connected to Weaviate!")
    else:
        print("❌ Failed to connect.")
    return client


def create_collection_if_not_exists(client: weaviate.WeaviateClient, collection_name: str):
    """Creates a collection if it doesn't exist."""
    if not client.collections.exists(collection_name):
        print(f"Creating collection '{collection_name}'...")
        client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="resume_id", data_type=wvc.config.DataType.TEXT)
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            )
        )
        print(f"✅ Collection '{collection_name}' created.")
    else:
        print(f"✅ Collection '{collection_name}' already exists.")

def process_and_add_resume(client: weaviate.WeaviateClient, collection_name: str):
    """Processes and adds a resume's data, but only if it doesn't already exist."""
    unique_resume_id = os.path.basename(CV_PATH)
    # Create collection if it doesn't exist
    create_collection_if_not_exists(client, collection_name)

    collection = client.collections.get(collection_name)

    # Check for duplicates
    response = collection.query.fetch_objects(
        filters=wvc.query.Filter.by_property("resume_id").equal(unique_resume_id),
        limit=1
    )

    if len(response.objects) > 0:
        print(f"❌ Resume '{unique_resume_id}' already exists. Skipping.")
        return

    # Process file - same as original code
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        **EMBEDDING_CONFIG
    )

    ocr_output_file_path = os.path.join(OCR_TEXTS_DIR, Path(CV_PATH).stem + ".txt")

    with open(ocr_output_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=25)
    chunks = text_splitter.split_text(raw_text)
    vectors = hf_embeddings.embed_documents(chunks)

    # Add to Weaviate
    with collection.batch.dynamic() as batch:
        for i, chunk_text in enumerate(chunks):
            batch.add_object(
                properties={"text": chunk_text, "resume_id": unique_resume_id},
                vector=vectors[i]
            )

    print(f"✅ Successfully added data for resume '{unique_resume_id}'.")


def retrieve_personalized_data(client: weaviate.WeaviateClient, collection_name: str, query: str, k_results: int):
    """Retrieves data filtered by a specific resume_id to ensure data isolation."""
    collection = client.collections.get(collection_name)
    resume_id = os.path.basename(CV_PATH)
    # Create query embedding
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        **EMBEDDING_CONFIG
    )
    query_vector = hf_embeddings.embed_query(query)

    # Search with resume filter
    response = collection.query.near_vector(near_vector=query_vector, limit=k_results,
                                            filters=wvc.query.Filter.by_property("resume_id").equal(resume_id))

    # Closing the client connection frees up resources
    client.close()
    # Return as simple list of text chunks
    return [obj.properties["text"] for obj in response.objects]


async def crawl_venues(retrieved_docs):
    user_prompt_extraction = "\n".join(retrieved_docs)
    print(user_prompt_extraction)
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy(user_prompt_extraction)
    session_id = "job finding"
    page_number = 1
    all_venues = []
    seen_names = set()
    # Keys which is need for Key map, defined in models/venue.py
    required_keys = list(Venue.model_fields.keys())

    async with AsyncWebCrawler(browser=browser_config) as crawler:
        while (page_number != 2):  # since this is just dummy project we dont go though lot of pages
            venues, no_results_found = await fetch_and_process_page(
                crawler,
                page_number,
                BASE_URL,
                CSS_SELECTOR,
                llm_strategy,
                session_id,
                required_keys,
                seen_names,
                user_prompt_extraction,
            )

            if no_results_found:
                print("No more venues found. Ending crawl.")
                break

            if not venues:
                print(f"No venues extracted from page {page_number}.")
                break

            all_venues.extend(venues)
            page_number += 1
            await asyncio.sleep(2)

    if all_venues:
        save_venues_to_csv(all_venues, "complete_venues.csv")
        print(f"Saved {len(all_venues)} venues to 'complete_venues.csv'.")
    else:
        print("No venues were found during the crawl.")

    llm_strategy.show_usage()

if __name__ == "__main__":
    run_ocr_extraction(CV_PATH, TESSERACT_PATH)
    weaviate_client = establish_weaviate_connection()
    process_and_add_resume(weaviate_client, "Resumes")
    # "Skills" is a dummy query; you can improve it later This is just for check The embedding is properly done or not
    retrieved_chunks = retrieve_personalized_data(weaviate_client, "Resumes", "Skills", 3)
    asyncio.run(crawl_venues(retrieved_chunks))

