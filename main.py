import asyncio
import subprocess
import sys
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
import os
from pathlib import Path
import weaviate
from weaviate.auth import Auth
from langchain_community.vectorstores import Weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.data_utils import save_venues_to_csv
from utils.scraper_utils import fetch_and_process_page, get_browser_config, get_llm_strategy
from models.venue import Venue

# load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("URL_WEAVIATE")
WEAVIATE_API_KEY = os.getenv("API_KEY_WEAVIATE")
BASE_URL = os.getenv("URL_JOB_PORTAL")  # Job Website URL
CSS_SELECTOR = os.getenv("CSS_SELECTOR_CLASS")  # Details about Web page
TESSERACT_PATH = os.getenv("PATH_TESSERACT")
CV_PATH = os.getenv("PATH_CV")
OCR_TEXTS_DIR = os.getenv("OCR_TEXT_DIR")


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


def weaviate_setup():
    # Create client for Weaviate Cloud resource
    if WEAVIATE_API_KEY is not None:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
    else:
        # Create client for local Weaviate; connects to http://localhost:8080 and gRPC at localhost:50051
        client = weaviate.connect_to_local()

    # Test connection
    if client.is_ready():
        print("✅ Successfully connected to Weaviate!")
    else:
        print("❌ Failed to connect.")
    return client


def weaviate_add(client):
    # Downloads standard model from HuggingFace once and runs it locally afterwards
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Default model used, check models on MTEB Leaderboard
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    ocr_output_file_path = os.path.join(OCR_TEXTS_DIR, Path(CV_PATH).stem + ".txt")  # which comes from the Executing the ocr.py (Converting pdf into Text File)

    # Read the plain text
    with open(ocr_output_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Create splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=25)

    # Split the plain string, NOT documents
    chunks = text_splitter.split_text(raw_text)

    # View the first chunk (for example)
    #print(len(chunks))

    # Convert each chunk (string) into a Document cuz Weaviate only takes Document list , Not just raw data type data
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Now pass the Document list to Weaviate
    vector_db = Weaviate.from_documents(
        documents,
        embedding=hf_embeddings,
        client=client,
        by_text=False
    )
    return vector_db


def weaviate_retrieve(vector_db, query, retrieve_nr):
    # Retrieve top 3 most relevant chunks from Waviate
    return vector_db.similarity_search(query, k=retrieve_nr)  # "Job" is a dummy query; you can improve it later This is just for check The embedding is properly done or not


async def crawl_venues(retrieved_docs):
    user_prompt_extraction = "\n".join([doc.page_content for doc in retrieved_docs])
    print(user_prompt_extraction)
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy(user_prompt_extraction)
    session_id = "job finding"
    page_number = 1
    all_venues = []
    seen_names = set()
    # Keys which is need for Key map, defined in models/venue.py
    # REQUIRED_KEYS = ["title", "company", "location", "employment_type",
    #                  "required_skills", "experience_level", "match_reason"]
    REQUIRED_KEYS = list(Venue.model_fields.keys())

    async with AsyncWebCrawler(browser=browser_config) as crawler:
        while (page_number != 10):  # since this is just dummy project we dont go though lot of pages
            venues, no_results_found = await fetch_and_process_page(
                crawler,
                page_number,
                BASE_URL,
                CSS_SELECTOR,
                llm_strategy,
                session_id,
                REQUIRED_KEYS,
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
    weaviate_client = weaviate_setup()
    weaviate_db = weaviate_add(weaviate_client)
    retrieved_chunks = weaviate_retrieve(weaviate_db, "Skills", 5)
    asyncio.run(crawl_venues(retrieved_chunks))
