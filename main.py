import asyncio
import subprocess
import sys
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
import os
import weaviate
from weaviate.auth import AuthApiKey
from langchain_community.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils.data_utils import save_venues_to_csv
from utils.scraper_utils import fetch_and_process_page, get_browser_config, get_llm_strategy
from models.venue import Venue

# load environment variables
load_dotenv()
weaviate_url = os.getenv("URL_WEAVIATE")
weaviate_api_key = os.getenv("API_KEY_WEAVIATE")
BASE_URL = os.getenv("URL_JOB_PORTAL")  # Job Website URL
CSS_SELECTOR = os.getenv("CSS_SELECTOR_CLASS")  # Details about Web page
TESSERACT_PATH = os.getenv("PATH_TESSERACT")
CV_PATH = os.getenv("PATH_CV")


def run_ocr_extraction(cv_path, tesseract_path):
    # Executes the ocr script
    subprocess.call([sys.executable,
                     "target_script.py",
                     "--cv", cv_path,
                     "--tesseract", tesseract_path,
                     ])


def weaviate_setup():
    # Create client
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=AuthApiKey(weaviate_api_key),
    )

    # Test connection
    if client.is_ready():
        print("✅ Successfully connected to Weaviate!")
    else:
        print("❌ Failed to connect.")
    return client


def weaviate_add(client):
    huggingface_embeddings = HuggingFaceEmbeddings()
    pages = "/extracted_texts/cv.txt"  # which comes from the Executing the ocr.py (Converting pdf into Text File)

    # Read the plain text
    with open("extracted_texts/cv.txt", "r", encoding="utf-8") as f:
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
        embedding=huggingface_embeddings,
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
