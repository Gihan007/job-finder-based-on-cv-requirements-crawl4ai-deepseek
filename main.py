import asyncio
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from langchain_community.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

#from config import BASE_URL, CSS_SELECTOR, REQUIRED_KEYS
from utils.data_utils import save_venues_to_csv
from utils.scraper_utils import fetch_and_process_page, get_browser_config, get_llm_strategy

load_dotenv()


huggingface_embeddings = HuggingFaceEmbeddings()
pages = "/extracted_texts/cv.txt" #which is came from the Executing the ocr.py (Converting pdf into Text File)

# Read the plain text
with open("extracted_texts/cv.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Create splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=25)

# Split the plain string, NOT documents
chunks = text_splitter.split_text(raw_text)

# View the first chunk (for example)
#print(len(chunks))

# # Your credentials
# weaviate_url = "https://uv5embawqwgf65fbuhm1q.c0.asia-southeast1.gcp.weaviate.cloud"
# weaviate_api_key = "PJr2228NasulpFvGwNPcJdjnnNnDwpN2iFJq"


weaviate_url = "https://oifn3nasg6goiib7a1ig.c0.asia-southeast1.gcp.weaviate.cloud"
weaviate_api_key = "R1QyVUgvbGhDN3hQQmlJel96aGN3Y0lFQjhWa25yelhzMVJaZ3FycXc2a2Y0WDNaYTVzQ2huT3lHUDRBPV92MjAw"


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
    

#Details about Web page , In demo i used dialog carres Which is opened now

BASE_URL = "https://hcmcloud.dialog.lk/CareerPortal/Careers?q=bEopnWmcv9llMiBG3zygOw%3D%3D"
CSS_SELECTOR = "[class^='hcm-jpp-job']" #Inspect mode Class name
REQUIRED_KEYS = ["title", "company", "location", "employment_type", "required_skills", "experience_level", "match_reason"]  #keys which is need for Key map

# Convert each chunk (string) into a Document cuz Weaviate only takes Document list , Not just raw data type data 
documents = [Document(page_content=chunk) for chunk in chunks]

# Now pass the Document list to Weaviate
vector_db = Weaviate.from_documents(
    documents,
    embedding=huggingface_embeddings,
    client=client,
    by_text=False
)

# Retrieve top 3 most relevant chunks from Waviate
retrieved_docs = vector_db.similarity_search("Skills", k=5)  # "Job" is a dummy query; you can improve it later This is just for check The embedding is properly done or not

async def crawl_venues():
    user_prompt_extractiobn = "\n".join([doc.page_content for doc in retrieved_docs])
    print(user_prompt_extractiobn)
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy(user_prompt_extractiobn)
    session_id = "job finding"
    page_number = 1
    all_venues = []
    seen_names = set()

    async with AsyncWebCrawler(browser = browser_config) as crawler:
        while (page_number != 10): #since this is just dummy project we dont go though lot of pages
            venues, no_results_found = await fetch_and_process_page(
                crawler,
                page_number,
                BASE_URL,
                CSS_SELECTOR,
                llm_strategy,
                session_id,
                REQUIRED_KEYS,
                seen_names,
                user_prompt_extractiobn, 
            )

            if no_results_found:
                #print("No more venues found. Ending crawl.")
                break

            if not venues:
                #print(f"No venues extracted from page {page_number}.")
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

async def main():
    await crawl_venues()

if __name__ == "__main__":
    asyncio.run(main())