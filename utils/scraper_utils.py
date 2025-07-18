import json
from typing import List, Set, Tuple
from models.venue import Venue
from utils.data_utils import is_complete_venue, is_duplicate_venue
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    LLMConfig,
)


def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=False,
        verbose=True,
    )


def get_llm_strategy(user_prompt_extraction: str ) -> LLMExtractionStrategy:
    llm_config = LLMConfig(
        provider=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1),
        api_token=os.getenv("GOOGLE_API_KEY")
    )

    return LLMExtractionStrategy(
        llm_config=llm_config,  
        schema=Venue.model_json_schema(),
        extraction_type="schema",
        instruction=(
            f"The following text represents key information extracted from a user's CV: {user_prompt_extraction} \n"
            "Based on this information, identify job listings from the content below that are highly relevant to the user's experience, skills, and preferences. "
            "You must only return jobs that closely align with the user's background, even if the match is partial.\n"
            "Return each matched job as a structured object including: title, company, location, employment type, required skills, experience level, and a one-line reason why this job fits the user."
        ),
        input_format="markdown",
        verbose=True,
    )


async def check_no_results(
    crawler: AsyncWebCrawler,
    url: str,
    session_id: str,
) -> bool:
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )
    if result.success:
        if "No Results Found" in result.cleaned_html:
            return True
    else:
        print(f"Error fetching page for 'No Results Found' check: {result.error_message}")
        
    return False


async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
    page_number: int,
    base_url: str,
    css_selector: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
    required_keys: List[str],
    seen_names: Set[str],
    user_prompt_extractiobn: str
) -> Tuple[List[dict], bool]:

    url = base_url

    no_results = await check_no_results(crawler, url, session_id)
    if no_results:
        return [], True

    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=llm_strategy,
            css_selector=css_selector,
            session_id=session_id,
        ),
    )

    if not (result.success and result.extracted_content):
        print(f"Error fetching page {page_number}: {result.error_message}")
        return [], False

    extracted_data = json.loads(result.extracted_content)
    if not extracted_data:
        print(f"No jobs found on page {page_number}.")
        return [], False

    print("Extracted data:", extracted_data)

    complete_jobs = []
    for job in extracted_data:
        if job.get("error") is False:
            job.pop("error", None)

        if not is_complete_venue(job, required_keys):
            continue

        job_name = job.get("title") or job.get("name")  # Adjust based on what LLM returns
        if not job_name or is_duplicate_venue(job_name, seen_names):
            continue

        seen_names.add(job_name)
        complete_jobs.append(job)

    if not complete_jobs:
        print(f"No complete jobs found on page {page_number}.")
        return [], False

    return complete_jobs, False
