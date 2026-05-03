import certifi
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import ssl
import asyncio
import scrapy
from scrapy.http import TextResponse

from scrapy.crawler import CrawlerProcess


from langchain_openai import OpenAIEmbeddings

from logger import (
    Colors,
    log_header,
    log_info,
    log_success,
    log_error,
    log_warning,
)

load_dotenv()

# ---------------- SSL CONFIG ----------------
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ---------------- EMBEDDINGS with openAI embeddings ----------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
    retry_min_seconds=10,
)





# ============================================================
# 🔹 SCRAPY SPIDER
# ============================================================
# Global list to hold the data so it can be passed to asyncio later
scraped_data = []

class DailyStarSpider(scrapy.Spider):
    name = "dailystar"
    start_urls = ["https://www.thedailystar.net/spotlight"]

    custom_settings = {
        "DEPTH_LIMIT": 3,
        "LOG_LEVEL": "WARNING",
        "CLOSESPIDER_PAGECOUNT": 100,
    }

    def parse(self, response):
        # 🚫 Skip non-text responses safely without needing extra imports
        if not hasattr(response, 'text'):
            return

        # Extract text from article paragraphs
        paragraphs = response.css("article p::text").getall()
        raw_text = " ".join(paragraphs).strip()

        # Save valid articles directly to the global list for the async pipeline
        if len(raw_text) > 300:
            scraped_data.append({
                "url": response.url,
                "raw_content": raw_text,
            })

        # Follow links (Scrapy automatically filters out duplicate URLs)
        for href in response.css("a::attr(href)").getall():
            if href and href.startswith(("http", "/")):
                yield response.follow(href, self.parse)



# ---------------- VECTOR STORE ----------------
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

# # ---------------- TAVILY ----------------
# tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=10000)
# tavily_crawl = TavilyCrawl()


# ============================================================
# 🔹 INDEXING FUNCTION (WITH SEMAPHORE CONTROL)
# ============================================================
async def index_documents_async(documents: List[Document], batch_size: int = 10):
    log_header("INDEXING DOCUMENTS INTO VECTOR STORE")

    log_info(
        f"VectorStore Indexing: Starting to index {len(documents)} documents in batches of {batch_size}",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # 🔒 Semaphore to prevent session conflicts
    semaphore = asyncio.Semaphore(1)

    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                log_success(
                    f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                )
                return True
            except Exception as e:
                log_error(
                    f"VectorStore Indexing: Failed to add batch {batch_num} - {e}"
                )
                return False

    # Run tasks (controlled by semaphore)
    tasks = [
        add_batch(batch, idx + 1)
        for idx, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes safely
    successful_batches = sum(1 for r in results if r is True)

    if successful_batches == len(batches):
        log_success(
            f"VectorStore Indexing: All {len(batches)} batches indexed successfully!"
        )
    else:
        log_warning(
            f"VectorStore Indexing: {successful_batches}/{len(batches)} batches indexed successfully."
        )


# ============================================================
# 🔹 MAIN PIPELINE
# ============================================================




async def main():
        
        # Convert the raw scraped dictionaries into LangChain Documents
        all_docs = [
            Document(
                page_content=item["raw_content"],
                metadata={"source": item["url"]},
            )
            for item in scraped_data
        ]

        if not all_docs:
            log_error("No documents were scraped. Exiting pipeline.")
            return

        log_success(
            f"Crawling completed. Total documents extracted: {len(all_docs)}"
        )

        # ---------------- CHUNKING ----------------
        log_header("DOCUMENT CHUNKING PHASE")

        log_info(
            f"✂️ Text Splitter: Processing {len(all_docs)} documents with 1000 chunk size and 100 overlap",
            Colors.YELLOW,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        splitted_docs = text_splitter.split_documents(all_docs)

        log_success(
            f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
        )

        # ---------------- INDEXING ----------------
        await index_documents_async(splitted_docs, batch_size=10)

        # ---------------- SUMMARY ----------------
        log_header("PIPELINE COMPLETE")

        log_success("🎉 Documentation ingestion pipeline finished successfully!")

        log_info("📊 Summary:", Colors.BOLD)
        log_info(f"   • Documents extracted: {len(all_docs)}")
        log_info(f"   • Chunks created: {len(splitted_docs)}")


# ============================================================
# 🔹 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # 1. Run Scrapy Synchronously First
    log_header("SCRAPY CRAWLING PHASE")
    log_info(
        "Starting to crawl documentation from https://www.thedailystar.net/",
        Colors.PURPLE,
    )
    
    process = CrawlerProcess()
    process.crawl(DailyStarSpider)
    process.start() # Blocks the main thread until the crawl finishes
    
    # 2. Once Scrapy finishes and shuts down its reactor, safely start asyncio
    asyncio.run(main())
