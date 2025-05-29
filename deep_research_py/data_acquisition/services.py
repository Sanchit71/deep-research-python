from enum import Enum
from typing import Dict, Optional, Any, List, TypedDict
import os
import json
import asyncio
from deep_research_py.utils import logger
from firecrawl import FirecrawlApp
from .manager import SearchAndScrapeManager
from .search import SerperSearchEngine
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()


class SearchServiceType(Enum):
    """Supported search service types."""

    FIRECRAWL = "firecrawl"
    PLAYWRIGHT_DDGS = "playwright_ddgs"
    PLAYWRIGHT_SERPER = "playwright_serper"


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class SearchService:
    """Unified search service that supports multiple implementations."""

    def __init__(self, service_type: Optional[str] = None):
        """Initialize the appropriate search service.

        Args:
            service_type: The type of search service to use. Defaults to env var or playwright_ddgs.
        """
        # Ensure .env is loaded before reading environment variables
        load_dotenv()
        
        # Determine which service to use
        if service_type is None:
            service_type = os.environ.get("DEFAULT_SCRAPER", "playwright_ddgs")
            logger.info(f"DEFAULT_SCRAPER from env: {service_type}")

        self.service_type = service_type

        # Add logging to confirm which service is being used
        logger.info(f"Initializing search service with type: {self.service_type}")

        # Initialize the appropriate service
        if service_type == SearchServiceType.FIRECRAWL.value:
            logger.info("Using Firecrawl for search and scraping")
            self.firecrawl = Firecrawl(
                api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
                api_url=os.environ.get("FIRECRAWL_BASE_URL"),
            )
            self.manager = None
        elif service_type == SearchServiceType.PLAYWRIGHT_SERPER.value:
            logger.info("Using Serper.dev for search with Playwright for scraping")
            # Use Serper for search with Playwright for scraping
            from .scraper import PlaywrightScraper
            self.firecrawl = None
            self.manager = SearchAndScrapeManager(
                search_engine=SerperSearchEngine(),
                scraper=PlaywrightScraper()
            )
            self._initialized = False
        else:
            logger.info("Using DuckDuckGo for search with Playwright for scraping")
            # Default to DDGS + Playwright
            from .search import DdgsSearchEngine
            from .scraper import PlaywrightScraper
            self.firecrawl = None
            self.manager = SearchAndScrapeManager(
                search_engine=DdgsSearchEngine(),
                scraper=PlaywrightScraper()
            )
            # Initialize resources asynchronously later
            self._initialized = False

    async def ensure_initialized(self):
        """Ensure the service is initialized."""
        if self.manager and not getattr(self, "_initialized", False):
            await self.manager.setup()
            self._initialized = True

    async def cleanup(self):
        """Clean up resources."""
        if self.manager and getattr(self, "_initialized", False):
            await self.manager.teardown()
            self._initialized = False

    async def search(
        self, query: str, limit: int = 5, save_content: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Search using the configured service.

        Returns data in a format compatible with the Firecrawl response format.
        """
        await self.ensure_initialized()

        logger.info(f"🔍 Starting search for query: '{query}' with limit={limit}")
        logger.debug(f"Service type: {self.service_type}")

        try:
            if self.service_type == SearchServiceType.FIRECRAWL.value:
                logger.debug("Using Firecrawl for search")
                response = await self.firecrawl.search(query, limit=limit, **kwargs)
            else:
                logger.debug("Using SearchAndScrapeManager for search and scrape")
                scraped_data = await self.manager.search_and_scrape(
                    query, num_results=limit, scrape_all=True, **kwargs
                )

                # Format the response to match Firecrawl format
                formatted_data = []
                for i, result in enumerate(scraped_data["search_results"], 1):
                    item = {
                        "url": result.url,
                        "title": result.title,
                        "content": "",  # Default empty content
                    }
                    
                    logger.debug(f"Search result {i}: {result.url}")
                    logger.debug(f"   Title: {result.title}")

                    # Add content if we scraped it
                    if result.url in scraped_data["scraped_contents"]:
                        scraped = scraped_data["scraped_contents"][result.url]
                        item["content"] = scraped.text
                        content_length = len(scraped.text)
                        logger.debug(f"   Content length: {content_length} characters")
                        logger.debug(f"   Status code: {scraped.status_code}")
                    else:
                        logger.warning(f"   No scraped content available for: {result.url}")

                    formatted_data.append(item)

                response = {"data": formatted_data}
                
                # Log final search results
                logger.info(f"✅ Search completed: {len(formatted_data)} results")
                urls_with_content = [item["url"] for item in formatted_data if item.get("content")]
                urls_without_content = [item["url"] for item in formatted_data if not item.get("content")]
                
                logger.info(f"📄 URLs with content: {len(urls_with_content)}")
                logger.info(f"❌ URLs without content: {len(urls_without_content)}")
                
                if urls_without_content:
                    logger.warning("URLs that failed to scrape:")
                    for url in urls_without_content:
                        logger.warning(f"   ❌ {url}")

            if save_content:
                logger.info(f"💾 Saving content to scraped_content/ directory")
                # Create the directory if it doesn't exist
                os.makedirs("scraped_content", exist_ok=True)

                # Save each result as a separate JSON file
                for i, item in enumerate(response.get("data", []), 1):
                    # Create a safe filename from the first 50 chars of the title
                    title = item.get("title", "untitled")
                    safe_filename = "".join(
                        c for c in title[:50] if c.isalnum() or c in " ._-"
                    ).strip()
                    safe_filename = safe_filename.replace(" ", "_")

                    filename = f"scraped_content/{safe_filename}_{i}.json"
                    # Save the content to a JSON file
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(item, f, ensure_ascii=False, indent=2)
                    
                    logger.debug(f"💾 Saved content to: {filename}")
                    logger.debug(f"   URL: {item.get('url', 'No URL')}")

            return response

        except Exception as e:
            logger.error(f"❌ Error during search: {str(e)}")
            logger.error(f"   Query: {query}")
            logger.error(f"   Service type: {self.service_type}")
            return {"data": []}


class Firecrawl:
    """Simple wrapper for Firecrawl SDK."""

    def __init__(self, api_key: str = "", api_url: Optional[str] = None):
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)

    async def search(
        self, query: str, timeout: int = 15000, limit: int = 5
    ) -> SearchResponse:
        """Search using Firecrawl SDK in a thread pool to keep it async."""
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.app.search(
                    query=query,
                ),
            )

            # Handle the response format from the SDK
            if isinstance(response, dict) and "data" in response:
                # Response is already in the right format
                return response
            elif isinstance(response, dict) and "success" in response:
                # Response is in the documented format
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                # Response is a list of results
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        # Handle non-dict items (like objects)
                        formatted_data.append(
                            {
                                "url": getattr(item, "url", ""),
                                "content": getattr(item, "markdown", "")
                                or getattr(item, "content", ""),
                                "title": getattr(item, "title", "")
                                or getattr(item, "metadata", {}).get("title", ""),
                            }
                        )
                return {"data": formatted_data}
            else:
                print(f"Unexpected response format from Firecrawl: {type(response)}")
                return {"data": []}

        except Exception as e:
            print(f"Error searching with Firecrawl: {e}")
            print(
                f"Response type: {type(response) if 'response' in locals() else 'N/A'}"
            )
            return {"data": []}


# Remove the global instance and create it dynamically
def get_search_service():
    """Get the search service instance based on environment configuration."""
    # Ensure .env is loaded
    load_dotenv()
    service_type = os.getenv("DEFAULT_SCRAPER", "playwright_ddgs")
    logger.info(f"Creating search service with type: {service_type}")
    logger.info(f"All environment variables: DEFAULT_SCRAPER={os.getenv('DEFAULT_SCRAPER')}")
    return SearchService(service_type=service_type)

# Don't create global instance at import time - create it when needed
search_service = None

def get_global_search_service():
    """Get or create the global search service instance."""
    global search_service
    if search_service is None:
        search_service = get_search_service()
    return search_service
