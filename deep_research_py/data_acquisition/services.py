from enum import Enum
from typing import Dict, Optional, Any, List, TypedDict
import os
import json
import asyncio
import httpx
from dotenv import load_dotenv
from deep_research_py.utils import logger
from firecrawl import FirecrawlApp
from .manager import SearchAndScrapeManager

# Load environment variables
load_dotenv()

class SearchServiceType(Enum):
    """Supported search service types."""

    FIRECRAWL = "firecrawl"
    PLAYWRIGHT_DDGS = "playwright_ddgs"
    SERPER = "serper"


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class SearchService:
    """Unified search service that supports multiple implementations."""

    def __init__(self, service_type: Optional[str] = None):
        """Initialize the appropriate search service.

        Args:
            service_type: The type of search service to use. Defaults to env var or serper.
        """
        # Determine which service to use
        if service_type is None:
            service_type = os.environ.get("DEFAULT_SCRAPER", "serper")
        
        logger.info(f"Initializing search service with type: {service_type}")
        self.service_type = service_type

        # Initialize the appropriate service
        if service_type == SearchServiceType.FIRECRAWL.value:
            logger.info("Using Firecrawl search service")
            self.firecrawl = Firecrawl(
                api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
                api_url=os.environ.get("FIRECRAWL_BASE_URL"),
            )
            self.manager = None
            self.serper = None
        elif service_type == SearchServiceType.SERPER.value:
            logger.info("Using Serper search service")
            self.firecrawl = None
            self.manager = None
            self.serper = Serper(
                api_key=os.environ.get("SERPER_API_KEY", "")
            )
        else:
            logger.info(f"Unknown service type {service_type}, defaulting to Serper")
            self.firecrawl = None
            self.manager = None
            self.serper = Serper(
                api_key=os.environ.get("SERPER_API_KEY", "")
            )
            self.service_type = SearchServiceType.SERPER.value
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

        try:
            if self.service_type == SearchServiceType.FIRECRAWL.value:
                response = await self.firecrawl.search(query, limit=limit, **kwargs)
            elif self.service_type == SearchServiceType.SERPER.value:
                response = await self.serper.search(query, limit=limit, **kwargs)
            elif self.service_type == SearchServiceType.PLAYWRIGHT_DDGS.value:
                scraped_data = await self.manager.search_and_scrape(
                    query, num_results=limit, scrape_all=True, **kwargs
                )

                # Format the response to match Firecrawl format
                formatted_data = []
                for result in scraped_data["search_results"]:
                    item = {
                        "url": result.url,
                        "title": result.title,
                        "content": "",  # Default empty content
                    }

                    # Add content if we scraped it
                    if result.url in scraped_data["scraped_contents"]:
                        scraped = scraped_data["scraped_contents"][result.url]
                        item["content"] = scraped.text

                    formatted_data.append(item)

                response = {"data": formatted_data}
            else:
                # Default to Serper if service type is unknown
                response = await self.serper.search(query, limit=limit, **kwargs)

            if save_content:
                # Create the directory if it doesn't exist
                os.makedirs("scraped_content", exist_ok=True)

                # Save each result as a separate JSON file
                for item in response.get("data", []):
                    # Create a safe filename from the first 50 chars of the title
                    title = item.get("title", "untitled")
                    safe_filename = "".join(
                        c for c in title[:50] if c.isalnum() or c in " ._-"
                    ).strip()
                    safe_filename = safe_filename.replace(" ", "_")

                    # Save the content to a JSON file
                    with open(
                        f"scraped_content/{safe_filename}.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(item, f, ensure_ascii=False, indent=2)

            return response

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {"data": []}


class Serper:
    """Simple wrapper for Serper.dev API."""

    def __init__(self, api_key: str = ""):
        # Try to get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("SERPER_API_KEY")
            
        if not api_key:
            logger.error("No Serper API key provided!")
            raise ValueError("Serper API key is required")
            
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        logger.info(f"Initialized Serper with API key: {api_key[:5]}...")

    async def search(
        self, query: str, limit: int = 5
    ) -> SearchResponse:
        """Search using Serper.dev API."""
        try:
            logger.info(f"Making Serper search request for query: {query}")
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": limit
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Serper API error: {response.status_code} - {response.text}")
                    return {"data": []}
                
                data = response.json()
                logger.info(f"Received response from Serper with {len(data.get('organic', []))} results")

                # Format the response to match our standard format
                formatted_data = []
                
                # Process organic results
                for result in data.get("organic", []):
                    formatted_data.append({
                        "url": result.get("link", ""),
                        "title": result.get("title", ""),
                        "content": result.get("snippet", "")
                    })

                # Process knowledge graph if available
                if "knowledgeGraph" in data:
                    kg = data["knowledgeGraph"]
                    formatted_data.append({
                        "url": kg.get("link", ""),
                        "title": kg.get("title", ""),
                        "content": kg.get("description", "")
                    })

                return {"data": formatted_data}

        except Exception as e:
            logger.error(f"Error searching with Serper: {e}")
            if isinstance(e, httpx.HTTPError):
                logger.error(f"HTTP Error details: {e.response.text if hasattr(e, 'response') else 'No response text'}")
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


# Initialize a global instance with the default settings
search_service = SearchService(
    service_type=os.getenv("DEFAULT_SCRAPER", "serper")
)
