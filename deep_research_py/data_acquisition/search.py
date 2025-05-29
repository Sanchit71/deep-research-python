import asyncio
import aiohttp
import os
from dataclasses import dataclass
from typing import List, Dict, Any
from deep_research_py.utils import logger
from abc import ABC, abstractmethod
from duckduckgo_search import DDGS


# ---- Data Models ----


@dataclass
class SearchResult:
    """Standardized search result format regardless of the search engine used."""

    title: str
    url: str
    description: str
    position: int
    metadata: Dict[str, Any] = None


# ---- Search Engine Interfaces ----


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @abstractmethod
    async def search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search and return standardized results."""
        pass


class DdgsSearchEngine:
    """DuckDuckGo search engine implementation."""

    def __init__(self):
        self.ddgs = DDGS()
        logger.info("Initialized DdgsSearchEngine")

    async def search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search using DDGS and return standardized results."""
        logger.info(f"Performing DuckDuckGo search for query: '{query}' with {num_results} results")
        
        try:
            # Convert to async operation
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: list(self.ddgs.text(query, max_results=num_results))
            )

            # Convert to standardized format
            standardized_results = []
            for i, result in enumerate(results):
                standardized_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", ""),
                        description=result.get("body", ""),
                        position=i + 1,
                        metadata=result,
                    )
                )

            logger.info(f"Successfully processed {len(standardized_results)} DuckDuckGo search results")
            return standardized_results

        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {str(e)}")
            return []


class SerperSearchEngine:
    """Serper.dev search engine implementation."""

    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.api_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable is required")
        
        logger.info(f"Initialized SerperSearchEngine with API URL: {self.api_url}")

    async def search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> List[SearchResult]:
        """Perform a search using Serper.dev and return standardized results."""
        logger.info(f"Performing Serper.dev search for query: '{query}' with {num_results} results")
        
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": min(num_results, 100)  # Serper.dev max is 100
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, 
                    json=payload, 
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Serper API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    logger.info(f"Serper.dev returned {len(data.get('organic', []))} results")

            # Convert to standardized format
            standardized_results = []
            organic_results = data.get("organic", [])
            
            for i, result in enumerate(organic_results[:num_results]):
                standardized_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        description=result.get("snippet", ""),
                        position=i + 1,
                        metadata=result,
                    )
                )

            logger.info(f"Successfully processed {len(standardized_results)} Serper.dev search results")
            return standardized_results

        except Exception as e:
            logger.error(f"Error during Serper search: {str(e)}")
            return []
