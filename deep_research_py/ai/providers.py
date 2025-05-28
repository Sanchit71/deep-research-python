import os
import typer
import json
import httpx
from openai import AsyncOpenAI
import tiktoken
from typing import Optional, List, Dict
from rich.console import Console
from dotenv import load_dotenv
from .text_splitter import RecursiveCharacterTextSplitter
from deep_research_py.config import EnvironmentConfig

load_dotenv()


class AIClientFactory:
    """Factory for creating AI clients for different providers."""

    @classmethod
    def create_client(cls, api_key: str, base_url: str) -> AsyncOpenAI:
        """Create an AsyncOpenAI-compatible client for the specified provider."""
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def get_client(
        cls,
        service_provider_name: Optional[str] = None,
        console: Optional[Console] = None,
    ) -> AsyncOpenAI:
        """Get a configured AsyncOpenAI client using environment variables."""
        console = console or Console()

        try:
            # Get and validate the provider configuration
            config = EnvironmentConfig.validate_provider_config(
                service_provider_name, console
            )

            # Create the client
            return cls.create_client(api_key=config.api_key, base_url=config.base_url)

        except ValueError:
            raise typer.Exit(1)
        except Exception as e:
            console.print(
                f"[red]Error initializing {service_provider_name or EnvironmentConfig.get_default_provider()} client: {e}[/red]"
            )
            raise typer.Exit(1)

    @classmethod
    def get_model(cls, service_provider_name: Optional[str] = None) -> str:
        """Get the configured model for the specified provider."""
        config = EnvironmentConfig.get_provider_config(service_provider_name)
        if not config.model:
            raise ValueError(f"No model configured for {config.service_provider_name}")
        return config.model


async def get_client_response(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    """Get response from AI client."""
    try:
        if model.startswith("gemini"):
            # Special handling for Gemini API
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            url = f"{base_url}/models/{model}:generateContent"
            
            # Get API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise Exception("No Gemini API key provided!")

            # Prepare the request
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }

            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                contents.append({
                    "parts": [{"text": msg["content"]}],
                    "role": msg["role"]
                })

            # Make the request
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    url,
                    headers=headers,
                    json={"contents": contents},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                # Extract the text content from the response
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    return content
                else:
                    raise Exception("No response content found in Gemini API response")

        else:
            # Handle other models (e.g., OpenAI)
            response = await client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
            )
            return response.choices[0].message.content

    except httpx.HTTPError as e:
        if e.response.status_code == 429:
            raise Exception("Rate limit exceeded. Please try again later.")
        elif e.response.status_code == 400:
            raise Exception("Invalid request to AI API. Please check your parameters.")
        elif e.response.status_code == 401:
            raise Exception("Authentication failed. Please check your API key.")
        elif e.response.status_code == 403:
            raise Exception("Access forbidden. Please check your API key permissions.")
        else:
            raise Exception(f"Failed to connect to AI API: {str(e)}")
    except Exception as e:
        print(f"Unexpected error while calling AI API: {str(e)}")
        raise Exception(f"Error calling AI API: {str(e)}")


MIN_CHUNK_SIZE = 140
encoder = tiktoken.get_encoding(
    "cl100k_base"
)  # Updated to use OpenAI's current encoding


def trim_prompt(
    prompt: str, context_size: int = int(os.getenv("CONTEXT_SIZE", "128000"))
) -> str:
    """Trims a prompt to fit within the specified context size."""
    if not prompt:
        return ""

    length = len(encoder.encode(prompt))
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    # Estimate characters to remove (3 chars per token on average)
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    trimmed_prompt = (
        splitter.split_text(prompt)[0] if splitter.split_text(prompt) else ""
    )

    # Handle edge case where trimmed prompt is same length
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    return trim_prompt(trimmed_prompt, context_size)
