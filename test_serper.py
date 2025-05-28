import asyncio
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_serper_api():
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("Error: SERPER_API_KEY not found in environment variables")
        return

    print(f"Using API key: {api_key[:5]}...")
    
    url = "https://google.serper.dev/search"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": "test query",
        "num": 5
    }

    try:
        async with httpx.AsyncClient() as client:
            print("Making request to Serper API...")
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code == 200:
                data = response.json()
                print("\nSuccess! API is working.")
                print(f"Number of organic results: {len(data.get('organic', []))}")
                
                # Print first result if available
                if data.get('organic'):
                    first_result = data['organic'][0]
                    print("\nFirst result:")
                    print(f"Title: {first_result.get('title')}")
                    print(f"Link: {first_result.get('link')}")
                    print(f"Snippet: {first_result.get('snippet')}")
            else:
                print(f"\nError: {response.text}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_serper_api()) 