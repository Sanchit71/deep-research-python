import asyncio
import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

async def test_gemini_api():
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return

    # Test message that will generate a JSON response
    test_message = {
        "role": "user",
        "content": """Please analyze the following text and provide a JSON response with:
        1. Key points
        2. Follow-up questions
        Format your response as a JSON object with these fields:
        {
            "key_points": ["point1", "point2"],
            "follow_up_questions": ["question1", "question2"]
        }
        
        Text to analyze: Artificial Intelligence (AI) is transforming various industries through automation and intelligent decision-making. Machine learning, a subset of AI, enables systems to learn from data and improve over time. Deep learning, using neural networks, has revolutionized tasks like image recognition and natural language processing."""
    }

    # Gemini API endpoint - Using the correct model name from the available list
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    # Request body
    data = {
        "contents": [{
            "parts": [{"text": test_message["content"]}],
            "role": "user"
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        }
    }

    try:
        print("Making request to Gemini API...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, try to list available models
            list_url = "https://generativelanguage.googleapis.com/v1/models"
            list_response = await client.get(list_url, headers={"x-goog-api-key": api_key})
            
            if list_response.status_code == 200:
                models = list_response.json()
                print("\nAvailable Models:")
                for model in models.get("models", []):
                    print(f"- {model['name']}")
            else:
                print(f"Error listing models: {list_response.text}")
            
            # Now make the actual request
            response = await client.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                print(f"Error Response: {response.text}")
                return
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Try to parse JSON
                try:
                    # Remove markdown code block markers if present
                    content = content.replace("```json", "").replace("```", "").strip()
                    
                    # Clean control characters
                    import re
                    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
                    
                    # Try multiple parsing approaches
                    try:
                        # First try: direct parse
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError as e:
                        # Second try: find JSON object
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_str = content[start:end]
                            try:
                                parsed_json = json.loads(json_str)
                            except json.JSONDecodeError:
                                # Third try: fix formatting
                                try:
                                    # Fix common JSON formatting issues
                                    content = re.sub(r'}\s*{', '},{', content)
                                    content = re.sub(r'\]\s*\[', '],[', content)
                                    content = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
                                    parsed_json = json.loads(content)
                                except json.JSONDecodeError:
                                    print("Failed to parse response as JSON")
                                    return
                        else:
                            print("No JSON object found in response")
                            return
                    
                    # Print formatted report
                    print("\n=== Analysis Report ===\n")
                    
                    # Print key points
                    print("Key Points:")
                    for i, point in enumerate(parsed_json.get("key_points", []), 1):
                        print(f"{i}. {point}")
                    
                    # Print follow-up questions
                    print("\nFollow-up Questions:")
                    for i, question in enumerate(parsed_json.get("follow_up_questions", []), 1):
                        print(f"{i}. {question}")
                    
                except Exception as e:
                    print(f"Error processing response: {str(e)}")
            else:
                print("No response from Gemini API")
                
    except Exception as e:
        print(f"Error making request: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_gemini_api()) 