from typing import List, Dict, TypedDict, Optional
from dataclasses import dataclass
import asyncio
import openai
from deep_research_py.data_acquisition.services import search_service
from .ai.providers import trim_prompt, get_client_response
from .prompt import system_prompt


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]


@dataclass
class SerpQuery:
    query: str
    research_goal: str


async def generate_serp_queries(
    query: str,
    client: openai.OpenAI,
    model: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Generate SERP queries based on user input and previous learnings."""

    prompt = f"""Given the following prompt from the user, generate {num_queries} SERP queries to research the topic. 
    Each query should be on a new line and include a brief research goal in parentheses.
    Make sure each query is unique and not similar to each other: 
    <prompt>{query}</prompt>"""

    if learnings:
        prompt += f"\n\nHere are some learnings from previous research, use them to generate more specific queries: {' '.join(learnings)}"

    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "text"},
    )

    try:
        # Parse the text response into queries
        queries = []
        for line in response.strip().split('\n'):
            if not line.strip():
                continue
            # Extract query and research goal from line
            parts = line.split('(', 1)
            if len(parts) == 2:
                query_text = parts[0].strip()
                research_goal = parts[1].rstrip(')').strip()
                queries.append(SerpQuery(query=query_text, research_goal=research_goal))
            else:
                queries.append(SerpQuery(query=line.strip(), research_goal="General research"))
        return queries[:num_queries]
    except Exception as e:
        print(f"Error parsing queries: {e}")
        print(f"Raw response: {response}")
        return []


async def process_serp_result(
    query: str,
    search_result: SearchResponse,
    client: openai.OpenAI,
    model: str,
    num_learnings: int = 3,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    contents = [
        trim_prompt(item.get("content", ""), 25_000)
        for item in search_result["data"]
        if item.get("content")
    ]

    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    prompt = (
        f"Given the following contents from a SERP search for the query <query>{query}</query>, "
        f"generate a list of learnings and follow-up questions. "
        f"Format your response as follows:\n\n"
        f"LEARNINGS:\n"
        f"- [List {num_learnings} unique, concise learnings]\n\n"
        f"FOLLOW-UP QUESTIONS:\n"
        f"- [List {num_follow_up_questions} follow-up questions]\n\n"
        f"<contents>{contents_str}</contents>"
    )

    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "text"},
    )

    try:
        # Parse the text response
        learnings = []
        follow_up_questions = []
        current_section = None

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if line == "LEARNINGS:":
                current_section = "learnings"
            elif line == "FOLLOW-UP QUESTIONS:":
                current_section = "questions"
            elif line.startswith('- '):
                if current_section == "learnings":
                    learnings.append(line[2:])
                elif current_section == "questions":
                    follow_up_questions.append(line[2:])

        return {
            "learnings": learnings[:num_learnings],
            "followUpQuestions": follow_up_questions[:num_follow_up_questions],
        }
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response}")
        return {"learnings": [], "followUpQuestions": []}


async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client: openai.OpenAI,
    model: str,
) -> str:
    """Generate final report based on all research learnings."""

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )

    user_prompt = (
        f"Given the following prompt from the user, write a final report on the topic using "
        f"the learnings from research. The report should be in plain text format with clear sections "
        f"and bullet points where appropriate. Do not use markdown formatting. "
        f"Make sure to include ALL the learnings from research and organize them logically:\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )

    try:
        response = await get_client_response(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "text"},
        )

        # Clean up the report
        report = str(response).strip()
        
        if not report:
            print("Warning: No report content found in response")
            print("Raw response:", response)
            report = "No report content was generated."

        # Append sources
        urls_section = "\n\nSources:\n" + "\n".join(
            [f"- {url}" for url in visited_urls]
        )
        return report + urls_section

    except Exception as e:
        print(f"Error processing response: {e}")
        print("Raw response:", response)
        
        # Try to extract any text content from the response
        if isinstance(response, str):
            # Clean up the response
            cleaned_response = response.strip()
            if cleaned_response:
                return cleaned_response + "\n\nSources:\n" + "\n".join([f"- {url}" for url in visited_urls])
        
        return "Error generating report. Please try again with different parameters."


async def deep_research(
    prompt: str,
    breadth: int = 4,
    depth: int = 2,
    concurrency: int = 3,
    max_retries: int = 3,
    retry_delay: int = 5,
    client: Optional[openai.OpenAI] = None,
    model: str = "gpt-4",
) -> str:
    """Perform deep research on a topic using AI and web search."""

    if not client:
        client = openai.OpenAI()

    learnings = []
    visited_urls = []

    async def research_deeper(query: str, current_depth: int) -> None:
        """Recursively research deeper into a topic."""
        if current_depth >= depth:
            return

        print(f"Researching deeper, breadth: {breadth}, current depth: {current_depth + 1} of {depth}")
        
        # Generate search queries
        serp_queries = await generate_serp_queries(
            query=query,
            client=client,
            model=model,
        )

        # Process queries with reduced breadth for deeper searches
        reduced_breadth = max(1, breadth // 2)
        async with asyncio.Semaphore(concurrency):
            tasks = []
            for q in serp_queries[:reduced_breadth]:
                task = asyncio.create_task(
                    process_query(
                        query=q,
                        client=client,
                        model=model,
                    )
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks)

        # Process results
        for result in results:
            if result.learnings:
                learnings.extend(result.learnings)
            if result.visited_urls:
                visited_urls.extend(result.visited_urls)

        # Recursively research deeper
        new_depth = current_depth + 1
        if new_depth < depth:
            for result in results:
                if result.learnings:
                    for learning in result.learnings:
                        await research_deeper(learning, new_depth)

    # Generate search queries
    serp_queries = await generate_serp_queries(
        query=prompt,
        client=client,
        model=model,
        num_queries=breadth,
        learnings=learnings,
    )

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)

    async def process_query(serp_query: SerpQuery) -> ResearchResult:
        async with semaphore:
            try:
                # Search for content
                result = await search_service.search(serp_query.query, limit=5)

                # Collect new URLs
                new_urls = [
                    item.get("url") for item in result["data"] if item.get("url")
                ]

                # Calculate new breadth and depth for next iteration
                new_breadth = max(1, breadth // 2)
                new_depth = depth - 1

                # Process the search results
                new_learnings = await process_serp_result(
                    query=serp_query.query,
                    search_result=result,
                    num_follow_up_questions=new_breadth,
                    client=client,
                    model=model,
                )

                all_learnings = learnings + new_learnings["learnings"]
                all_urls = visited_urls + new_urls

                # If we have more depth to go, continue research
                if new_depth > 0:
                    print(
                        f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}"
                    )

                    next_query = f"""
                    Previous research goal: {serp_query.research_goal}
                    Follow-up research directions: {" ".join(new_learnings["followUpQuestions"])}
                    """.strip()

                    return await deep_research(
                        prompt=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        concurrency=concurrency,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        client=client,
                        model=model,
                    )

                return {"learnings": all_learnings, "visited_urls": all_urls}

            except Exception as e:
                if "Timeout" in str(e):
                    print(f"Timeout error running query: {serp_query.query}: {e}")
                else:
                    print(f"Error running query: {serp_query.query}: {e}")
                return {"learnings": [], "visited_urls": []}

    # Process all queries concurrently
    results = await asyncio.gather(*[process_query(query) for query in serp_queries])

    # Combine all results
    all_learnings = list(
        set(learning for result in results for learning in result["learnings"])
    )

    all_urls = list(set(url for result in results for url in result["visited_urls"]))

    return {"learnings": all_learnings, "visited_urls": all_urls}
