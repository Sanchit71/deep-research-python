from typing import List, Tuple
import openai
import json
from .prompt import system_prompt
from .ai.providers import get_client_response
from deep_research_py.utils import logger
from prompt_toolkit import PromptSession


async def generate_feedback(query: str, client: openai.OpenAI, model: str) -> Tuple[List[str], List[str]]:
    """Generates follow-up questions and collects answers to clarify research direction."""

    logger.info("🤔 Generating follow-up questions for research clarification")
    logger.debug(f"Initial query: {query}")

    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": f"Given this research topic: {query}, generate 3-5 follow-up questions to better understand the user's research needs and goals. Focus on clarifying scope, specific interests, target audience, and success criteria. Return the response as a JSON object with a 'questions' array field.",
            },
        ],
        response_format={"type": "json_object"},
    )

    # Parse the JSON response
    try:
        questions = response.get("questions", [])
        logger.info(f"✅ Generated {len(questions)} follow-up questions")
        
        # Collect answers interactively
        session = PromptSession()
        answers = []
        
        print("\n[bold yellow]Follow-up Questions:[/bold yellow]")
        for i, question in enumerate(questions, 1):
            logger.debug(f"Asking question {i}: {question}")
            print(f"\n[bold blue]Q{i}:[/bold blue] {question}")
            answer = await session.prompt_async("➤ Your answer: ")
            answers.append(answer)
            logger.debug(f"Answer {i}: {answer}")
            print()
        
        logger.info(f"✅ Collected {len(answers)} answers")
        return questions, answers
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing JSON response: {e}")
        logger.debug(f"Raw response: {response}")
        return [], []
