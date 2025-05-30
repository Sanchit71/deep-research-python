from typing import List, Tuple
import openai
import json
from .prompt import system_prompt
from .ai.providers import get_client_response
from deep_research_py.utils import logger
from prompt_toolkit import PromptSession


def enhanced_feedback_generation_prompt(query: str) -> str:
    """Enhanced prompt for generating follow-up questions."""
    return f"""TASK: Generate strategic follow-up questions to optimize research planning and goal setting.

RESEARCH TOPIC: {query}

OBJECTIVE: Create 3-5 targeted questions that will help clarify the user's specific research needs, scope, and success criteria. These questions will be used to generate a comprehensive research goal framework.

QUESTION CATEGORIES TO CONSIDER:

1. **SCOPE AND FOCUS**:
   - What specific aspects or subtopics are most important?
   - What level of detail or depth is required?
   - Are there particular timeframes, regions, or contexts of interest?

2. **PURPOSE AND APPLICATION**:
   - How will this research be used or applied?
   - Who is the target audience for the findings?
   - What decisions or actions will this research inform?

3. **INFORMATION PRIORITIES**:
   - What types of information are most valuable (technical, commercial, academic, practical)?
   - Are there specific questions that must be answered?
   - What would constitute a successful research outcome?

4. **CONSTRAINTS AND PREFERENCES**:
   - Are there particular sources, perspectives, or approaches to prioritize?
   - What information gaps or challenges have been encountered before?
   - Are there any limitations or constraints to consider?

QUESTION QUALITY STANDARDS:
- **Specific**: Target particular aspects rather than asking general questions
- **Actionable**: Answers should directly inform research strategy
- **Clarifying**: Help distinguish between different research approaches
- **Goal-oriented**: Support the creation of measurable success criteria
- **Open-ended**: Allow for detailed, informative responses

OUTPUT REQUIREMENTS:
Generate a JSON object with a "questions" array containing 3-5 strategic questions.

EXAMPLE STRUCTURE:
```json
{{
  "questions": [
    "What specific aspects of [topic] are you most interested in exploring (e.g., technical implementation, market trends, regulatory considerations)?",
    "What is the primary purpose of this research - are you looking to make a decision, understand current state, or identify opportunities?",
    "What level of technical depth do you need - high-level overview, detailed analysis, or implementation-ready information?",
    "Are there particular timeframes, geographic regions, or industry sectors that should be the focus?",
    "What would make this research most valuable to you - specific metrics, case studies, expert opinions, or comparative analysis?"
  ]
}}
```

STRATEGIC CONSIDERATIONS:
- Questions should help identify the most valuable research directions
- Focus on clarifying ambiguity in the original query
- Help establish measurable success criteria
- Consider different types of information needs
- Support efficient resource allocation during research

Generate the strategic follow-up questions now:"""


async def generate_feedback(query: str, client: openai.OpenAI, model: str) -> Tuple[List[str], List[str]]:
    """Generates follow-up questions and collects answers to clarify research direction."""

    logger.info("🤔 Generating enhanced follow-up questions for research clarification")
    logger.debug(f"Initial query: {query}")

    # Use enhanced prompt
    prompt = enhanced_feedback_generation_prompt(query)

    logger.info("📝 ENHANCED FOLLOW-UP QUESTIONS GENERATION PROMPT:")
    logger.info("=" * 80)
    logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    logger.info("=" * 80)

    response = await get_client_response(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    # Parse the JSON response
    try:
        questions = response.get("questions", [])
        logger.info(f"✅ Generated {len(questions)} enhanced follow-up questions")
        
        for i, question in enumerate(questions, 1):
            logger.info(f"   ❓ Question {i}: {question}")
        
        # Collect answers interactively
        session = PromptSession()
        answers = []
        
        print("\n[bold yellow]Strategic Research Planning Questions:[/bold yellow]")
        print("[dim]These questions will help create a focused research goal and strategy.[/dim]\n")
        
        for i, question in enumerate(questions, 1):
            logger.debug(f"Asking enhanced question {i}: {question}")
            print(f"[bold blue]Q{i}:[/bold blue] {question}")
            answer = await session.prompt_async("➤ Your answer: ")
            answers.append(answer)
            logger.info(f"   💬 Answer {i}: {answer}")
            print()
        
        logger.info(f"✅ Collected {len(answers)} strategic answers")
        logger.info("📋 Complete Enhanced Q&A Summary:")
        for i, (q, a) in enumerate(zip(questions, answers), 1):
            logger.info(f"   Q{i}: {q}")
            logger.info(f"   A{i}: {a}")
        
        return questions, answers
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing JSON response: {e}")
        logger.debug(f"Raw response: {response}")
        
        # Enhanced fallback questions
        fallback_questions = [
            "What specific aspects of this topic are most important for your research goals?",
            "What is the primary purpose of this research and how will you use the findings?",
            "What level of detail do you need - overview, detailed analysis, or implementation guidance?",
            "Are there particular timeframes, regions, or contexts that should be prioritized?",
            "What would constitute a successful research outcome for your needs?"
        ]
        
        logger.warning("🔄 Using enhanced fallback questions")
        return fallback_questions, []
