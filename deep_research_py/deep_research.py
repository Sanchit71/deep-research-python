from typing import List, Dict, TypedDict, Optional
from dataclasses import dataclass
import asyncio
import openai
from deep_research_py.data_acquisition.services import get_global_search_service
from .ai.providers import trim_prompt, get_client_response
from .prompt import system_prompt
from deep_research_py.utils import logger
import json


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]
    goal_alignment_score: float
    epochs_completed: int
    goal_achieved: bool


@dataclass
class SerpQuery:
    query: str
    research_goal: str


@dataclass
class UserGoal:
    primary_objective: str
    success_criteria: List[str]
    specific_questions: List[str]


async def generate_user_goal(
    initial_query: str,
    follow_up_answers: List[str],
    follow_up_questions: List[str],
    client: openai.OpenAI,
    model: str,
) -> UserGoal:
    """Generate a structured user goal based on initial query and follow-up Q&A."""
    
    logger.info("🎯 Generating structured user goal from initial query and follow-up Q&A")
    logger.debug(f"Initial query: {initial_query}")
    logger.debug(f"Follow-up questions: {follow_up_questions}")
    logger.debug(f"Follow-up answers: {follow_up_answers}")
    
    qa_pairs = "\n".join([
        f"Q: {q}\nA: {a}" 
        for q, a in zip(follow_up_questions, follow_up_answers)
    ])
    
    prompt = f"""
    Based on the initial research query and follow-up Q&A, define a clear research goal with success criteria.
    
    Initial Query: {initial_query}
    
    Follow-up Q&A:
    {qa_pairs}
    
    Generate a JSON object with:
    - "primary_objective": A clear, specific research objective (1-2 sentences)
    - "success_criteria": List of 3-5 specific criteria that would indicate successful research completion
    - "specific_questions": List of 3-5 specific questions that need to be answered to achieve the goal
    
    Make the goal SMART (Specific, Measurable, Achievable, Relevant, Time-bound where applicable).
    Focus on creating actionable, measurable criteria that can be evaluated objectively.
    """
    
    logger.debug(f"Sending goal generation prompt to {model}")
    
    try:
        response = await get_client_response(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        
        logger.debug(f"Raw goal generation response: {response}")
        
        user_goal = UserGoal(
            primary_objective=response.get("primary_objective", ""),
            success_criteria=response.get("success_criteria", []),
            specific_questions=response.get("specific_questions", [])
        )
        
        logger.info(f"✅ Generated user goal successfully")
        logger.info(f"📋 Primary Objective: {user_goal.primary_objective}")
        logger.info(f"📊 Success Criteria ({len(user_goal.success_criteria)}): {user_goal.success_criteria}")
        logger.info(f"❓ Specific Questions ({len(user_goal.specific_questions)}): {user_goal.specific_questions}")
        
        return user_goal
        
    except Exception as e:
        logger.error(f"❌ Error generating user goal: {e}")
        fallback_goal = UserGoal(
            primary_objective=initial_query,
            success_criteria=["Find relevant information"],
            specific_questions=["What are the key findings?"]
        )
        logger.warning(f"🔄 Using fallback goal: {fallback_goal.primary_objective}")
        return fallback_goal


async def evaluate_goal_alignment(
    user_goal: UserGoal,
    current_learnings: List[str],
    epoch: int,
    client: openai.OpenAI,
    model: str,
) -> Dict[str, any]:
    """Evaluate how well current learnings align with the user's goal."""
    
    logger.info(f"🔍 Evaluating goal alignment for epoch {epoch}")
    logger.debug(f"Current learnings count: {len(current_learnings)}")
    logger.debug(f"User goal criteria count: {len(user_goal.success_criteria)}")
    logger.debug(f"User goal questions count: {len(user_goal.specific_questions)}")
    
    learnings_text = "\n".join([f"- {learning}" for learning in current_learnings])
    
    prompt = f"""
    Evaluate how well the current research learnings align with the user's research goal.
    
    USER GOAL:
    Primary Objective: {user_goal.primary_objective}
    
    Success Criteria:
    {chr(10).join([f"- {criteria}" for criteria in user_goal.success_criteria])}
    
    Specific Questions to Answer:
    {chr(10).join([f"- {question}" for question in user_goal.specific_questions])}
    
    CURRENT LEARNINGS (Epoch {epoch}):
    {learnings_text}
    
    Provide a JSON response with:
    - "alignment_score": Float between 0.0-1.0 indicating how well learnings address the goal
    - "criteria_met": List of success criteria that have been adequately addressed
    - "questions_answered": List of specific questions that have been answered
    - "missing_aspects": List of important aspects still missing from the research
    - "goal_achieved": Boolean indicating if the research goal is sufficiently achieved
    - "continue_research": Boolean indicating if more research is needed
    - "next_research_directions": List of 2-3 specific areas to focus on if continuing research
    
    Consider the research goal achieved (goal_achieved: true) if:
    - Alignment score is >= 0.8
    - At least 80% of success criteria are met
    - At least 80% of specific questions are answered
    
    Be thorough in your evaluation and provide specific reasoning for your assessment.
    """
    
    logger.debug(f"Sending goal alignment evaluation to {model}")
    
    try:
        response = await get_client_response(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        
        logger.debug(f"Raw goal alignment response: {response}")
        
        evaluation = {
            "alignment_score": response.get("alignment_score", 0.0),
            "criteria_met": response.get("criteria_met", []),
            "questions_answered": response.get("questions_answered", []),
            "missing_aspects": response.get("missing_aspects", []),
            "goal_achieved": response.get("goal_achieved", False),
            "continue_research": response.get("continue_research", True),
            "next_research_directions": response.get("next_research_directions", [])
        }
        
        logger.info(f"📊 Goal Alignment Results for Epoch {epoch}:")
        logger.info(f"   🎯 Alignment Score: {evaluation['alignment_score']:.2f}/1.0")
        logger.info(f"   ✅ Criteria Met: {len(evaluation['criteria_met'])}/{len(user_goal.success_criteria)}")
        logger.info(f"   ❓ Questions Answered: {len(evaluation['questions_answered'])}/{len(user_goal.specific_questions)}")
        logger.info(f"   🎉 Goal Achieved: {evaluation['goal_achieved']}")
        
        if evaluation['criteria_met']:
            logger.debug(f"✅ Met criteria: {evaluation['criteria_met']}")
        if evaluation['questions_answered']:
            logger.debug(f"❓ Answered questions: {evaluation['questions_answered']}")
        if evaluation['missing_aspects']:
            logger.info(f"❌ Missing aspects: {evaluation['missing_aspects']}")
        if evaluation['next_research_directions']:
            logger.info(f"🔄 Next research directions: {evaluation['next_research_directions']}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"❌ Error evaluating goal alignment: {e}")
        fallback_evaluation = {
            "alignment_score": 0.5,
            "criteria_met": [],
            "questions_answered": [],
            "missing_aspects": ["Unable to evaluate"],
            "goal_achieved": False,
            "continue_research": True,
            "next_research_directions": ["Continue general research"]
        }
        logger.warning(f"🔄 Using fallback evaluation with score 0.5")
        return fallback_evaluation


async def generate_serp_queries(
    query: str,
    client: openai.OpenAI,
    model: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Generate SERP queries based on user input and previous learnings."""

    logger.info(f"🔍 Generating {num_queries} SERP queries")
    logger.debug(f"Base query: {query}")
    logger.debug(f"Previous learnings count: {len(learnings) if learnings else 0}")

    prompt = f"""Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a JSON object with a 'queries' array field containing {num_queries} queries (or less if the original prompt is clear). Each query object should have 'query' and 'research_goal' fields. Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>"""

    if learnings:
        recent_learnings = learnings[-5:]  # Use only recent learnings to avoid prompt bloat
        prompt += f"\n\nHere are some recent learnings from previous research, use them to generate more specific and targeted queries: {' '.join(recent_learnings)}"
        logger.debug(f"Including {len(recent_learnings)} recent learnings in query generation")

    logger.debug(f"Sending SERP query generation to {model}")

    try:
        response = await get_client_response(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        logger.debug(f"Raw SERP queries response: {response}")

        queries = response.get("queries", [])
        serp_queries = [SerpQuery(**q) for q in queries][:num_queries]
        
        logger.info(f"✅ Generated {len(serp_queries)} SERP queries:")
        for i, sq in enumerate(serp_queries, 1):
            logger.info(f"   {i}. Query: '{sq.query}' | Goal: {sq.research_goal}")
        
        return serp_queries
        
    except Exception as e:
        logger.error(f"❌ Error generating SERP queries: {e}")
        fallback_query = SerpQuery(query=query, research_goal="General research")
        logger.warning(f"🔄 Using fallback query: {fallback_query.query}")
        return [fallback_query]


async def process_serp_result(
    query: str,
    search_result: SearchResponse,
    client: openai.OpenAI,
    model: str,
    num_learnings: int = 3,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    logger.info(f"📚 Processing SERP results for query: '{query}'")
    logger.debug(f"Search results count: {len(search_result['data'])}")
    logger.debug(f"Target learnings: {num_learnings}, Target follow-up questions: {num_follow_up_questions}")

    # Log all URLs found in search results
    urls_found = []
    for i, item in enumerate(search_result['data'], 1):
        url = item.get("url", "")
        title = item.get("title", "No title")
        if url:
            urls_found.append(url)
            logger.info(f"   🔗 Result {i}: {url}")
            logger.debug(f"      📄 Title: {title}")
        else:
            logger.warning(f"   ⚠️ Result {i}: No URL found")
    
    logger.info(f"📊 Found {len(urls_found)} URLs for processing")

    contents = [
        trim_prompt(item.get("content", ""), 25_000)
        for item in search_result["data"]
        if item.get("content")
    ]

    logger.debug(f"Content pieces to process: {len(contents)}")
    total_content_length = sum(len(content) for content in contents)
    logger.debug(f"Total content length: {total_content_length} characters")

    # Log which URLs have content vs which don't
    content_urls = []
    no_content_urls = []
    for item in search_result["data"]:
        url = item.get("url", "")
        if url:
            if item.get("content"):
                content_urls.append(url)
                logger.debug(f"   ✅ Content available: {url}")
            else:
                no_content_urls.append(url)
                logger.warning(f"   ❌ No content: {url}")
    
    if content_urls:
        logger.info(f"📄 URLs with content ({len(content_urls)}):")
        for url in content_urls:
            logger.info(f"   ✅ {url}")
    
    if no_content_urls:
        logger.warning(f"⚠️ URLs without content ({len(no_content_urls)}):")
        for url in no_content_urls:
            logger.warning(f"   ❌ {url}")

    # Create the contents string separately
    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    prompt = (
        f"Given the following contents from a SERP search for the query <query>{query}</query>, "
        f"generate a list of learnings from the contents. Return a JSON object with 'learnings' "
        f"and 'followUpQuestions' keys with array of strings as values. Include up to {num_learnings} learnings and "
        f"{num_follow_up_questions} follow-up questions. The learnings should be unique, "
        "concise, and information-dense, including entities, metrics, numbers, and dates. "
        "Focus on actionable insights and specific findings rather than general statements.\n\n"
        f"<contents>{contents_str}</contents>"
    )

    logger.debug(f"Sending content processing to {model}")

    try:
        response = await get_client_response(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        logger.debug(f"Raw content processing response: {response}")

        result = {
            "learnings": response.get("learnings", [])[:num_learnings],
            "followUpQuestions": response.get("followUpQuestions", [])[:num_follow_up_questions],
        }
        
        logger.info(f"✅ Extracted {len(result['learnings'])} learnings and {len(result['followUpQuestions'])} follow-up questions from {len(content_urls)} URLs")
        
        for i, learning in enumerate(result['learnings'], 1):
            logger.debug(f"   Learning {i}: {learning}")
        
        for i, question in enumerate(result['followUpQuestions'], 1):
            logger.debug(f"   Follow-up {i}: {question}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing SERP result: {e}")
        logger.error(f"   Failed URLs: {urls_found}")
        return {"learnings": [], "followUpQuestions": []}


async def goal_driven_research(
    user_goal: UserGoal,
    breadth: int,
    max_epochs: int,
    concurrency: int,
    client: openai.OpenAI,
    model: str,
) -> ResearchResult:
    """
    Perform goal-driven research that continues until the user's goal is achieved or max epochs reached.
    """
    
    logger.info("🚀 Starting goal-driven research")
    logger.info(f"🎯 Research Goal: {user_goal.primary_objective}")
    logger.info(f"📊 Configuration: Max Epochs={max_epochs}, Breadth={breadth}, Concurrency={concurrency}")
    logger.info(f"🤖 Using AI Model: {model}")
    
    all_learnings = []
    all_urls = []
    epoch = 0
    goal_achieved = False
    evaluation = None
    
    print(f"\n🎯 Research Goal: {user_goal.primary_objective}")
    print(f"📊 Max Epochs: {max_epochs}, Breadth: {breadth}")
    
    while epoch < max_epochs and not goal_achieved:
        epoch += 1
        logger.info(f"🔄 Starting Epoch {epoch}/{max_epochs}")
        print(f"\n🔄 Starting Epoch {epoch}/{max_epochs}")
        
        # Generate research queries based on current state
        if epoch == 1:
            # First epoch: focus on primary objective
            query_context = user_goal.primary_objective
            logger.info("📋 First epoch: focusing on primary objective")
        else:
            # Subsequent epochs: focus on missing aspects
            logger.info("🔍 Subsequent epoch: focusing on missing aspects from previous evaluation")
            evaluation = await evaluate_goal_alignment(
                user_goal, all_learnings, epoch - 1, client, model
            )
            query_context = f"""
            Primary Goal: {user_goal.primary_objective}
            Missing Aspects: {', '.join(evaluation['missing_aspects'])}
            Next Research Directions: {', '.join(evaluation['next_research_directions'])}
            """
            logger.debug(f"Query context for epoch {epoch}: {query_context}")
        
        # Generate search queries for this epoch
        logger.info(f"🔍 Generating {breadth} search queries for epoch {epoch}")
        serp_queries = await generate_serp_queries(
            query=query_context,
            client=client,
            model=model,
            num_queries=breadth,
            learnings=all_learnings[-10:] if all_learnings else None  # Use recent learnings
        )
        
        # Perform searches and extract learnings
        epoch_learnings = []
        epoch_urls = []
        
        # Create semaphores for rate limiting
        api_semaphore = asyncio.Semaphore(1)
        search_semaphore = asyncio.Semaphore(concurrency)
        
        logger.info(f"⚙️ Processing {len(serp_queries)} queries with concurrency={concurrency}")
        
        async def process_query(serp_query: SerpQuery, query_index: int):
            logger.info(f"🔍 Processing query {query_index + 1}/{len(serp_queries)}: '{serp_query.query}'")
            
            async with search_semaphore:
                try:
                    # Search for content
                    search_service = get_global_search_service()
                    logger.debug(f"Searching with limit=5 for query: {serp_query.query}")
                    result = await search_service.search(serp_query.query, limit=5)
                    
                    # Collect and log URLs
                    new_urls = [
                        item.get("url") for item in result["data"] if item.get("url")
                    ]
                    
                    logger.info(f"🔗 Query {query_index + 1} found {len(new_urls)} URLs:")
                    for i, url in enumerate(new_urls, 1):
                        logger.info(f"   {i}. {url}")
                        # Check if this URL was already processed
                        if url in all_urls:
                            logger.debug(f"      ♻️ Already processed in previous epoch")
                        else:
                            logger.debug(f"      🆕 New URL for processing")
                    
                    # Log duplicate URLs within this query
                    unique_new_urls = list(dict.fromkeys(new_urls))
                    if len(unique_new_urls) != len(new_urls):
                        logger.warning(f"   ⚠️ Found {len(new_urls) - len(unique_new_urls)} duplicate URLs in query results")
                    
                    # Process search results with API rate limiting
                    async with api_semaphore:
                        logger.debug(f"Processing content for query {query_index + 1}")
                        new_learnings = await process_serp_result(
                            query=serp_query.query,
                            search_result=result,
                            num_learnings=5,  # More learnings per query
                            num_follow_up_questions=2,
                            client=client,
                            model=model,
                        )
                        await asyncio.sleep(1)  # Rate limiting
                        logger.debug(f"Completed processing query {query_index + 1}")
                    
                    return {
                        "learnings": new_learnings["learnings"],
                        "urls": unique_new_urls,  # Use deduplicated URLs
                        "query_index": query_index
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error processing query {query_index + 1} in epoch {epoch}: {e}")
                    return {"learnings": [], "urls": [], "query_index": query_index}
        
        # Process all queries for this epoch
        logger.info(f"🔄 Processing all {len(serp_queries)} queries concurrently")
        results = await asyncio.gather(*[
            process_query(query, i) for i, query in enumerate(serp_queries)
        ])
        
        # Collect epoch results and log URL statistics
        epoch_new_urls = []
        epoch_duplicate_urls = []
        
        for result in results:
            epoch_learnings.extend(result["learnings"])
            
            # Track new vs duplicate URLs
            for url in result["urls"]:
                if url in all_urls:
                    epoch_duplicate_urls.append(url)
                else:
                    epoch_new_urls.append(url)
            
            epoch_urls.extend(result["urls"])
            logger.debug(f"Query {result['query_index'] + 1} contributed {len(result['learnings'])} learnings and {len(result['urls'])} URLs")
        
        # Log epoch URL statistics
        logger.info(f"📊 Epoch {epoch} URL Statistics:")
        logger.info(f"   🆕 New URLs: {len(epoch_new_urls)}")
        logger.info(f"   ♻️ Duplicate URLs: {len(epoch_duplicate_urls)}")
        logger.info(f"   📝 Total URLs this epoch: {len(epoch_urls)}")
        
        if epoch_new_urls:
            logger.info(f"🆕 New URLs discovered in epoch {epoch}:")
            for i, url in enumerate(epoch_new_urls, 1):
                logger.info(f"   {i}. {url}")
        
        if epoch_duplicate_urls:
            logger.debug(f"♻️ Duplicate URLs in epoch {epoch}:")
            for i, url in enumerate(set(epoch_duplicate_urls), 1):
                count = epoch_duplicate_urls.count(url)
                logger.debug(f"   {i}. {url} (seen {count} times)")
        
        # Add to overall results
        all_learnings.extend(epoch_learnings)
        all_urls.extend(epoch_urls)
        
        logger.info(f"📚 Epoch {epoch} completed: {len(epoch_learnings)} new learnings, {len(epoch_urls)} new URLs")
        logger.info(f"📊 Total accumulated: {len(all_learnings)} learnings, {len(all_urls)} URLs")
        print(f"📚 Epoch {epoch} completed: {len(epoch_learnings)} new learnings")
        
        # Evaluate goal alignment
        logger.info(f"🎯 Evaluating goal alignment after epoch {epoch}")
        evaluation = await evaluate_goal_alignment(
            user_goal, all_learnings, epoch, client, model
        )
        
        alignment_score = evaluation["alignment_score"]
        goal_achieved = evaluation["goal_achieved"]
        
        logger.info(f"📊 Epoch {epoch} Results:")
        logger.info(f"   🎯 Goal Alignment Score: {alignment_score:.2f}")
        logger.info(f"   ✅ Criteria Met: {len(evaluation['criteria_met'])}/{len(user_goal.success_criteria)}")
        logger.info(f"   ❓ Questions Answered: {len(evaluation['questions_answered'])}/{len(user_goal.specific_questions)}")
        logger.info(f"   🎉 Goal Achieved: {goal_achieved}")
        
        print(f"🎯 Goal Alignment Score: {alignment_score:.2f}")
        print(f"✅ Criteria Met: {len(evaluation['criteria_met'])}/{len(user_goal.success_criteria)}")
        print(f"❓ Questions Answered: {len(evaluation['questions_answered'])}/{len(user_goal.specific_questions)}")
        
        if goal_achieved:
            logger.info(f"🎉 Research goal achieved in epoch {epoch}!")
            print(f"🎉 Research goal achieved in epoch {epoch}!")
            break
        elif epoch < max_epochs:
            missing_aspects = evaluation['missing_aspects'][:3]
            logger.info(f"🔄 Continuing research. Missing: {', '.join(missing_aspects)}")
            print(f"🔄 Continuing research. Missing: {', '.join(missing_aspects)}")
        else:
            logger.info(f"⏰ Reached maximum epochs ({max_epochs}) without achieving goal")
            print(f"⏰ Reached maximum epochs ({max_epochs}) without achieving goal")
    
    # Remove duplicates and log final statistics
    unique_learnings = list(dict.fromkeys(all_learnings))
    unique_urls = list(dict.fromkeys(all_urls))
    
    # Log URL deduplication statistics
    total_url_duplicates = len(all_urls) - len(unique_urls)
    logger.info(f"🔗 Final URL Statistics:")
    logger.info(f"   📊 Total URLs collected: {len(all_urls)}")
    logger.info(f"   🆕 Unique URLs: {len(unique_urls)}")
    logger.info(f"   ♻️ Duplicate URLs removed: {total_url_duplicates}")
    
    if unique_urls:
        logger.info(f"📋 Final unique URLs list:")
        for i, url in enumerate(unique_urls, 1):
            logger.info(f"   {i:2d}. {url}")
    
    logger.info(f"🏁 Research completed:")
    logger.info(f"   📚 Total unique learnings: {len(unique_learnings)}")
    logger.info(f"   🔗 Total unique URLs: {len(unique_urls)}")
    logger.info(f"   🔄 Epochs completed: {epoch}")
    logger.info(f"   🎯 Final alignment score: {evaluation.get('alignment_score', 0.0) if evaluation else 0.0:.2f}")
    logger.info(f"   🎉 Goal achieved: {goal_achieved}")
    
    return {
        "learnings": unique_learnings,
        "visited_urls": unique_urls,
        "goal_alignment_score": evaluation.get("alignment_score", 0.0) if evaluation else 0.0,
        "epochs_completed": epoch,
        "goal_achieved": goal_achieved
    }


# Replace the existing deep_research function with this goal-driven version
async def deep_research(
    query: str,
    breadth: int,
    depth: int,  # Keep for backward compatibility, but convert to max_epochs
    concurrency: int,
    client: openai.OpenAI,
    model: str,
    follow_up_questions: List[str] = None,
    follow_up_answers: List[str] = None,
) -> ResearchResult:
    """
    Main research function that uses goal-driven approach.
    
    Args:
        query: Initial research query
        breadth: Number of parallel searches per epoch
        depth: Converted to max_epochs (depth + 1)
        concurrency: Number of concurrent operations
        client: AI client
        model: AI model
        follow_up_questions: List of follow-up questions
        follow_up_answers: List of answers to follow-up questions
    """
    
    logger.info("🚀 Starting deep research with goal-driven approach")
    logger.debug(f"Input parameters: query='{query}', breadth={breadth}, depth={depth}, concurrency={concurrency}")
    logger.debug(f"Follow-up Q&A provided: {bool(follow_up_questions and follow_up_answers)}")
    
    # Convert depth to max_epochs (with minimum of 1, maximum of 3)
    max_epochs = min(max(depth, 1), 3)
    logger.info(f"📊 Converted depth {depth} to max_epochs {max_epochs}")
    
    # Generate user goal if follow-up Q&A is provided
    if follow_up_questions and follow_up_answers:
        logger.info("🎯 Generating structured user goal from follow-up Q&A")
        user_goal = await generate_user_goal(
            query, follow_up_answers, follow_up_questions, client, model
        )
    else:
        logger.info("🎯 Using fallback goal (no follow-up Q&A provided)")
        user_goal = UserGoal(
            primary_objective=query,
            success_criteria=["Find comprehensive information about the topic"],
            specific_questions=["What are the key findings and latest developments?"]
        )
        logger.debug(f"Fallback goal: {user_goal.primary_objective}")
    
    # Perform goal-driven research
    logger.info("🔄 Starting goal-driven research process")
    return await goal_driven_research(
        user_goal=user_goal,
        breadth=breadth,
        max_epochs=max_epochs,
        concurrency=concurrency,
        client=client,
        model=model
    )


async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client: openai.OpenAI,
    model: str,
) -> str:
    """Generate final report based on all research learnings."""

    logger.info("📝 Generating final research report")
    logger.debug(f"Report input: {len(learnings)} learnings, {len(visited_urls)} URLs")
    logger.debug(f"Using model: {model}")

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )

    user_prompt = (
        f"Given the following prompt from the user, write a final report on the topic using "
        f"the learnings from research. Return a JSON object with a 'reportMarkdown' field "
        f"containing a detailed markdown report (aim for 3+ pages). Include ALL the learnings "
        f"from research:\n\n<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )

    logger.debug(f"Report generation prompt length: {len(user_prompt)} characters")

    try:
        logger.info("🤖 Sending report generation request to AI model")
        response = await get_client_response(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        logger.debug(f"Raw report generation response received")
        report = response.get("reportMarkdown", "")
        
        if report:
            logger.info(f"✅ Successfully generated report ({len(report)} characters)")
        else:
            logger.warning("⚠️ Empty report generated, using fallback")

        # Append sources with enhanced logging
        logger.info(f"📎 Appending {len(visited_urls)} source URLs to report")
        urls_section = "\n\n## Sources\n\n" + "\n".join(
            [f"- {url}" for url in visited_urls]
        )
        
        final_report = report + urls_section
        logger.info(f"📄 Final report length: {len(final_report)} characters")
        
        return final_report
        
    except Exception as e:
        logger.error(f"❌ Error generating final report: {e}")
        logger.info("🔄 Generating fallback report")
        
        # Fallback report with enhanced structure
        fallback_report = f"""# Research Report

## Executive Summary
Research was conducted on: {prompt}

## Key Findings
""" + "\n".join([f"- {learning}" for learning in learnings[:20]])  # Limit to top 20 learnings
        
        if len(learnings) > 20:
            fallback_report += f"\n\n*Note: {len(learnings) - 20} additional learnings were discovered during research.*"
        
        urls_section = "\n\n## Sources\n\n" + "\n".join(
            [f"- {url}" for url in visited_urls]
        )
        
        final_fallback = fallback_report + urls_section
        logger.info(f"📄 Fallback report generated ({len(final_fallback)} characters)")
        
        return final_fallback
