from dotenv import load_dotenv
import typer
from prompt_toolkit import PromptSession
from rich.console import Console
from enum import Enum
from typing import Dict, Any
import logging
from datetime import datetime

from deep_research_py.deep_research import deep_research, write_final_report
from deep_research_py.feedback import generate_feedback
from deep_research_py.ai.providers import AIClientFactory
from deep_research_py.config import EnvironmentConfig
from deep_research_py.utils import setup_logging, logger

from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import ChatInput, ChatResponse

load_dotenv()

# Set up logging for the web app
setup_logging(log_level=logging.INFO, log_to_file=True)

app = typer.Typer()
console = Console()
session = PromptSession()

kitchenai_app = KitchenAIApp(
    namespace="Deep Research",
)

class ResearchState(Enum):
    AWAITING_QUERY = "awaiting_query"
    AWAITING_BREADTH = "awaiting_breadth"
    AWAITING_DEPTH = "awaiting_depth"
    ASKING_QUESTIONS = "asking_questions"
    RESEARCHING = "researching"
    COMPLETE = "complete"

# Global state storage (you might want to use a proper database in production)
conversation_states: Dict[str, Dict[str, Any]] = {}

@kitchenai_app.chat.handler("chat.completions")
async def main(input: ChatInput) -> ChatResponse:
    # Log the incoming request
    logger.info("🌐 Web API Request Received")
    logger.debug(f"Input metadata: {input.metadata}")
    logger.debug(f"Input messages count: {len(input.messages)}")
    
    # Debug logging
    print("Input metadata:", input.metadata)
    print("Input messages:", [
        {
            "role": msg.role,
            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        } 
        for msg in input.messages
    ])
    
    # Get service and client using factory
    service = EnvironmentConfig.get_default_provider()
    client = AIClientFactory.get_client()
    model = AIClientFactory.get_model()
    
    logger.info(f"🤖 Using AI service: {service}, model: {model}")
    
    # Rest of conversation ID logic...
    conversation_id = None
    if input.metadata:
        conversation_id = input.metadata.get("conversation_id")
    
    if not conversation_id and input.messages:
        conversation_text = "".join(msg.content for msg in input.messages[:1])
        conversation_id = str(hash(conversation_text))
    
    if not conversation_id:
        conversation_id = "default"
    
    logger.debug(f"Conversation ID: {conversation_id}")
    
    current_message = input.messages[-1].content if input.messages else ""
    logger.debug(f"Current message: {current_message}")
    
    # Initialize or get existing state
    if conversation_id not in conversation_states:
        conversation_states[conversation_id] = {
            "state": ResearchState.AWAITING_QUERY,
            "query": None,
            "breadth": None,
            "depth": None,
            "questions": [],
            "answers": [],
            "current_question_idx": 0,
            "research_results": None
        }
        logger.info(f"🆕 New conversation started: {conversation_id}")
        return ChatResponse(
            content="🔍 What would you like to research?"
        )
    
    state_data = conversation_states[conversation_id]
    current_state = state_data["state"]
    logger.debug(f"Current state: {current_state}")
    
    # State machine for research flow
    if state_data["state"] == ResearchState.AWAITING_QUERY:
        state_data["query"] = current_message
        state_data["state"] = ResearchState.AWAITING_BREADTH
        logger.info(f"📝 Query set: {current_message}")
        return ChatResponse(
            content="📊 Research breadth (recommended 2-10) [4]: "
        )
        
    elif state_data["state"] == ResearchState.AWAITING_BREADTH:
        try:
            state_data["breadth"] = int(current_message or "4")
            state_data["state"] = ResearchState.AWAITING_DEPTH
            logger.info(f"📊 Breadth set: {state_data['breadth']}")
            return ChatResponse(
                content="🔍 Research depth (recommended 1-5) [2]: "
            )
        except ValueError:
            logger.warning(f"Invalid breadth input: {current_message}")
            return ChatResponse(
                content="Please enter a valid number for research breadth:"
            )
            
    elif state_data["state"] == ResearchState.AWAITING_DEPTH:
        try:
            state_data["depth"] = int(current_message or "2")
            logger.info(f"🔍 Depth set: {state_data['depth']}")
            
            # Generate follow-up questions using factory client
            logger.info("🤔 Generating follow-up questions...")
            state_data["questions"] = await generate_feedback(state_data["query"], client, model)
            state_data["state"] = ResearchState.ASKING_QUESTIONS
            
            return ChatResponse(
                content=f"[Q1] {state_data['questions'][0]}"
            )
            
        except ValueError:
            logger.warning(f"Invalid depth input: {current_message}")
            return ChatResponse(
                content="Please enter a valid number for research depth:"
            )
            
    elif state_data["state"] == ResearchState.ASKING_QUESTIONS:
        # Store the answer to the current question
        state_data["answers"].append(current_message)
        logger.info(f"💬 Answer {len(state_data['answers'])}: {current_message}")
        
        # Move to next question or start research
        if len(state_data["answers"]) < len(state_data["questions"]):
            next_q_idx = len(state_data["answers"])
            logger.debug(f"Moving to question {next_q_idx + 1}")
            return ChatResponse(
                content=f"[Q{next_q_idx + 1}] {state_data['questions'][next_q_idx]}"
            )
        else:
            state_data["state"] = ResearchState.RESEARCHING
            logger.info("🔬 Starting research phase...")
            
            # Combine information for research
            combined_query = f"""
            Initial Query: {state_data['query']}
            Follow-up Questions and Answers:
            {chr(10).join(f"Q: {q} A: {a}" for q, a in zip(state_data['questions'], state_data['answers']))}
            """
            
            logger.debug(f"Combined query: {combined_query}")
            
            # Perform research using factory client
            research_start_time = datetime.now()
            research_results = await deep_research(
                query=combined_query,
                breadth=state_data["breadth"],
                depth=state_data["depth"],
                concurrency=4,  # Make configurable
                client=client,
                model=model,
            )
            research_duration = datetime.now() - research_start_time
            logger.info(f"🔬 Research completed in {research_duration}")
            
            # Generate final report
            logger.info("📝 Generating final report...")
            report = await write_final_report(
                prompt=combined_query,
                learnings=research_results["learnings"],
                visited_urls=research_results["visited_urls"],
                client=client,
                model=model,
            )
            
            # Format final response
            final_response = f"""Research Complete!

Final Report:
{report}

Sources:
{chr(10).join(f"• {url}" for url in research_results['visited_urls'])}
"""
            
            logger.info("✅ Research session completed successfully")
            logger.info(f"📊 Results: {len(research_results['learnings'])} learnings, {len(research_results['visited_urls'])} sources")
            
            state_data["state"] = ResearchState.COMPLETE
            return ChatResponse(
                content=final_response
            )
    
    elif state_data["state"] == ResearchState.COMPLETE:
        # Reset state for new research
        logger.info("🔄 Resetting conversation for new research")
        conversation_states[conversation_id] = {
            "state": ResearchState.AWAITING_QUERY,
            "query": None,
            "breadth": None,
            "depth": None,
            "questions": [],
            "answers": [],
            "current_question_idx": 0,
            "research_results": None
        }
        return ChatResponse(
            content="Would you like to start a new research? What topic would you like to explore?"
        )
    
    # Fallback response
    logger.warning("🚨 Unexpected state, resetting conversation")
    return ChatResponse(
        content="I'm sorry, something went wrong. Let's start over. What would you like to research?"
    )


