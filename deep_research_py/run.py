import asyncio
import typer
from functools import wraps
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import os
import logging
from datetime import datetime
from pathlib import Path

from deep_research_py.deep_research import deep_research, write_final_report
from deep_research_py.feedback import generate_feedback
from deep_research_py.ai.providers import AIClientFactory
from deep_research_py.config import EnvironmentConfig
from deep_research_py.utils import setup_logging, logger

app = typer.Typer()
console = Console()
session = PromptSession()


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


async def async_prompt(message: str, default: str = "") -> str:
    """Async wrapper for prompt_toolkit."""
    return await session.prompt_async(message)


@app.command()
@coro
async def main(
    concurrency: int = typer.Option(
        default=2, help="Number of concurrent tasks, depending on your API rate limits."
    ),
    log_level: str = typer.Option(
        default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR"
    ),
    save_logs: bool = typer.Option(
        default=True, help="Save detailed logs to file"
    ),
    log_file: str = typer.Option(
        default=None, help="Custom log file path (optional)"
    ),
):
    """Deep Research CLI with comprehensive logging"""
    
    # Set up logging first
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO, 
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    log_file_path = setup_logging(
        log_level=log_level_map.get(log_level.upper(), logging.INFO),
        log_to_file=save_logs,
        log_file_path=log_file
    )
    
    # Log session start
    logger.info("🚀 Deep Research Session Started")
    logger.info("=" * 80)
    logger.info(f"📅 Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔧 Log Level: {log_level.upper()}")
    if log_file_path:
        logger.info(f"📁 Log File: {log_file_path}")
    logger.info(f"⚙️ Concurrency: {concurrency}")
    logger.info("=" * 80)
    
    console.print(
        Panel.fit(
            "[bold blue]Deep Research Assistant[/bold blue]\n"
            "[dim]An AI-powered research tool with goal-driven approach[/dim]"
        )
    )

    # Debug environment variables
    logger.debug("🔍 Environment Configuration:")
    logger.debug(f"   DEFAULT_SCRAPER = {os.getenv('DEFAULT_SCRAPER')}")
    logger.debug(f"   SERPER_API_KEY = {'SET' if os.getenv('SERPER_API_KEY') else 'NOT SET'}")
    logger.debug(f"   DEFAULT_SERVICE = {os.getenv('DEFAULT_SERVICE')}")
    logger.debug(f"   GEMINI_API_KEY = {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
    
    console.print(f"[yellow]DEBUG: DEFAULT_SCRAPER = {os.getenv('DEFAULT_SCRAPER')}[/yellow]")
    console.print(f"[yellow]DEBUG: SERPER_API_KEY = {'SET' if os.getenv('SERPER_API_KEY') else 'NOT SET'}[/yellow]")
    console.print(f"[yellow]DEBUG: DEFAULT_SERVICE = {os.getenv('DEFAULT_SERVICE')}[/yellow]")

    service = EnvironmentConfig.get_default_provider()
    console.print(f"🛠️ Using [bold green]{service.upper()}[/bold green] service.")
    logger.info(f"🛠️ AI Service Provider: {service.upper()}")

    client = AIClientFactory.get_client()
    model = AIClientFactory.get_model()
    
    console.print(f"🤖 Using model: [bold cyan]{model}[/bold cyan]")
    logger.info(f"🤖 AI Model: {model}")

    # Get initial inputs with clear formatting
    logger.info("📝 Collecting user inputs...")
    
    query = await async_prompt("\n🔍 What would you like to research? ")
    logger.info(f"🔍 Research Query: {query}")
    console.print()

    breadth_prompt = "📊 Research breadth (recommended 2-10) [4]: "
    breadth = int((await async_prompt(breadth_prompt)) or "4")
    logger.info(f"📊 Research Breadth: {breadth}")
    console.print()

    depth_prompt = "🔍 Research depth/max epochs (recommended 1-3) [2]: "
    depth = int((await async_prompt(depth_prompt)) or "2")
    logger.info(f"🔍 Research Depth: {depth}")
    console.print()

    # Generate follow-up questions and collect answers
    console.print("\n[yellow]Creating research plan...[/yellow]")
    logger.info("🤔 Generating follow-up questions...")
    
    follow_up_questions, answers = await generate_feedback(query, client, model)

    # Combine information for comprehensive research context
    combined_query = f"""
    Initial Query: {query}
    Follow-up Questions and Answers:
    {chr(10).join(f"Q: {q} A: {a}" for q, a in zip(follow_up_questions, answers))}
    """
    
    logger.info("📋 Research Context Created:")
    logger.info(f"   Initial Query: {query}")
    logger.info(f"   Follow-up Q&A pairs: {len(follow_up_questions)}")
    logger.debug(f"   Combined Query: {combined_query}")

    # Research phase with enhanced progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Perform goal-driven research
        task = progress.add_task(
            "[yellow]Performing goal-driven research...[/yellow]", total=None
        )
        
        logger.info("🔬 Starting goal-driven research...")
        research_start_time = datetime.now()
        
        research_results = await deep_research(
            query=combined_query,
            breadth=breadth,
            depth=depth,
            concurrency=concurrency,
            client=client,
            model=model,
            follow_up_questions=follow_up_questions,
            follow_up_answers=answers,
        )
        
        research_end_time = datetime.now()
        research_duration = research_end_time - research_start_time
        logger.info(f"⏱️ Research completed in {research_duration}")
        
        progress.remove_task(task)

        # Show enhanced results with goal achievement metrics
        console.print(f"\n[green]Research completed in {research_results['epochs_completed']} epochs![/green]")
        console.print(f"[yellow]Goal Achievement Score: {research_results['goal_alignment_score']:.2f}/1.0[/yellow]")
        
        logger.info("📊 Research Results Summary:")
        logger.info(f"   ⏱️ Duration: {research_duration}")
        logger.info(f"   🔄 Epochs: {research_results['epochs_completed']}")
        logger.info(f"   🎯 Goal Score: {research_results['goal_alignment_score']:.2f}")
        logger.info(f"   🎉 Goal Achieved: {research_results['goal_achieved']}")
        logger.info(f"   📚 Total Learnings: {len(research_results['learnings'])}")
        logger.info(f"   🔗 Total URLs: {len(research_results['visited_urls'])}")
        
        goal_status_color = "green" if research_results['goal_achieved'] else "yellow"
        goal_status_icon = "🎉" if research_results['goal_achieved'] else "⚠️"
        console.print(f"[{goal_status_color}]{goal_status_icon} Goal Achieved: {research_results['goal_achieved']}[/{goal_status_color}]")

        # Show learnings with numbering
        console.print(f"\n[yellow]📚 Research Learnings ({len(research_results['learnings'])}):[/yellow]")
        logger.info("📚 All Research Learnings:")
        for i, learning in enumerate(research_results["learnings"], 1):
            rprint(f"{i:2d}. {learning}")
            logger.info(f"   {i:2d}. {learning}")

        # Generate comprehensive report
        task = progress.add_task("Writing comprehensive report...", total=None)
        logger.info("📝 Generating final report...")
        
        report_start_time = datetime.now()
        report = await write_final_report(
            prompt=combined_query,
            learnings=research_results["learnings"],
            visited_urls=research_results["visited_urls"],
            client=client,
            model=model,
        )
        report_end_time = datetime.now()
        report_duration = report_end_time - report_start_time
        logger.info(f"📝 Report generated in {report_duration}")
        
        progress.remove_task(task)

        # Display final results
        console.print("\n[bold green]🎉 Research Complete![/bold green]")
        console.print("\n[yellow]📄 Final Report:[/yellow]")
        console.print(Panel(report, title="Research Report"))

        # Show sources with detailed logging
        console.print(f"\n[yellow]🔗 Sources ({len(research_results['visited_urls'])}):[/yellow]")
        
        # Log URL categories
        logger.info(f"📋 Final URL breakdown:")
        unique_domains = {}
        for url in research_results['visited_urls']:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                unique_domains[domain] = unique_domains.get(domain, 0) + 1
            except:
                domain = "unknown"
                unique_domains[domain] = unique_domains.get(domain, 0) + 1
        
        logger.info(f"🌐 Unique domains accessed: {len(unique_domains)}")
        for domain, count in sorted(unique_domains.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {domain}: {count} URLs")
        
        for i, url in enumerate(research_results["visited_urls"], 1):
            rprint(f"{i:2d}. {url}")
            logger.debug(f"Final source {i}: {url}")

        # Save report as plain text file with enhanced metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.txt"  # Changed from .md to .txt
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f"RESEARCH REPORT\n\n")
            f.write(f"Research Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Research Duration: {research_duration}\n")
            f.write(f"Goal Achievement Score: {research_results['goal_alignment_score']:.2f}/1.0\n")
            f.write(f"Goal Achieved: {research_results['goal_achieved']}\n")
            f.write(f"Epochs Completed: {research_results['epochs_completed']}\n")
            f.write(f"Total Learnings: {len(research_results['learnings'])}\n")
            f.write(f"Total Sources: {len(research_results['visited_urls'])}\n")
            f.write(f"Unique Domains: {len(unique_domains)}\n\n")
            
            # Add domain breakdown in plain text
            f.write("DOMAIN BREAKDOWN\n\n")
            for domain, count in sorted(unique_domains.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {domain}: {count} URLs\n")
            f.write("\n" + "="*50 + "\n\n")
            
            f.write(report)
            
        logger.info(f"📁 Report saved to: {output_file}")
        logger.info(f"📁 Report contains {len(research_results['visited_urls'])} source URLs")
        
        # Log session end
        total_session_time = datetime.now() - research_start_time
        logger.info("=" * 80)
        logger.info("🏁 Deep Research Session Completed")
        logger.info(f"⏱️ Total Session Time: {total_session_time}")
        logger.info(f"📄 Report File: {output_file}")
        if log_file_path:
            logger.info(f"📁 Log File: {log_file_path}")
        logger.info("=" * 80)
        
        console.print(f"\n[dim]📁 Report saved to {output_file}[/dim]")
        if log_file_path:
            console.print(f"[dim]📁 Detailed logs saved to {log_file_path}[/dim]")


def run():
    """Synchronous entry point for the CLI tool."""
    app()


if __name__ == "__main__":
    app()
