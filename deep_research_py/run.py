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

from deep_research_py.deep_research import deep_research, write_final_report
from deep_research_py.feedback import generate_feedback
from deep_research_py.ai.providers import AIClientFactory
from deep_research_py.config import EnvironmentConfig

app = typer.Typer()
console = Console()
session = PromptSession()
logger = logging.getLogger(__name__)


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
):
    """Deep Research CLI"""
    
    # Debug environment variables
    console.print(f"[yellow]DEBUG: DEFAULT_SCRAPER = {os.getenv('DEFAULT_SCRAPER')}[/yellow]")
    console.print(f"[yellow]DEBUG: SERPER_API_KEY = {'SET' if os.getenv('SERPER_API_KEY') else 'NOT SET'}[/yellow]")
    console.print(f"[yellow]DEBUG: DEFAULT_SERVICE = {os.getenv('DEFAULT_SERVICE')}[/yellow]")
    
    console.print(
        Panel.fit(
            "[bold blue]Deep Research Assistant[/bold blue]\n"
            "[dim]An AI-powered research tool with goal-driven approach[/dim]"
        )
    )

    service = EnvironmentConfig.get_default_provider()
    console.print(f"🛠️ Using [bold green]{service.upper()}[/bold green] service.")

    client = AIClientFactory.get_client()
    model = AIClientFactory.get_model()
    
    console.print(f"🤖 Using model: [bold cyan]{model}[/bold cyan]")

    # Get initial inputs with clear formatting
    query = await async_prompt("\n🔍 What would you like to research? ")
    console.print()

    breadth_prompt = "📊 Research breadth (recommended 2-10) [4]: "
    breadth = int((await async_prompt(breadth_prompt)) or "4")
    console.print()

    depth_prompt = "🔍 Research depth/max epochs (recommended 1-3) [2]: "
    depth = int((await async_prompt(depth_prompt)) or "2")
    console.print()

    # Generate follow-up questions and collect answers
    console.print("\n[yellow]Creating research plan...[/yellow]")
    follow_up_questions, answers = await generate_feedback(query, client, model)

    # Combine information for comprehensive research context
    combined_query = f"""
    Initial Query: {query}
    Follow-up Questions and Answers:
    {chr(10).join(f"Q: {q} A: {a}" for q, a in zip(follow_up_questions, answers))}
    """

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
        progress.remove_task(task)

        # Show enhanced results with goal achievement metrics
        console.print(f"\n[green]Research completed in {research_results['epochs_completed']} epochs![/green]")
        console.print(f"[yellow]Goal Achievement Score: {research_results['goal_alignment_score']:.2f}/1.0[/yellow]")
        
        goal_status_color = "green" if research_results['goal_achieved'] else "yellow"
        goal_status_icon = "🎉" if research_results['goal_achieved'] else "⚠️"
        console.print(f"[{goal_status_color}]{goal_status_icon} Goal Achieved: {research_results['goal_achieved']}[/{goal_status_color}]")

        # Show learnings with numbering
        console.print(f"\n[yellow]📚 Research Learnings ({len(research_results['learnings'])}):[/yellow]")
        for i, learning in enumerate(research_results["learnings"], 1):
            rprint(f"{i:2d}. {learning}")

        # Generate comprehensive report
        task = progress.add_task("Writing comprehensive report...", total=None)
        report = await write_final_report(
            prompt=combined_query,
            learnings=research_results["learnings"],
            visited_urls=research_results["visited_urls"],
            client=client,
            model=model,
        )
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

        # Save report with enhanced URL metadata
        with open("output.md", "w") as f:
            f.write(f"# Research Report\n\n")
            f.write(f"**Goal Achievement Score:** {research_results['goal_alignment_score']:.2f}/1.0\n")
            f.write(f"**Goal Achieved:** {research_results['goal_achieved']}\n")
            f.write(f"**Epochs Completed:** {research_results['epochs_completed']}\n")
            f.write(f"**Total Learnings:** {len(research_results['learnings'])}\n")
            f.write(f"**Total Sources:** {len(research_results['visited_urls'])}\n")
            f.write(f"**Unique Domains:** {len(unique_domains)}\n\n")
            
            # Add domain breakdown
            f.write("## Domain Breakdown\n\n")
            for domain, count in sorted(unique_domains.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{domain}**: {count} URLs\n")
            f.write("\n---\n\n")
            
            f.write(report)
            
        logger.info(f"📁 Report saved with {len(research_results['visited_urls'])} source URLs")
        console.print("\n[dim]📁 Report has been saved to output.md with URL metadata[/dim]")


def run():
    """Synchronous entry point for the CLI tool."""
    asyncio.run(app())


if __name__ == "__main__":
    asyncio.run(app())
