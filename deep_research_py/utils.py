import logging
import os
from datetime import datetime
from pathlib import Path
import json
import re

def setup_logging(log_level=logging.INFO, log_to_file=True, log_file_path=None):
    """
    Set up comprehensive logging for the deep research system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write logs to file
        log_file_path: Custom log file path (optional)
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp if not provided
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"deep_research_{timestamp}.log"
    
    # Create formatter for detailed logging
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create formatter for console (less verbose)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if log_to_file:
        # File handler (for detailed file logging)
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG level in file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Log the setup
        logger = logging.getLogger(__name__)
        logger.info(f"📁 Logging initialized - File: {log_file_path}")
        logger.info(f"🔧 Console level: {logging.getLevelName(log_level)}")
        logger.info(f"🔧 File level: DEBUG")
        
        return str(log_file_path)
    
    return None


def validate_and_fix_json(json_str: str) -> dict:
    """Validate and attempt to fix JSON strings with common issues."""
    try:
        # First try direct parsing
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"JSON validation failed: {e}")
        
        # Try to fix common issues
        fixed_str = json_str
        
        # Remove any markdown code blocks
        fixed_str = re.sub(r'```json\s*', '', fixed_str)
        fixed_str = re.sub(r'\s*```', '', fixed_str)
        
        # Fix unescaped newlines and tabs
        fixed_str = fixed_str.replace('\n', '\\n')
        fixed_str = fixed_str.replace('\t', '\\t')
        fixed_str = fixed_str.replace('\r', '\\r')
        
        # Try parsing again
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            logger.error("Could not fix JSON formatting")
            return {}


# Create the main logger instance
logger = logging.getLogger("deep_research")
