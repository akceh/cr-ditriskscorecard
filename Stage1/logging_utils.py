import logging
from datetime import datetime

# Configure the logger
logging.basicConfig(
    filename='assistant_logs.log',  # Log file name
    level=logging.INFO,             # Log level
    format='%(asctime)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'     # Date and time format
)

def log_interaction(user_input, assistant_response):
    """Logs the interaction with the assistant."""
    logging.info(f"User: {user_input}")
    logging.info(f"Assistant: {assistant_response}")
