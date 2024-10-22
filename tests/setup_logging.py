import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)
