# utils.py
import base64
import re
from typing import List
from fuzzywuzzy import fuzz
import logging

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_esab_logo(logo_path: str = "esab-logo.png") -> str:
    """
    Loads and base64-encodes the ESAB logo.
    
    Args:
        logo_path (str): Path to the ESAB logo image.
    
    Returns:
        str: Base64-encoded string of the logo image.
    """
    logger.info("Loading ESAB logo.")
    try:
        with open(logo_path, "rb") as f:
            logo = base64.b64encode(f.read()).decode()
            logger.info("ESAB logo loaded successfully.")
            return logo
    except FileNotFoundError:
        logger.warning(f"ESAB logo file '{logo_path}' not found.")
        return None

def detect_machine_in_query(query: str, esab_machines: List[str]) -> List[str]:
    """
    Attempts to detect specific machine names from the user's query using exact and fuzzy matching.
    
    Args:
        query (str): The user's input query.
        esab_machines (List[str]): List of valid ESAB machine names.
    
    Returns:
        List[str]: List of detected machine names.
    """
    logger.info(f"Detecting machines in query: '{query}'.")
    
    # Normalize the query by replacing hyphens and other special characters
    query_normalized = re.sub(r'[-]', ' ', query.lower())
    machine_names = {re.sub(r'[-]', ' ', m.lower()): m for m in esab_machines}
    detected = []

    # Exact match
    for machine_key, machine_original in machine_names.items():
        pattern = re.compile(rf"\b{re.escape(machine_key)}\b", re.IGNORECASE)
        if pattern.findall(query_normalized):
            detected.append(machine_original)
            logger.info(f"Exact match found for machine '{machine_original}' in query.")

    if detected:
        logger.info(f"Detected machines (exact match): {detected}.")
        return list(set(detected))

    # Fuzzy matching
    for machine_key, machine_original in machine_names.items():
        partial_score = fuzz.partial_ratio(machine_key, query_normalized)
        token_set_score = fuzz.token_set_ratio(machine_key, query_normalized)
        if max(partial_score, token_set_score) > 70:
            detected.append(machine_original)
            logger.info(f"Fuzzy match found for machine '{machine_original}' with scores {partial_score}, {token_set_score}.")

    # Simple "plural" check
    for machine_key, machine_original in machine_names.items():
        if machine_original.endswith('i') and not machine_original.endswith('ii'):
            plural = machine_original + 's'
            if plural.lower() in query_normalized:
                detected.append(machine_original)
                logger.info(f"Plural match found for machine '{machine_original}' in query.")

    logger.info(f"Detected machines (final): {list(set(detected))}.")
    return list(set(detected))

