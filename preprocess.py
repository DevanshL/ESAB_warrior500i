# preprocess.py

import os
import re
import pdfplumber
import glob
import pandas as pd
from typing import Dict, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging
import traceback

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def is_pdf_valid(pdf_path: str) -> bool:
    """
    Validates a PDF by checking for the presence of the word 'dimensions'.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        bool: True if 'dimensions' is found, False otherwise.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and "dimensions" in text.lower():
                    logger.info(f"'dimensions' text found in PDF '{pdf_path}'.")
                    return True
        logger.warning(f"'dimensions' text not found in PDF '{pdf_path}'.")
        return False
    except Exception as e:
        logger.error(f"Error validating PDF {pdf_path}: {e}")
        logger.debug(traceback.format_exc())  # Detailed traceback for debugging
        return False

def load_esab_machines(pdf_dir: str) -> List[str]:
    """
    Validates PDFs in the specified directory and returns a list of valid ESAB machine names.
    
    Args:
        pdf_dir (str): Directory containing PDF manuals.
    
    Returns:
        List[str]: List of valid ESAB machine names.
    """
    logger.info(f"Loading ESAB machines from directory '{pdf_dir}'.")
    ESAB_MACHINES = []
    for manual_name in os.listdir(pdf_dir):
        if manual_name.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, manual_name)
            machine_name = manual_name[:-4]
            logger.info(f"Validating PDF for machine '{machine_name}'.")
            if is_pdf_valid(pdf_path):
                ESAB_MACHINES.append(machine_name)
                logger.info(f"Added machine '{machine_name}' to ESAB_MACHINES.")
            else:
                logger.warning(f"PDF for '{machine_name}' is invalid or lacks 'dimensions' data. Skipping.")
    logger.info(f"Total valid machines found: {len(ESAB_MACHINES)}.")
    return ESAB_MACHINES

def extract_all_content_as_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Converts each PDF's text and tables into Document objects for FAISS indexing.
    
    Args:
        pdf_paths (List[str]): List of PDF file paths.
    
    Returns:
        List[Document]: List of Document objects.
    """
    logger.info("Extracting all content from PDFs as Documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = []
    for pdf_path in pdf_paths:
        machine_name = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"Extracting content from PDF '{pdf_path}'.")
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text:
                    split_texts = text_splitter.split_text(page_text)
                    for idx, chunk in enumerate(split_texts):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "machine": machine_name,
                                "page": page_number,
                                "chunk_idx": idx
                            }
                        ))
                    logger.info(f"Extracted {len(split_texts)} text chunks from page {page_number} of '{machine_name}'.")
                # Convert tables to CSV
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0]).fillna("")
                            table_csv = df.to_csv(index=False)
                            documents.append(Document(
                                page_content=table_csv,
                                metadata={
                                    "machine": machine_name,
                                    "page": page_number,
                                    "table_idx": table_idx
                                }
                            ))
                            logger.info(f"Extracted table {table_idx} from page {page_number} of '{machine_name}' as CSV.")
    logger.info(f"Total documents extracted: {len(documents)}.")
    return documents

def extract_sections(pdf_paths: List[str], sections_to_extract: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extracts specified sections from the given PDF files.
    Returns a nested dict: {machine_name: {section: "content", ...}, ...}
    
    Args:
        pdf_paths (List[str]): List of PDF file paths.
        sections_to_extract (List[str]): List of section names to extract.
    
    Returns:
        Dict[str, Dict[str, str]]: Extracted sections.
    """
    logger.info("Starting extraction of specified sections from PDFs.")
    extracted_data = {}
    # Patterns for the headers we care about
    section_header_patterns = {
        section: [
            re.compile(rf'^\d+\.\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^[IVXLCDM]+\.\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^\d+\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^[IVXLCDM]+\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^\d+\.\s+{re.escape(section.upper())}[:\-]$', re.IGNORECASE),
            re.compile(rf'^[IVXLCDM]+\.\s+{re.escape(section.upper())}[:\-]$', re.IGNORECASE),
            re.compile(rf'^{re.escape(section.upper())}[:\-]$', re.IGNORECASE),
        ]
        for section in sections_to_extract
    }
    # Pattern for any new "unwanted" section header (stop capturing text if we hit a new one)
    any_section_header_pattern = re.compile(r'^\d+\s+[A-Z\s\-]+$', re.IGNORECASE)

    for pdf_path in pdf_paths:
        machine_name = os.path.splitext(os.path.basename(pdf_path))[0]
        extracted_data[machine_name] = {section: "" for section in sections_to_extract}
        logger.info(f"Processing PDF for machine '{machine_name}'.")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_section = None
                collected_text = {section: "" for section in sections_to_extract}

                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    lines = page_text.split('\n')

                    for i, line in enumerate(lines):
                        line_stripped = line.strip()
                        line_upper = line_stripped.upper()

                        # Check if line is a header for a desired section
                        matched_section = None
                        for section, patterns in section_header_patterns.items():
                            for pattern in patterns:
                                if pattern.match(line_stripped):
                                    matched_section = section
                                    break
                            if matched_section:
                                break

                        if matched_section:
                            current_section = matched_section
                            # Collect the rest of the page after the matched header
                            if i + 1 < len(lines):
                                collected_text[current_section] += '\n'.join(lines[i+1:]) + '\n'
                            logger.info(f"Detected section '{matched_section}' in '{machine_name}'.")
                            break  # move to next page
                        elif any_section_header_pattern.match(line_upper):
                            # Found a new section not in sections_to_extract
                            current_section = None
                            continue

                        # If we're inside a desired section, accumulate text
                        if current_section:
                            collected_text[current_section] += line + '\n'

                # Final assignment of extracted text
                for section in sections_to_extract:
                    content = collected_text[section].strip()
                    if content:
                        extracted_data[machine_name][section] = content
                        logger.info(f"Extracted content for section '{section}' in '{machine_name}'.")
                    else:
                        extracted_data[machine_name][section] = f"[INFO] No {section.upper()} section found."
                        logger.warning(f"No content found for section '{section}' in '{machine_name}'.")
        except Exception as e:
            logger.error(f"[ERROR] Processing {pdf_path}: {e}")
            logger.debug(traceback.format_exc())  # Detailed traceback for debugging

    logger.info("Completed extraction of sections from all PDFs.")
    return extracted_data

def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Converts a DataFrame of welding processes and machines into
    a list of string-based Documents for indexing and retrieval.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Welding Process' and 'Machines'.
    
    Returns:
        List[Document]: List of Document objects.
    """
    logger.info("Converting DataFrame to Document objects.")
    documents = []
    for _, row in df.iterrows():
        process = row["Welding Process"]
        machines = row["Machines"]
        content = f"The welding process {process} is compatible with the following machines: {machines}."
        documents.append(Document(page_content=content))
    logger.info(f"Converted {len(documents)} DataFrame rows to Documents.")
    return documents

def detect_welding_processes(sections: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Detects which machines support MMA, MIG/MAG, TIG, FCAW, etc.
    Returns a DataFrame with 'Welding Process' and 'Compatible Machines'.
    
    Args:
        sections (Dict[str, Dict[str, str]]): Extracted sections from PDFs.
    
    Returns:
        pd.DataFrame: DataFrame containing welding processes and compatible machines.
    """
    logger.info("Starting detection of welding processes supported by machines.")
    welding_processes = {
        "MMA": ["MMA", "STICK", "SMAW"],
        "MIG/MAG": ["MIG", "MAG", "GMAW"],
        "TIG": ["TIG", "GTAW"],
        "FCAW": ["FCAW", "FLUX CORED"]
    }
    process_machines = {proc: [] for proc in welding_processes.keys()}

    for machine, content_dict in sections.items():
        # Combine "INTRODUCTION" + "TECHNICAL DATA" text
        combined_content = ""
        for sec_name in ["INTRODUCTION", "TECHNICAL DATA"]:
            sec_text = content_dict.get(sec_name, "")
            combined_content += sec_text.upper() + " "

        # Match processes
        for process_name, keywords in welding_processes.items():
            for kw in keywords:
                pattern = r'\b' + re.escape(kw.upper()) + r'\b'
                if re.search(pattern, combined_content):
                    process_machines[process_name].append(machine)
                    logger.info(f"Machine '{machine}' supports welding process '{process_name}'.")
                    break

    data = {"Welding Process": [], "Machines": []}
    for process_name, machines_list in process_machines.items():
        if machines_list:
            data["Welding Process"].append(process_name)
            data["Machines"].append(", ".join(machines_list))
            logger.info(f"Added welding process '{process_name}' with machines: {machines_list}.")

    df = pd.DataFrame(data)
    df = df.sort_values("Welding Process").reset_index(drop=True)
    logger.info("Completed detection of welding processes.")
    return df

def extract_all_content_as_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Converts each PDF's text and tables into Document objects for FAISS indexing.
    
    Args:
        pdf_paths (List[str]): List of PDF file paths.
    
    Returns:
        List[Document]: List of Document objects.
    """
    logger.info("Extracting all content from PDFs as Documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = []
    for pdf_path in pdf_paths:
        machine_name = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"Extracting content from PDF '{pdf_path}'.")
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text:
                    split_texts = text_splitter.split_text(page_text)
                    for idx, chunk in enumerate(split_texts):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "machine": machine_name,
                                "page": page_number,
                                "chunk_idx": idx
                            }
                        ))
                    logger.info(f"Extracted {len(split_texts)} text chunks from page {page_number} of '{machine_name}'.")
                # Convert tables to CSV
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0]).fillna("")
                            table_csv = df.to_csv(index=False)
                            documents.append(Document(
                                page_content=table_csv,
                                metadata={
                                    "machine": machine_name,
                                    "page": page_number,
                                    "table_idx": table_idx
                                }
                            ))
                            logger.info(f"Extracted table {table_idx} from page {page_number} of '{machine_name}' as CSV.")
    logger.info(f"Total documents extracted: {len(documents)}.")
    return documents

def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Converts a DataFrame of welding processes and machines into
    a list of string-based Documents for indexing and retrieval.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Welding Process' and 'Machines'.
    
    Returns:
        List[Document]: List of Document objects.
    """
    logger.info("Converting DataFrame to Document objects.")
    documents = []
    for _, row in df.iterrows():
        process = row["Welding Process"]
        machines = row["Machines"]
        content = f"The welding process {process} is compatible with the following machines: {machines}."
        documents.append(Document(page_content=content))
    logger.info(f"Converted {len(documents)} DataFrame rows to Documents.")
    return documents

def detect_welding_processes(sections: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Detects which machines support MMA, MIG/MAG, TIG, FCAW, etc.
    Returns a DataFrame with 'Welding Process' and 'Compatible Machines'.
    
    Args:
        sections (Dict[str, Dict[str, str]]): Extracted sections from PDFs.
    
    Returns:
        pd.DataFrame: DataFrame containing welding processes and compatible machines.
    """
    logger.info("Starting detection of welding processes supported by machines.")
    welding_processes = {
        "MMA": ["MMA", "STICK", "SMAW"],
        "MIG/MAG": ["MIG", "MAG", "GMAW"],
        "TIG": ["TIG", "GTAW"],
        "FCAW": ["FCAW", "FLUX CORED"]
    }
    process_machines = {proc: [] for proc in welding_processes.keys()}

    for machine, content_dict in sections.items():
        # Combine "INTRODUCTION" + "TECHNICAL DATA" text
        combined_content = ""
        for sec_name in ["INTRODUCTION", "TECHNICAL DATA"]:
            sec_text = content_dict.get(sec_name, "")
            combined_content += sec_text.upper() + " "

        # Match processes
        for process_name, keywords in welding_processes.items():
            for kw in keywords:
                pattern = r'\b' + re.escape(kw.upper()) + r'\b'
                if re.search(pattern, combined_content):
                    process_machines[process_name].append(machine)
                    logger.info(f"Machine '{machine}' supports welding process '{process_name}'.")
                    break

    data = {"Welding Process": [], "Machines": []}
    for process_name, machines_list in process_machines.items():
        if machines_list:
            data["Welding Process"].append(process_name)
            data["Machines"].append(", ".join(machines_list))
            logger.info(f"Added welding process '{process_name}' with machines: {machines_list}.")

    df = pd.DataFrame(data)
    df = df.sort_values("Welding Process").reset_index(drop=True)
    logger.info("Completed detection of welding processes.")
    return df

def create_faiss_db(esab_machines: List[str], pdf_dir: str = 'pdfs', faiss_db_dir: str = 'faiss_dbs') -> FAISS:
    """
    Creates or loads a FAISS database from the provided PDFs and ESAB machines.
    
    Args:
        esab_machines (List[str]): List of valid ESAB machine names.
        pdf_dir (str): Directory containing PDF manuals.
        faiss_db_dir (str): Directory to store/load FAISS databases.
    
    Returns:
        FAISS: Loaded or newly created FAISS database object.
    """
    faiss_db_path = os.path.join(faiss_db_dir, "combined_faiss_db")
    logger.info("Preparing to load or create FAISS database.")

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API")
    )
    logger.info("Initialized GoogleGenerativeAIEmbeddings.")

    # If FAISS DB already exists, load it
    if os.path.exists(faiss_db_path):
        logger.info("Loading existing FAISS database...")
        try:
            db = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS database loaded successfully.")
            return db
        except Exception as e:
            logger.error(f"Failed to load FAISS database from '{faiss_db_path}': {e}")
            logger.debug(traceback.format_exc())  # Detailed traceback for debugging
            return None

    # Otherwise create a new one
    logger.info("Creating new FAISS database...")
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_paths:
        logger.error(f"No PDF files found in '{pdf_dir}' directory.")
        return None

    # 1) Extract base documents from PDFs
    documents = extract_all_content_as_documents(pdf_paths)
    logger.info(f"Extracted {len(documents)} documents from PDFs.")

    # 2) Also create a doc listing all ESAB machines
    all_machines_text = "ESAB Machines List:\n" + "\n".join(esab_machines)
    machine_list_doc = Document(
        page_content=all_machines_text,
        metadata={"source": "machine_list"}
    )
    documents.append(machine_list_doc)
    logger.info("Added machine list document to FAISS database.")

    # 3) Extract sections and detect welding processes
    sections_to_extract = ["INTRODUCTION", "TECHNICAL DATA"]
    extracted_sections = extract_sections(pdf_paths, sections_to_extract)
    process_df = detect_welding_processes(extracted_sections)

    if not process_df.empty:
        # Convert that DataFrame into Document objects
        process_documents = dataframe_to_documents(process_df)
        # Add them to the list of documents
        documents.extend(process_documents)
        logger.info(f"Added {len(process_documents)} welding process documents to the knowledge base.")
    else:
        logger.info("No welding processes identified or no relevant sections found.")

    # 4) Build the FAISS DB
    try:
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(faiss_db_path)
        logger.info(f"FAISS database created and saved to '{faiss_db_path}'.")
        return db
    except Exception as e:
        logger.error(f"Failed to create/save FAISS database: {e}")
        logger.debug(traceback.format_exc())  # Detailed traceback for debugging
        return None

def initialize_resources() -> Tuple[List[str], FAISS]:
    """
    Initializes and returns ESAB machines and FAISS database.
    
    Returns:
        Tuple[List[str], FAISS]: List of ESAB machines and FAISS database object.
    """
    logger.info("Initializing resources - should only appear once.")
    esab_machines = load_esab_machines('pdfs')
    if not esab_machines:
        logger.error("No valid ESAB machines found during initialization.")
        return esab_machines, None
    faiss_db = create_faiss_db(esab_machines, pdf_dir='pdfs', faiss_db_dir='faiss_dbs')
    return esab_machines, faiss_db

if __name__ == "__main__":
    """
    Preprocessing Script
    --------------------
    This script validates PDFs, extracts necessary information, and creates a FAISS database.
    Run this script before starting the Streamlit application to ensure the FAISS database is ready.
    """
    try:
        esab_machines = load_esab_machines('pdfs')
        if not esab_machines:
            logger.error("No valid machine manuals found. Exiting preprocessing.")
            exit(1)
        faiss_db = create_faiss_db(esab_machines, pdf_dir='pdfs', faiss_db_dir='faiss_dbs')
        if faiss_db:
            logger.info("Preprocessing completed successfully.")
        else:
            logger.error("Failed to create FAISS database.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        logger.debug(traceback.format_exc())  # Detailed traceback for debugging
