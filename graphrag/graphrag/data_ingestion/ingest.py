import json

import logging

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import pypdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the most recent PDF from data/raw
def get_latest_pdf(directory="graphrag/data/raw"):
    # Convert to Path object and resolve to absolute path
    dir_path = Path(directory).resolve()
    logger.info(f"Scanning {dir_path} for PDF files")
    
    # Ensure directory exists
    if not dir_path.exists():
        logger.error(f"Directory {dir_path} does not exist")
        raise FileNotFoundError(f"Directory {dir_path} does not exist")
    
    # Use Path.glob for cross-platform compatibility
    pdf_files = list(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {dir_path}")
        raise FileNotFoundError(f"No PDF files found in {dir_path}")
    
    # Select the most recently modified PDF
    latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Selected latest PDF: {latest_pdf}")
    return str(latest_pdf)

# Load the document with fallback
def load_document(file_path):
    # Convert to Path object and resolve to absolute path
    file_path = Path(file_path).resolve()
    logger.info(f"Loading document: {file_path}")
    
    # Verify file exists
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages with PyPDFLoader")
        return documents
    except UnicodeDecodeError as e:
        logger.warning(f"UnicodeDecodeError with PyPDFLoader: {str(e)}. Falling back to pypdf.")
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                documents = []
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    try:
                        text = page.extract_text() or ""
                        text = text.encode('utf-8', errors='replace').decode('utf-8')
                        documents.append({
                            "page_content": text,
                            "metadata": {"source": str(file_path), "page": page_num}
                        })
                    except Exception as page_error:
                        logger.warning(f"Error processing page {page_num}: {str(page_error)}")
                        documents.append({
                            "page_content": "",
                            "metadata": {"source": str(file_path), "page": page_num, "error": "Failed to extract text"}
                        })
                logger.info(f"Loaded {len(documents)} pages with pypdf fallback")
                return [Document(**doc) for doc in documents]
        except Exception as fallback_error:
            logger.error(f"Fallback failed: {str(fallback_error)}")
            raise
    except Exception as e:
        logger.error(f"Error loading document with PyPDFLoader: {str(e)}")
        raise

# Split document into chunks
def split_chunks(documents, chunk_size=500, chunk_overlap=50):
    logger.info(f"Splitting {len(documents)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

# Generate embeddings using open-source model
def generate_embeddings(chunks):
    logger.info("Generating embeddings with SentenceTransformer")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    logger.info(f"Generated embeddings for {len(embeddings)} chunks")
    return embeddings

# Save chunk data to JSON
def save_chunks_to_json(chunks, filename="chunk_data.json"):
    # Create directory using Path
    chunks_dir = Path("graphrag/data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = chunks_dir / filename
    logger.info(f"Saving chunks to {file_path}")
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "id": f"chunk_{i}",
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} chunks to JSON")

# Save embeddings as a JSON object
def save_embeddings(embeddings, chunks, filename="embedding_graph.json"):
    # Create directory using Path
    embedding_dir = Path("graphrag/data/embedding")
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = embedding_dir / filename
    logger.info(f"Saving embeddings to {file_path}")
    data = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        data.append({
            "chunk_id": f"chunk_{i}",
            "embedding": embedding,
            "metadata": chunk.metadata
        })
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} embeddings to JSON")

# Main pipeline
def main():
    logger.info("Starting document ingestion pipeline")
    try:
        # Get the latest PDF
        file_path = get_latest_pdf()
        documents = load_document(file_path)
        chunks = split_chunks(documents)
        embeddings = generate_embeddings(chunks)
        save_chunks_to_json(chunks)
        save_embeddings(embeddings, chunks)
        logger.info(f"Pipeline completed successfully for {file_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
