import os
import sys
import json
import requests
from dotenv import load_dotenv
sys.path.append('../graphrag')
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graphrag.data_ingestion.ingest import load_document, split_chunks, generate_embeddings, save_chunks_to_json, save_embeddings
from graphrag.graph_creation.graph_generation import run_graph_pipeline
from graphrag.RAG.hybrid_rag import load_data, HybridVectorGraphRetriever, KnowledgeGraph
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_ENDPOINT = os.getenv("GEMINI_API_ENDPOINT")
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "gemini_prompt.json")

app = FastAPI(title="Document Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class SearchQuery(BaseModel):
    query: str
    k: int = 3
    search_type: str = "hybrid"  # Options: "hybrid", "vector", "sparse", "graph"

# Initialize RAG system
try:
    chunks, embeddings, _ = load_data()
    
    # Load graph data from chunk_results.json
    with open("graphrag/data/intermediate/chunk_results.json", "r") as f:
        graph_data = json.load(f)
    
    knowledge_graph = KnowledgeGraph()
    if not knowledge_graph.load_from_json(graph_data):
        raise Exception("Failed to load knowledge graph from JSON data")
    
    retriever = HybridVectorGraphRetriever(
        chunks=chunks,
        embeddings=embeddings,
        knowledge_graph=knowledge_graph,
        vector_weight=0.4,
        sparse_weight=0.3,
        graph_weight=0.3
    )
except Exception as e:
    print(f"Error initializing RAG system: {str(e)}")
    retriever = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Document Processing API"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Create data/raw directory if it doesn't exist
        os.makedirs("graphrag/data/raw", exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join("graphrag/data/raw", file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        documents = load_document(file_path)
        chunks = split_chunks(documents)
        embeddings = generate_embeddings(chunks)
        
        # Initialize storage for graph generation
        from graph_creation.kg.local_storage import LocalGraphStorage, LocalVectorStorage
        knowledge_graph = LocalGraphStorage()
        entity_vdb = LocalVectorStorage()
        relationships_vdb = LocalVectorStorage()
        
        # Generate graph
        await run_graph_pipeline(
            json_path=file_path,
            knowledge_graph_inst=knowledge_graph,
            entity_vdb=entity_vdb,
            relationships_vdb=relationships_vdb,
            global_config={"language": "English", "model": "gemini-2.0-flash"}
        )
        
        # Save the processed data
        save_chunks_to_json(chunks)
        save_embeddings(embeddings, chunks)
        chunks, embeddings, _ = load_data()
        
        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/chat-query")
async def chat_query(search_query: SearchQuery):
    """
    Process a chat query using the specified search method and generate a response using Gemini.
    
    Args:
        search_query (SearchQuery): The search parameters including:
            - query (str): The search query
            - k (int): Number of results to return (default: 3)
            - search_type (str): Type of search to perform (default: "hybrid")
                Options: "hybrid", "vector", "sparse", "graph"
    
    Returns:
        Dict containing the query, search type, model response, and search results
    """
    if not retriever:
        raise HTTPException(
            status_code=500,
            detail="RAG system not properly initialized. Please ensure data files exist."
        )
    
    try:
        # Perform search based on type
        if search_query.search_type == "vector":
            results = retriever.vector_search(search_query.query, k=search_query.k)
        elif search_query.search_type == "sparse":
            sparse_results = retriever.sparse_search(search_query.query, k=search_query.k)
            # Convert sparse results to Document objects
            results = []
            for idx, score in sparse_results:
                if idx < len(retriever.chunks):
                    chunk = retriever.chunks[idx]
                    node_id = f"chunk_{idx}"
                    node = retriever.knowledge_graph.nodes.get(node_id)
                    
                    metadata = {
                        **chunk.get("metadata", {}),
                        "hybrid_score": float(score),
                        "chunk_id": idx,
                        "keywords": chunk.get("keywords", []),
                        "entities": node.metadata.get('entities', []) if node else [],
                        "topics": node.metadata.get('topics', []) if node else [],
                        "importance_score": node.importance_score if node else 0.0,
                        "connected_chunks": len(node.connections) if node else 0,
                        "search_methods": {
                            "vector": False,
                            "sparse": True,
                            "graph": False
                        }
                    }
                    
                    doc = Document(page_content=chunk["text"], metadata=metadata)
                    results.append(doc)
        elif search_query.search_type == "graph":
            results = retriever.graph_search(search_query.query, k=search_query.k)
        else:  # hybrid
            results = retriever.hybrid_search(search_query.query, k=search_query.k)
        
        # Format results for response
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": {
                    "hybrid_score": doc.metadata["hybrid_score"],
                    "entities": doc.metadata["entities"],
                    "topics": doc.metadata["topics"],
                    "search_methods": doc.metadata["search_methods"]
                }
            })
        
        # Prepare search results JSON
        search_results = {
            "query": search_query.query,
            "search_type": search_query.search_type,
            "k": search_query.k,
            "results": formatted_results
        }
        
        # Load prompt from JSON file
        try:
            with open(PROMPT_FILE_PATH, "r") as f:
                prompt_data = json.load(f)
                prompt_template = prompt_data["prompt"]
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=f"Prompt file not found at {PROMPT_FILE_PATH}"
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Invalid JSON in prompt file"
            )
        
        # Format prompt with search results
        formatted_prompt = prompt_template.replace("{search_results}", json.dumps(search_results, indent=2))
        formatted_prompt = formatted_prompt.replace("{query}", search_query.query)
        
        # Send to Gemini API
        try:
            payload = {
                "contents": [{
                    "parts": [{
                        "text": formatted_prompt
                    }]
                }]
            }
            headers = {
                "Content-Type": "application/json"
            }
            params = {
                "key": GEMINI_API_KEY
            }
            response = requests.post(GEMINI_API_ENDPOINT, json=payload, headers=headers, params=params)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {response.status_code}, {response.text}"
                )
            
            gemini_response = response.json()
            # Extract the response text from Gemini's response structure
            model_output = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
        
        except requests.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to Gemini API: {str(e)}"
            )
        
        # Return response
        return {
            "query": search_query.query,
            "search_type": search_query.search_type,
            "k": search_query.k,
            "model_response": model_output,
            "search_results": formatted_results  # Include original results for reference
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 