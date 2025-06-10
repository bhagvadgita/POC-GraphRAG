# graph_generation.py

import json
import asyncio
import re
import os
import sys
import aiohttp
from dotenv import load_dotenv
from collections import defaultdict
import ssl

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from graphrag.graph_creation.prompts import PROMPTS
from graphrag.graph_creation.merge import merge_nodes_and_edges

# Load environment variables from .env file
load_dotenv()

DEFAULT_ENTITY_TYPES = PROMPTS["DEFAULT_ENTITY_TYPES"]

async def use_llm(prompt: str, model: str = "gemini-2.0-flash") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Configure timeout
    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds total timeout
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload, ssl=ssl_context) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                result = await response.json()
                response_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                return response_text
    except aiohttp.ClientError as e:
        print(f"Connection error: {str(e)}")
        # Retry once after a short delay
        await asyncio.sleep(2)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, ssl=ssl_context) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
                    return response_text
        except Exception as e:
            print(f"Retry failed: {str(e)}")
            raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

def parse_llm_output(llm_output: str, delimiter: str, record_delimiter: str) -> tuple:
    nodes = defaultdict(list)
    edges = defaultdict(list)

    # Split by record delimiter and clean up
    records = [r.strip() for r in llm_output.split(record_delimiter) if r.strip()]
    for record in records:
        # Extract content between parentheses
        match = re.search(r"\((.*?)\)", record)
        if not match:
            continue
        content = match.group(1)
        parts = [p.strip() for p in content.split(delimiter)]
        if not parts:
            continue
        # Remove quotes from the type identifier
        record_type = parts[0].strip('"')
        if record_type == "entity" and len(parts) >= 4:
            entity = {
                "entity_name": parts[1],
                "entity_type": parts[2],
                "description": parts[3]
            }
            nodes[entity["entity_name"]].append(entity)
        elif record_type == "relationship" and len(parts) >= 6:
            # Clean up the relationship strength value
            strength = parts[5].replace("<relationship_strength>", "").replace("</relationship_strength>", "")
            relation = {
                "src_id": parts[1],
                "tgt_id": parts[2],
                "description": parts[3],
                "keywords": parts[4],
                "strength": strength
            }
            edges[(relation["src_id"], relation["tgt_id"])].append(relation)
    return nodes, edges

async def extract_entities_from_json(json_path: str, language: str = "English", start_chunk: int = 0) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    entity_extract_prompt_template = PROMPTS["entity_extraction"]
    continue_prompt_template = PROMPTS["entity_continue_extraction"]

    delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delim = PROMPTS["DEFAULT_RECORD_DELIMITER"]
    complete_delim = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    context_base = {
        "tuple_delimiter": delimiter,
        "record_delimiter": record_delim,
        "completion_delimiter": complete_delim,
        "entity_types": ", ".join(DEFAULT_ENTITY_TYPES),
        "examples": "",
        "language": language,
    }

    chunk_results = []
    
    # Create a directory for intermediate results if it doesn't exist
    os.makedirs("graphrag/data/intermediate", exist_ok=True)
    
    # Load existing results if any
    intermediate_file = "graphrag/data/intermediate/chunk_results.json"
    if os.path.exists(intermediate_file):
        with open(intermediate_file, "r") as f:
            chunk_results = json.load(f)
            print(f"Loaded {len(chunk_results)} existing chunk results")

    # Process chunks starting from the specified index
    for i, chunk in enumerate(chunks[start_chunk:], start=start_chunk):
        content = chunk["text"]
        chunk_id = chunk["id"]
        
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        initial_prompt = entity_extract_prompt_template.format(
            **context_base, input_text=content
        )

        try:
            initial_output = await use_llm(initial_prompt)
            glean_prompt = continue_prompt_template.format(**context_base)
            glean_output = await use_llm(glean_prompt)
            combined_output = initial_output + "\n" + glean_output

            nodes, edges = parse_llm_output(
                combined_output,
                delimiter=delimiter,
                record_delimiter=record_delim
            )

            # Convert defaultdict to regular dict for JSON serialization
            nodes_dict = {k: list(v) for k, v in nodes.items()}
            # Add chunk_id to each node's metadata
            for entity_name, entity_list in nodes_dict.items():
                for entity in entity_list:
                    entity["chunk_id"] = chunk_id
                    entity["metadata"] = entity.get("metadata", {})
                    entity["metadata"]["chunk_id"] = chunk_id
            
            # Convert tuple keys to strings for JSON serialization
            edges_dict = {f"{k[0]}->{k[1]}": list(v) for k, v in edges.items()}
            
            chunk_results.append((nodes_dict, edges_dict))
            
            # Save intermediate results after each chunk
            with open(intermediate_file, "w") as f:
                json.dump(chunk_results, f, indent=2)
                
            print(f"Successfully processed chunk {i+1}")
            
        except Exception as e:
            print(f"Error in chunk {chunk_id}: {str(e)}")
            print(f"Progress saved. You can resume from chunk {i+1}")
            # Save the current progress before raising the error
            with open(intermediate_file, "w") as f:
                json.dump(chunk_results, f, indent=2)
            raise

    return chunk_results

async def run_graph_pipeline(
    json_path: str,
    knowledge_graph_inst,
    entity_vdb,
    relationships_vdb,
    global_config: dict,
    start_chunk: int = 0
):
    chunk_results = await extract_entities_from_json(json_path, start_chunk=start_chunk)
    await merge_nodes_and_edges(
        chunk_results=chunk_results,
        knowledge_graph_inst=knowledge_graph_inst,
        entity_vdb=entity_vdb,
        relationships_vdb=relationships_vdb,
        global_config=global_config,
        file_path=json_path,
        current_file_number=start_chunk + 1,
        total_files=len(chunk_results),
        pipeline_status={},
        pipeline_status_lock=asyncio.Lock(),
        llm_response_cache=None
    )

# Example usage
if __name__ == "__main__":
    import asyncio
    from kg.local_storage import LocalGraphStorage, LocalVectorStorage

    async def main():
        # Initialize storage
        knowledge_graph = LocalGraphStorage()
        entity_vdb = LocalVectorStorage()
        relationships_vdb = LocalVectorStorage()
        
        # Configuration
        global_config = {
            "language": "English",
            "model": "gemini-2.0-flash"
        }
        
        # Path to your chunk file
        chunk_file_path = "graphrag/data/chunks/chunk_data.json"
        
        try:
            # Try to resume from the last processed chunk
            intermediate_file = "graphrag/data/intermediate/chunk_results.json"
            start_chunk = 0
            if os.path.exists(intermediate_file):
                with open(intermediate_file, "r") as f:
                    existing_results = json.load(f)
                    start_chunk = len(existing_results)
                    print(f"Resuming from chunk {start_chunk}")
            
            # Run the pipeline
            await run_graph_pipeline(
                json_path=chunk_file_path,
                knowledge_graph_inst=knowledge_graph,
                entity_vdb=entity_vdb,
                relationships_vdb=relationships_vdb,
                global_config=global_config,
                start_chunk=start_chunk
            )
            
            print("Knowledge graph generation completed!")
            print(f"Graph data stored in: {knowledge_graph.graph_file}")
            print(f"Vector data stored in: {entity_vdb.vector_file} and {relationships_vdb.vector_file}")
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            print("You can resume the pipeline later from where it left off.")

    # Run the async main function
    asyncio.run(main())
