
import logging
from collections import defaultdict
from typing import Optional, Dict, List

from .kg.local_storage import LocalGraphStorage, LocalVectorStorage, LocalKVStorage
from .hashing import compute_hash_id

logger = logging.getLogger(__name__)

async def _merge_nodes_then_upsert(
    entity_name: str,
    entities: List[Dict],
    knowledge_graph_inst: LocalGraphStorage,
    global_config: Dict[str, str],
    pipeline_status: Optional[Dict] = None,
    pipeline_status_lock = None,
    llm_response_cache: Optional[LocalKVStorage] = None,
) -> Dict:
    # Simple merge: take the first entity's type and combine descriptions
    if not entities:
        return {}
    
    merged_entity = {
        "entity_name": entity_name,
        "entity_type": entities[0]["entity_type"],
        "description": " ".join(e["description"] for e in entities),
        "source_id": entities[0].get("source_id", "unknown"),
        "file_path": entities[0].get("file_path", "unknown_source")
    }
    
    # Store in graph
    await knowledge_graph_inst.upsert({
        "nodes": {entity_name: merged_entity}
    })
    
    return merged_entity

async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges: List[Dict],
    knowledge_graph_inst: LocalGraphStorage,
    global_config: Dict[str, str],
    pipeline_status: Optional[Dict] = None,
    pipeline_status_lock = None,
    llm_response_cache: Optional[LocalKVStorage] = None,
) -> Optional[Dict]:
    if not edges:
        return None
    
    # Simple merge: combine descriptions and keywords
    merged_edge = {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "description": " ".join(e["description"] for e in edges),
        "keywords": " ".join(e["keywords"] for e in edges),
        "strength": max(float(e["strength"]) for e in edges),
        "source_id": edges[0].get("source_id", "unknown"),
        "file_path": edges[0].get("file_path", "unknown_source")
    }
    
    # Store in graph
    edge_key = f"{src_id}->{tgt_id}"
    await knowledge_graph_inst.upsert({
        "edges": {edge_key: merged_edge}
    })
    
    return merged_edge

async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: LocalGraphStorage,
    entity_vdb: LocalVectorStorage,
    relationships_vdb: LocalVectorStorage,
    global_config: Dict[str, str],
    pipeline_status: Dict = None,
    pipeline_status_lock = None,
    llm_response_cache: LocalKVStorage = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> None:
    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    # Process and update all entities
    entities_data = []
    for entity_name, entities in all_nodes.items():
        entity_data = await _merge_nodes_then_upsert(
            entity_name,
            entities,
            knowledge_graph_inst,
            global_config,
            pipeline_status,
            pipeline_status_lock,
            llm_response_cache,
        )
        if entity_data:
            entities_data.append(entity_data)

    # Process and update all relationships
    relationships_data = []
    for edge_key, edges in all_edges.items():
        edge_data = await _merge_edges_then_upsert(
            edge_key[0],
            edge_key[1],
            edges,
            knowledge_graph_inst,
            global_config,
            pipeline_status,
            pipeline_status_lock,
            llm_response_cache,
        )
        if edge_data:
            relationships_data.append(edge_data)

    # Update vector databases with all collected data
    if entity_vdb is not None and entities_data:
        data_for_vdb = {
            compute_hash_id(dp["entity_name"], prefix="ent-"): {
                "entity_name": dp["entity_name"],
                "entity_type": dp["entity_type"],
                "content": f"{dp['entity_name']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "file_path": dp.get("file_path", "unknown_source"),
            }
            for dp in entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None and relationships_data:
        data_for_vdb = {
            compute_hash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "keywords": dp["keywords"],
                "content": f"{dp['src_id']}\t{dp['tgt_id']}\n{dp['keywords']}\n{dp['description']}",
                "source_id": dp["source_id"],
                "file_path": dp.get("file_path", "unknown_source"),
            }
            for dp in relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)