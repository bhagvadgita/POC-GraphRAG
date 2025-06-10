import json
import numpy as np
import faiss
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import warnings
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

warnings.filterwarnings('ignore')

# Hardcoded stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
    'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'or', 'but', 'this',
    'there', 'which', 'who', 'whom', 'what', 'when', 'where', 'how', 'all', 'any', 'both', 'each',
    'more', 'some', 'such', 'than', 'too', 'very', 'can', 'do', 'does', 'doing', 'have', 'having'
}


def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d+)([A-Z])', r'\1 \2', text)
    return text

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text"""
    if not isinstance(text, str):
        text = str(text) if hasattr(text, '__str__') else ""
    
    if not text.strip():
        return []
    
    # Tokenize and filter
    tokens = [token.lower() for token in re.findall(r'\b\w+\b', text) 
             if token.isalpha() and len(token) > 2 and token.lower() not in STOPWORDS]
    
    # Count frequencies
    freq_dist = Counter(tokens)
    
    # Return top keywords
    return [word for word, _ in freq_dist.most_common(5)]

def extract_entities(text: str) -> List[str]:
    """Extract entities from text matching graph entities"""
    if not isinstance(text, str):
        text = str(text) if hasattr(text, '__str__') else ""
    
    if not text.strip():
        return []
    
    # Entity patterns based on graph and query context
    entity_patterns = {
        'finance': ['finance', 'financial'],
        'businesses': ['business', 'businesses', 'commerce'],
        'college': ['college', 'university'],
        'individuals': ['individual', 'individuals', 'person', 'people', 'client', 'clients'],
        'government_entities': ['government', 'entities', 'government entities'],
        'personal_finance': ['personal finance'],
        'corporate_finance': ['corporate finance'],
        'public_finance': ['public finance']
    }
    
    text_lower = text.lower()
    entities = []
    
    for entity, patterns in entity_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            entities.append(entity)
    
    return entities

def extract_topics(text: str) -> List[str]:
    """Extract topics from text matching graph entity types and query context"""
    if not isinstance(text, str):
        text = str(text) if hasattr(text, '__str__') else ""
    
    if not text.strip():
        return []
    
    # Topics based on graph entity types and regulatory concepts
    topic_keywords = {
        'category': ['finance', 'personal finance', 'corporate finance', 'public finance'],
        'organization': ['college', 'university', 'businesses', 'government entities'],
        'person': ['individuals', 'person', 'people', 'client', 'clients'],
        'regulation': ['protection', 'requirements', 'compliance', 'obligation', 'rule']
    }
    
    text_lower = text.lower()
    topics = []
    
    for topic, keywords in topic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score >= 1:
            topics.append(topic)
    
    return topics

class Document:
    """Document class for storing content and metadata"""
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class GraphNode:
    """Node in the knowledge graph"""
    def __init__(self, node_id: str, content: str, node_type: str = "chunk",
                 entities: List[str] = None, topics: List[str] = None):
        self.node_id = node_id
        self.content = content
        self.node_type = node_type
        self.entities = entities or []
        self.topics = topics or []
        self.connections = []
        self.importance_score = 0.0
        self.metadata = {}

class GraphEdge:
    """Edge in the knowledge graph"""
    def __init__(self, source: str, target: str, edge_type: str = "connects", weight: float = 1.0, description: str = ""):
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.weight = weight
        self.description = description

class KnowledgeGraph:
    """Knowledge graph for document relationships"""
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {}
        self.edges = defaultdict(list)
        self.entity_to_nodes = defaultdict(list)
        self.topic_to_nodes = defaultdict(list)

    def load_from_json(self, graph_data: List):
        """Load graph structure from JSON data in chunk_results.json format"""
        try:
            # Clear existing data
            self.graph.clear()
            self.nodes.clear()
            self.edges.clear()
            self.entity_to_nodes.clear()
            self.topic_to_nodes.clear()

            # Track unique entities to avoid duplicates
            seen_entities = set()

            # Process each chunk's data
            for chunk_data in graph_data:
                # Extract nodes and edges from the chunk
                nodes_data, edges_data = chunk_data
                
                # Process nodes
                for entity_name, entity_info_list in nodes_data.items():
                    if entity_name in seen_entities:
                        continue
                    seen_entities.add(entity_name)
                    
                    entity_info = entity_info_list[0]  # Get first item from list
                    node_id = entity_name
                    # Clean entity_type
                    entity_type = entity_info.get("entity_type", "unknown").replace("<", "").replace(">", "")
                    
                    node = GraphNode(
                        node_id=node_id,
                        content=entity_info.get("description", ""),
                        node_type=entity_type,
                        entities=[entity_name],  # Add the entity name itself
                        topics=[entity_type]  # Add entity_type as a topic
                    )
                    node.metadata = {
                        "description": entity_info.get("description", ""),
                        "entity_type": entity_type,
                        "chunk_id": entity_info.get("chunk_id", "")
                    }
                    
                    self.nodes[node_id] = node
                    
                    # Update entity mapping
                    self.entity_to_nodes[entity_name.lower()].append(node_id)
                    
                    # Add entity_type as a topic
                    if entity_type:
                        self.topic_to_nodes[entity_type.lower()].append(node_id)

                # Process edges
                for edge_key, edge_info_list in edges_data.items():
                    if not edge_info_list:  # Skip empty edge lists
                        continue
                        
                    edge_info = edge_info_list[0]  # Get first item from list
                    source = edge_info["src_id"]
                    # Clean target ID
                    target = edge_info["tgt_id"].replace("<", "")
                    
                    # Skip if source or target not in nodes
                    if source not in self.nodes or target not in self.nodes:
                        print(f"Skipping edge {source}->{target}: Node(s) not found")
                        continue
                    
                    # Clean strength value
                    strength = edge_info.get("strength", "0.5")
                    strength = float(re.sub(r'[^\d.]', '', strength))
                    
                    # Create edge
                    edge = GraphEdge(
                        source=source,
                        target=target,
                        edge_type=edge_info.get("keywords", "connects"),
                        weight=strength,
                        description=edge_info.get("description", "")
                    )
                    
                    # Add to graph
                    self.graph.add_edge(source, target, 
                                      type=edge.edge_type,
                                      weight=edge.weight)
                    
                    # Update node connections
                    if source in self.nodes:
                        self.nodes[source].connections.append(target)
                    if target in self.nodes:
                        self.nodes[target].connections.append(source)
                    
                    # Store edge
                    self.edges[source].append(edge)
                    self.edges[target].append(edge)

            # Compute node importance scores
            self.compute_node_importance()

            print(f"Loaded graph with {len(self.nodes)} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            print(f"Error loading graph from JSON: {str(e)}")
            return False

    def compute_node_importance(self):
        """Compute importance scores for nodes based on degree centrality and edge weights"""
        try:
            # Calculate degree centrality
            centrality = nx.degree_centrality(self.graph)
            
            # Update node importance scores
            for node_id in self.nodes:
                # Get edge weights for this node
                edge_weights = sum(edge.weight for edge in self.edges[node_id])
                
                # Combine centrality and edge weights
                self.nodes[node_id].importance_score = (
                    0.5 * centrality.get(node_id, 0) +  # Degree centrality component
                    0.5 * edge_weights  # Edge weight component
                )
                
            print("Computed node importance scores")
        except Exception as e:
            print(f"Error computing node importance: {str(e)}")

    def get_connected_nodes(self, node_id: str, max_depth: int = 2) -> List[str]:
        """Get nodes connected to a given node within max_depth"""
        if node_id not in self.graph:
            return []
        
        connected = set()
        current_level = {node_id}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors = set(self.graph.neighbors(node))
                connected.update(neighbors)
                next_level.update(neighbors)
            current_level = next_level - connected
            if not current_level:
                break
        
        return list(connected)

    def find_nodes_by_entity(self, entity: str) -> List[str]:
        """Find nodes containing a specific entity"""
        return self.entity_to_nodes.get(entity.lower(), [])

    def find_nodes_by_topic(self, topic: str) -> List[str]:
        """Find nodes containing a specific topic"""
        return self.topic_to_nodes.get(topic.lower(), [])

    def get_graph_stats(self) -> Dict:
        """Get basic statistics about the graph"""
        chunk_nodes = sum(1 for node_id in self.nodes if node_id.startswith("chunk_"))
        entity_nodes = sum(1 for node_id in self.nodes if node_id.startswith("entity_") or node_id.startswith("regulation_") or node_id.startswith("organization_") or node_id.startswith("instrument_"))
        topic_nodes = sum(1 for node_id in self.nodes if node_id.startswith("topic_"))
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": self.graph.number_of_edges(),
            "chunk_nodes": chunk_nodes,
            "entity_nodes": entity_nodes,
            "topic_nodes": topic_nodes,
            "avg_connections": np.mean([len(node.connections) for node in self.nodes.values()]) if self.nodes else 0,
            "avg_importance": np.mean([node.importance_score for node in self.nodes.values()]) if self.nodes else 0
        }


class HybridVectorGraphRetriever:
    def __init__(self, chunks: List[Dict], embeddings: List[List[float]], knowledge_graph: KnowledgeGraph,
                 vector_weight: float = 0.4, sparse_weight: float = 0.3, graph_weight: float = 0.3):
        self.chunks = chunks
        self.embeddings = np.array(embeddings).astype("float32")
        self.knowledge_graph = knowledge_graph
        self.vector_weight = vector_weight
        self.sparse_weight = sparse_weight
        self.graph_weight = graph_weight
        
        self.setup_vector_retrieval()
        self.setup_sparse_retrieval()

    def setup_vector_retrieval(self):
        """Setup FAISS vector search index"""
        print("Setting up vector retrieval...")
        dimension = self.embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.vector_index.add(self.embeddings)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✓ Vector index ready with {self.vector_index.ntotal} vectors")

    def setup_sparse_retrieval(self):
        """Setup BM25 sparse retrieval"""
        print("Setting up sparse (BM25) retrieval...")
        texts = [chunk["text"] for chunk in self.chunks]
        tokenized_texts = []
        
        for text in texts:
            try:
                tokens = re.findall(r'\b\w+\b', text.lower())
                filtered_tokens = [token for token in tokens 
                                 if token.isalpha() and len(token) > 2 and token not in STOPWORDS]
                tokenized_texts.append(filtered_tokens)
            except Exception as e:
                print(f"Error processing text for BM25: {e}")
                tokenized_texts.append(text.lower().split())
        
        self.bm25 = BM25Okapi(tokenized_texts)
        print(f"BM25 index ready with {len(tokenized_texts)} documents")

    def vector_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform vector similarity search"""
        try:
            query_embedding = self.embedding_model.encode([query])
            query_vector = np.array(query_embedding).astype("float32")
            faiss.normalize_L2(query_vector)
            scores, indices = self.vector_index.search(query_vector, k)
            
            documents = []
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    node_id = f"chunk_{idx}"
                    node = self.knowledge_graph.nodes.get(node_id)
                    
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
                            "vector": True,
                            "sparse": False,
                            "graph": False
                        }
                    }
                    
                    doc = Document(page_content=chunk["text"], metadata=metadata)
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def sparse_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Perform BM25 sparse search"""
        try:
            query_tokens = [token.lower() for token in re.findall(r'\b\w+\b', query) 
                           if token.isalpha() and len(token) > 2 and token.lower() not in STOPWORDS]
            
            if not query_tokens:
                return []
            
            bm25_scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            return [(int(idx), float(bm25_scores[idx])) for idx in top_indices if bm25_scores[idx] > 0]
        except Exception as e:
            print(f"Sparse search error: {e}")
            return []

    def graph_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform graph-based search using entities, topics, and graph structure"""
        try:
            query_entities = extract_entities(query)
            query_topics = extract_topics(query)
            query_keywords = extract_keywords(query)
            
            print(f"Query features: Entities={query_entities}, Topics={query_topics}, Keywords={query_keywords}")
            
            relevant_nodes = set()
            for entity in query_entities:
                nodes = self.knowledge_graph.find_nodes_by_entity(entity)
                print(f"Entity '{entity}': {len(nodes)} nodes")
                relevant_nodes.update(nodes)
            
            for topic in query_topics:
                nodes = self.knowledge_graph.find_nodes_by_topic(topic)
                print(f"Topic '{topic}': {len(nodes)} nodes")
                relevant_nodes.update(nodes)
            
            # Enhanced keyword matching
            for keyword in query_keywords:
                for node_id, node in self.knowledge_graph.nodes.items():
                    # Check node content, topics, and metadata
                    node_content = node.content.lower()
                    node_topics = [t.lower() for t in node.metadata.get('topics', [])]
                    # Check edge types and keywords
                    edge_types = ' '.join(e.edge_type.lower() for e in self.knowledge_graph.edges[node_id])
                    if (keyword in node_content or 
                        any(keyword in t for t in node_topics) or
                        keyword in edge_types):
                        relevant_nodes.add(node_id)
                        print(f"Keyword '{keyword}' matched node: {node_id}")
            
            print(f"Total relevant nodes before expansion: {len(relevant_nodes)}")
            
            expanded_nodes = set(relevant_nodes)
            for node_id in list(relevant_nodes):
                connected = self.knowledge_graph.get_connected_nodes(node_id, max_depth=2)
                print(f"Node '{node_id}' expanded to {len(connected)} connected nodes")
                expanded_nodes.update(connected)
            
            print(f"Total nodes after expansion: {len(expanded_nodes)}")
            
            scored_nodes = []
            for node_id in expanded_nodes:
                node = self.knowledge_graph.nodes.get(node_id)
                if node:
                    relevance = 0.0
                    node_entities = node.metadata.get('entities', [])
                    node_topics = node.metadata.get('topics', [])
                    
                    entity_overlap = len(set(query_entities) & set(node_entities))
                    if entity_overlap > 0:
                        relevance += 0.4 * entity_overlap / max(len(query_entities), 1)
                    
                    topic_overlap = len(set(query_topics) & set(node_topics))
                    if topic_overlap > 0:
                        relevance += 0.3 * topic_overlap / max(len(query_topics), 1)
                    
                    keyword_matches = sum(1 for kw in query_keywords 
                                        if kw in node.content.lower() or
                                        any(kw in e.edge_type.lower() for e in self.knowledge_graph.edges[node_id]))
                    if keyword_matches > 0:
                        relevance += 0.2 * keyword_matches / max(len(query_keywords), 1)
                    
                    if node_id in relevant_nodes:
                        relevance += 0.3
                    
                    final_score = 0.7 * relevance + 0.3 * node.importance_score
                    print(f"Node '{node_id}': relevance={relevance:.3f}, importance={node.importance_score:.3f}, final_score={final_score:.3f}")
                    
                    chunk_id = node.metadata.get('chunk_id', '')
                    chunk_idx = None
                    if chunk_id:
                        try:
                            chunk_idx = int(chunk_id.split('_')[1])
                        except (IndexError, ValueError):
                            print(f"Invalid chunk_id for node {node_id}: {chunk_id}")
                            continue
                    
                    if chunk_idx is not None and final_score > 0:
                        scored_nodes.append((chunk_idx, final_score))
            
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            top_nodes = scored_nodes[:k]
            print(f"Found {len(top_nodes)} scored nodes")
            
            documents = []
            for idx, score in top_nodes:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    node_id = next((nid for nid, node in self.knowledge_graph.nodes.items() 
                                   if node.metadata.get('chunk_id') == f'chunk_{idx}'), None)
                    node = self.knowledge_graph.nodes.get(node_id, None)
                    
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
                            "sparse": False,
                            "graph": True
                        }
                    }
                    
                    doc = Document(page_content=chunk["text"], metadata=metadata)
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Graph search error: {e}")
            return []

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform hybrid search combining vector, sparse, and graph methods"""
        print(f"Running hybrid search for query: '{query}'")
        
        # Get results from all three methods
        vector_results = self.vector_search(query, k * 2)
        sparse_results = self.sparse_search(query, k * 2)
        graph_results = self.graph_search(query, k * 2)
        
        print(f"   Vector results: {len(vector_results)} items")
        print(f"   Sparse results: {len(sparse_results)} items")
        print(f"   Graph results: {len(graph_results)} items")
        
        # Adjust weights if graph results are empty
        vector_weight = self.vector_weight
        sparse_weight = self.sparse_weight
        graph_weight = self.graph_weight
        if not graph_results:
            vector_weight = 0.6  # Increase vector weight
            sparse_weight = 0.4  # Increase sparse weight
            graph_weight = 0.0
            print("   No graph results, adjusting weights: vector=0.6, sparse=0.4, graph=0.0")
        
        # Combine and normalize scores
        combined_scores = {}
        
        if vector_results:
            max_vector = max(doc.metadata['hybrid_score'] for doc in vector_results) if vector_results else 1.0
            for doc in vector_results:
                idx = doc.metadata['chunk_id']
                normalized_score = doc.metadata['hybrid_score'] / max_vector
                combined_scores[idx] = combined_scores.get(idx, 0) + vector_weight * normalized_score
        
        if sparse_results:
            max_sparse = max(score for _, score in sparse_results) if sparse_results else 1.0
            for idx, score in sparse_results:
                normalized_score = score / max_sparse
                combined_scores[idx] = combined_scores.get(idx, 0) + sparse_weight * normalized_score
        
        if graph_results:
            max_graph = max(doc.metadata['hybrid_score'] for doc in graph_results) if graph_results else 1.0
            for doc in graph_results:
                idx = doc.metadata['chunk_id']
                normalized_score = doc.metadata['hybrid_score'] / max_graph
                combined_scores[idx] = combined_scores.get(idx, 0) + graph_weight * normalized_score
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        print(f"   Combined results: {len(sorted_results)} items")
        
        documents = []
        for idx, score in sorted_results:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                node_id = next((nid for nid, node in self.knowledge_graph.nodes.items() 
                               if node.metadata.get('chunk_id') == f'chunk_{idx}'), None)
                node = self.knowledge_graph.nodes.get(node_id, None)
                
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
                        "vector": idx in [doc.metadata['chunk_id'] for doc in vector_results],
                        "sparse": idx in [i for i, _ in sparse_results],
                        "graph": idx in [doc.metadata['chunk_id'] for doc in graph_results]
                    }
                }
                
                doc = Document(page_content=chunk["text"], metadata=metadata)
                documents.append(doc)
        
        return documents


def load_data():
    """Load and process data from JSON files"""
    print("Loading and processing data...")
    try:
        # Load data from local files
        with open("graphrag/data/chunks/chunk_data.json", "r") as f:
            chunk_data = json.load(f)
        with open("graphrag/data/embedding/embedding_graph.json", "r") as f:
            embedding_data = json.load(f)
        with open("graphrag/data/intermediate/chunk_results.json", "r") as f:
            graph_data = json.load(f)
        
        processed_chunks = []
        for i, chunk in enumerate(chunk_data):
            processed_text = preprocess_text(chunk["text"])
            keywords = extract_keywords(processed_text)
            enhanced_chunk = {
                "id": i,
                "text": processed_text,
                "original_text": chunk["text"],
                "keywords": keywords,
                "metadata": chunk.get("metadata", {"chunk_id": i})
            }
            processed_chunks.append(enhanced_chunk)
        
        embeddings = [item["embedding"] for item in embedding_data]
        print(f"✓ Loaded {len(processed_chunks)} chunks and {len(embeddings)} embeddings.")
        return processed_chunks, embeddings, graph_data
        
    except FileNotFoundError as e:
        print(f"Error loading data: {str(e)}")
        print("Please ensure the following files exist:")
        print("- graphrag/data/chunks/chunk_data.json")
        print("- graphrag/data/embedding/embedding_graph.json")
        print("- graphrag/data/intermediate/chunk_results.json")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {str(e)}")
        raise

def main():
    """Main function to demonstrate the Hybrid Vector + Graph RAG system"""
    try:
        # Load data
        chunks, embeddings, graph_data = load_data()
        
        # Initialize knowledge graph from loaded data
        knowledge_graph = KnowledgeGraph()
        knowledge_graph.load_from_json(graph_data)
        
        # Print graph statistics
        stats = knowledge_graph.get_graph_stats()
        print("\nKnowledge Graph Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Initialize hybrid retriever
        retriever = HybridVectorGraphRetriever(
            chunks=chunks,
            embeddings=embeddings,
            knowledge_graph=knowledge_graph,
            vector_weight=0.4,
            sparse_weight=0.3,
            graph_weight=0.3
        )
        
        # Example query
        query = "What are the client protection requirements?"
        print(f"\nPerforming hybrid search for query: '{query}'")
        
        # Perform hybrid search
        results = retriever.hybrid_search(query, k=3)
        
        # Display results
        print("\nSearch Results:")
        if not results:
            print("No results found. Check data loading and search methods.")
        else:
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Hybrid Score: {doc.metadata['hybrid_score']:.4f}")
                print(f"Entities: {doc.metadata['entities']}")
                print(f"Topics: {doc.metadata['topics']}")
                print(f"Search Methods: {doc.metadata['search_methods']}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()