"""
Graph Creation Package

This package contains modules for creating and managing knowledge graphs from documents.
"""

from .graph_generation import run_graph_pipeline, extract_entities_from_json
from .merge import merge_nodes_and_edges

__all__ = ['run_graph_pipeline', 'extract_entities_from_json', 'merge_nodes_and_edges'] 