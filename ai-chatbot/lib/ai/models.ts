export const DEFAULT_CHAT_MODEL: string = 'hybrid-search';

export interface ChatModel {
  id: string;
  name: string;
  description: string;
}

export const chatModels: Array<ChatModel> = [
  {
    id: 'hybrid-search',
    name: 'Hybrid Search',
    description: 'Combines vector and graph search capabilities',
  },
  {
    id: 'vector-search',
    name: 'Vector Search',
    description: 'Semantic search using vector embeddings',
  },
  {
    id: 'graph-search',
    name: 'Graph Search',
    description: 'Knowledge graph based search',
  },
  {
    id: 'sparse-search',
    name: 'Sparse Search',
    description: 'Keyword-based search using BM25 algorithm',
  },
];
