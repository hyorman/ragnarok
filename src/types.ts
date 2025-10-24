/**
 * Core types for the RAG extension
 */

export interface Topic {
  id: string;
  name: string;
  description?: string;
  createdAt: number;
  updatedAt: number;
  documentCount: number;
}

export interface Document {
  id: string;
  topicId: string;
  name: string;
  filePath: string;
  fileType: 'pdf' | 'markdown' | 'html';
  addedAt: number;
  chunkCount: number;
}

export interface TextChunk {
  id: string;
  documentId: string;
  topicId: string;
  text: string;
  embedding: number[];
  metadata: {
    documentName: string;
    chunkIndex: number;
    startPosition: number;
    endPosition: number;
    headingPath?: string[];  // Hierarchical path: ["Memory Allocation", "Malloc"]
    headingLevel?: number;   // Heading level (1-6)
    sectionTitle?: string;   // Direct parent heading
  };
}

export interface VectorDatabase {
  topics: { [topicId: string]: Topic };
  documents: { [documentId: string]: Document };
  chunks: { [chunkId: string]: TextChunk };
  modelName: string;
  lastUpdated: number;
}

export interface SearchResult {
  chunk: TextChunk;
  similarity: number;
  documentName: string;
}

export interface RAGQueryParams {
  topic: string;
  query: string;
  topK?: number;
}

export interface RAGQueryResult {
  results: Array<{
    text: string;
    documentName: string;
    similarity: number;
    metadata: {
      chunkIndex: number;
      position: string;
      headingPath?: string;  // e.g., "Memory Allocation → Malloc"
      sectionTitle?: string;
    };
  }>;
  query: string;
  topicName: string;
  topicMatched: 'exact' | 'similar' | 'fallback';
  requestedTopic?: string;
  availableTopics?: string[];
}

