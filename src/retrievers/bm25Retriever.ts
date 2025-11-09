/**
 * BM25 Retriever - Pure keyword-based search using Okapi BM25 algorithm
 * No embeddings required - ideal for exact term matching
 *
 * Use cases: Product codes, technical terms, exact phrase matching
 * Benefits: Fast, no embedding computation, excellent for keyword-heavy queries
 * Limitation: Requires all documents in memory
 */

import { Document as LangChainDocument } from '@langchain/core/documents';
import { BM25Retriever as LangChainBM25Retriever } from '@langchain/community/retrievers/bm25';
import { Logger } from '../utils/logger';

export interface BM25SearchOptions {
  /** Number of results to return */
  k?: number;
}

export interface BM25SearchResult {
  document: LangChainDocument;
  score?: number; // BM25Retriever doesn't return scores
}

/**
 * BM25 retriever for pure keyword search
 */
export class BM25RetrieverWrapper {
  private logger: Logger;
  private bm25Retriever?: LangChainBM25Retriever;
  private documents: LangChainDocument[] = [];

  private readonly DEFAULT_K = 5;

  constructor() {
    this.logger = new Logger('BM25Retriever');
    this.logger.info('BM25Retriever initialized');
  }

  /**
   * Initialize the BM25 retriever with documents
   * All documents must be loaded in memory for BM25 to work
   */
  public async initialize(documents: LangChainDocument[]): Promise<void> {
    this.logger.info('Initializing BM25 retriever', { documentCount: documents.length });

    if (!documents || documents.length === 0) {
      throw new Error('BM25Retriever requires documents to initialize');
    }

    this.documents = documents;

    // Create BM25 retriever from documents
    this.bm25Retriever = LangChainBM25Retriever.fromDocuments(this.documents, {
      k: this.DEFAULT_K,
    });

    this.logger.info('BM25 retriever initialized successfully', {
      documentCount: this.documents.length,
    });
  }

  /**
   * Perform BM25 keyword search
   */
  public async search(
    query: string,
    options: BM25SearchOptions = {}
  ): Promise<BM25SearchResult[]> {
    if (!this.bm25Retriever) {
      throw new Error('BM25Retriever not initialized. Call initialize() first.');
    }

    const startTime = Date.now();
    const k = options.k || this.DEFAULT_K;

    this.logger.info('Starting BM25 search', {
      query: query.substring(0, 100),
      k,
    });

    try {
      // Perform BM25 retrieval
      const results = await this.bm25Retriever.invoke(query);

      // Limit to k results
      const limitedResults = results.slice(0, k);

      const searchTime = Date.now() - startTime;

      this.logger.info('BM25 search complete', {
        resultCount: limitedResults.length,
        searchTime,
      });

      // Convert to our result format
      return limitedResults.map((doc: LangChainDocument) => ({
        document: doc,
        // Note: BM25Retriever doesn't provide scores
      }));
    } catch (error) {
      this.logger.error('BM25 search failed', {
        error: error instanceof Error ? error.message : String(error),
        query: query.substring(0, 100),
      });
      throw error;
    }
  }

  /**
   * Check if the retriever is initialized
   */
  public isInitialized(): boolean {
    return this.bm25Retriever !== undefined;
  }

  /**
   * Get document count
   */
  public getDocumentCount(): number {
    return this.documents.length;
  }

  /**
   * Refresh with new documents
   */
  public async refresh(documents: LangChainDocument[]): Promise<void> {
    this.logger.info('Refreshing BM25 retriever');
    this.bm25Retriever = undefined;
    this.documents = [];
    await this.initialize(documents);
  }
}
