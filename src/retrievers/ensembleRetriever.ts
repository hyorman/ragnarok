/**
 * Ensemble Retriever - Combines multiple retrieval strategies using Reciprocal Rank Fusion
 * Manual implementation since langchain's EnsembleRetriever is not yet exported
 *
 * Architecture: Uses custom RRF implementation with BM25 + Vector search
 * Benefits: More robust to different score scales, better fusion algorithm
 * Tradeoff: BM25 requires loading all documents in memory (less scalable than HybridRetriever)
 */

import { VectorStore } from '@langchain/core/vectorstores';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { Logger } from '../utils/logger';

export interface EnsembleSearchOptions {
  /** Number of results to return */
  k?: number;

  /** Weight for vector retriever (0-1, default 0.5) */
  vectorWeight?: number;

  /** Weight for BM25 retriever (0-1, default 0.5) */
  bm25Weight?: number;
}

export interface EnsembleSearchResult {
  document: LangChainDocument;
  score?: number; // Note: EnsembleRetriever doesn't return scores
}

/**
 * Ensemble retriever using RRF to combine vector and BM25 search
 */
export class EnsembleRetrieverWrapper {
  private logger: Logger;
  private vectorStore: VectorStore;
  private bm25Retriever?: BM25Retriever;
  private documents: LangChainDocument[] = [];

  private readonly DEFAULT_K = 5;
  private readonly DEFAULT_VECTOR_WEIGHT = 0.5;
  private readonly DEFAULT_BM25_WEIGHT = 0.5;
  private readonly RRF_CONSTANT = 60; // Standard RRF constant

  constructor(vectorStore: VectorStore) {
    this.logger = new Logger('EnsembleRetriever');
    this.vectorStore = vectorStore;
    this.logger.info('EnsembleRetriever initialized');
  }

  /**
   * Initialize the ensemble retriever by loading all documents from vector store
   * This is required for BM25 to work (it needs all docs in memory)
   */
  public async initialize(documents?: LangChainDocument[]): Promise<void> {
    this.logger.info('Initializing ensemble retriever with BM25');

    if (documents && documents.length > 0) {
      this.documents = documents;
    } else {
      // Fetch documents from vector store
      // Note: This requires fetching ALL documents which may be memory-intensive
      this.logger.warn('No documents provided, fetching from vector store. This may be slow for large datasets.');

      try {
        // Fetch a large number of documents (approximate all)
        // This is a limitation of BM25 - it needs all docs in memory
        const allDocs = await this.vectorStore.similaritySearch('', 10000);
        this.documents = allDocs;
        this.logger.info('Loaded documents from vector store', { count: allDocs.length });
      } catch (error) {
        this.logger.error('Failed to load documents from vector store', error);
        throw new Error('Failed to initialize EnsembleRetriever: could not load documents');
      }
    }

    if (this.documents.length === 0) {
      throw new Error('EnsembleRetriever requires documents to initialize BM25');
    }

    // Create BM25 retriever from documents
    this.bm25Retriever = BM25Retriever.fromDocuments(this.documents, {
      k: this.DEFAULT_K,
    });

    this.logger.info('Ensemble retriever initialized successfully', {
      documentCount: this.documents.length,
    });
  }

  /**
   * Perform ensemble search using manual RRF
   */
  public async search(
    query: string,
    options: EnsembleSearchOptions = {}
  ): Promise<EnsembleSearchResult[]> {
    if (!this.bm25Retriever) {
      throw new Error('EnsembleRetriever not initialized. Call initialize() first.');
    }

    const startTime = Date.now();
    const k = options.k || this.DEFAULT_K;
    const vectorWeight = options.vectorWeight ?? this.DEFAULT_VECTOR_WEIGHT;
    const bm25Weight = options.bm25Weight ?? this.DEFAULT_BM25_WEIGHT;

    this.logger.info('Starting ensemble search with manual RRF', {
      query: query.substring(0, 100),
      k,
      vectorWeight,
      bm25Weight,
    });

    try {
      // Fetch extra documents for re-ranking
      const fetchCount = k * 3;

      // Get results from both retrievers
      const [vectorResults, bm25Results] = await Promise.all([
        this.vectorStore.similaritySearch(query, fetchCount),
        this.bm25Retriever.invoke(query),
      ]);

      // Apply Reciprocal Rank Fusion (RRF)
      const fusedResults = this.reciprocalRankFusion(
        vectorResults,
        bm25Results.slice(0, fetchCount),
        vectorWeight,
        bm25Weight
      );

      // Limit to k results
      const limitedResults = fusedResults.slice(0, k);

      const searchTime = Date.now() - startTime;

      this.logger.info('Ensemble search complete', {
        resultCount: limitedResults.length,
        searchTime,
      });

      // Convert to our result format
      return limitedResults.map((doc: LangChainDocument) => ({
        document: doc,
        // Note: RRF doesn't provide meaningful scores
      }));
    } catch (error) {
      this.logger.error('Ensemble search failed', {
        error: error instanceof Error ? error.message : String(error),
        query: query.substring(0, 100),
      });
      throw error;
    }
  }

  // ==================== Private Methods ====================

  /**
   * Reciprocal Rank Fusion algorithm
   * Combines rankings from multiple retrievers
   */
  private reciprocalRankFusion(
    vectorResults: LangChainDocument[],
    bm25Results: LangChainDocument[],
    vectorWeight: number,
    bm25Weight: number
  ): LangChainDocument[] {
    // Create a map of document ID -> RRF score
    const scoreMap = new Map<string, { doc: LangChainDocument; score: number }>();

    // Add vector results with their ranks
    vectorResults.forEach((doc, index) => {
      const docId = this.getDocumentId(doc);
      const rrf = vectorWeight / (this.RRF_CONSTANT + index + 1);

      if (scoreMap.has(docId)) {
        scoreMap.get(docId)!.score += rrf;
      } else {
        scoreMap.set(docId, { doc, score: rrf });
      }
    });

    // Add BM25 results with their ranks
    bm25Results.forEach((doc, index) => {
      const docId = this.getDocumentId(doc);
      const rrf = bm25Weight / (this.RRF_CONSTANT + index + 1);

      if (scoreMap.has(docId)) {
        scoreMap.get(docId)!.score += rrf;
      } else {
        scoreMap.set(docId, { doc, score: rrf });
      }
    });

    // Sort by RRF score (descending)
    const rankedResults = Array.from(scoreMap.values())
      .sort((a, b) => b.score - a.score)
      .map((item) => item.doc);

    return rankedResults;
  }

  /**
   * Get a unique ID for a document
   */
  private getDocumentId(doc: LangChainDocument): string {
    // Use pageContent + metadata as a simple unique identifier
    return `${doc.pageContent.substring(0, 100)}-${JSON.stringify(doc.metadata)}`;
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
   * Update the vector store (useful for testing or switching topics)
   */
  public setVectorStore(vectorStore: VectorStore): void {
    this.vectorStore = vectorStore;
    this.bm25Retriever = undefined; // Force re-initialization
    this.logger.debug('Vector store updated, ensemble retriever needs re-initialization');
  }

  /**
   * Refresh documents from vector store and reinitialize
   */
  public async refresh(): Promise<void> {
    this.logger.info('Refreshing ensemble retriever');
    this.bm25Retriever = undefined;
    this.documents = [];
    await this.initialize();
  }
}
