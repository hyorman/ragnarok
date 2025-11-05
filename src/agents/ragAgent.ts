/**
 * RAG Agent - Main orchestrator for Agentic RAG
 * Coordinates query planning, retrieval, and iterative refinement
 *
 * Architecture: Agent pattern with confidence-based iteration
 * Integrates: QueryPlannerAgent + HybridRetriever + Result Evaluation
 */

import * as vscode from 'vscode';
import { VectorStore } from '@langchain/core/vectorstores';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { QueryPlannerAgent, QueryPlan, SubQuery } from './queryPlannerAgent';
import { HybridRetriever, HybridSearchResult } from '../retrievers/hybridRetriever';
import { Logger } from '../utils/logger';
import { CONFIG } from '../constants';

export interface RAGAgentOptions {
  /** Topic name for context */
  topicName?: string;

  /** Workspace context */
  workspaceContext?: string;

  /** Enable iterative refinement */
  enableIterativeRefinement?: boolean;

  /** Maximum iterations */
  maxIterations?: number;

  /** Confidence threshold (0-1) */
  confidenceThreshold?: number;

  /** Use LLM for query planning */
  useLLM?: boolean;

  /** Retrieval strategy */
  retrievalStrategy?: 'vector' | 'hybrid';

  /** Default topK */
  topK?: number;
}

export interface RetrievalResult {
  document: LangChainDocument;
  score: number;
  source: 'vector' | 'hybrid' | 'keyword';
  subQuery?: string;
  explanation?: string;
}

export interface RAGResult {
  /** Original query */
  query: string;

  /** Query plan used */
  plan: QueryPlan;

  /** Retrieved documents */
  results: RetrievalResult[];

  /** Number of iterations performed */
  iterations: number;

  /** Average confidence score */
  avgConfidence: number;

  /** Whether confidence threshold was met */
  confidenceMet: boolean;

  /** Total execution time */
  executionTime: number;

  /** Metadata about the search */
  metadata: {
    totalResults: number;
    uniqueDocuments: number;
    strategy: string;
    subQueriesExecuted: number;
  };
}

/**
 * Main RAG Agent orchestrator
 */
export class RAGAgent {
  private logger: Logger;
  private queryPlanner: QueryPlannerAgent;
  private retriever: HybridRetriever | null = null;
  private vectorStore: VectorStore | null = null;

  // Configuration from VS Code settings
  private enableIterativeRefinement: boolean;
  private maxIterations: number;
  private confidenceThreshold: number;
  private retrievalStrategy: 'vector' | 'hybrid';
  private useLLM: boolean;

  constructor() {
    this.logger = new Logger('RAGAgent');
    this.queryPlanner = new QueryPlannerAgent();

    // Load configuration
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    this.enableIterativeRefinement = config.get<boolean>(
      CONFIG.AGENTIC_ITERATIVE_REFINEMENT,
      true
    );
    this.maxIterations = config.get<number>(CONFIG.AGENTIC_MAX_ITERATIONS, 3);
    this.confidenceThreshold = config.get<number>(
      CONFIG.AGENTIC_CONFIDENCE_THRESHOLD,
      0.7
    );
    this.retrievalStrategy = config.get<string>(
      CONFIG.AGENTIC_RETRIEVAL_STRATEGY,
      'hybrid'
    ) as 'vector' | 'hybrid';
    this.useLLM = config.get<boolean>(CONFIG.AGENTIC_USE_LLM, true);

    this.logger.info('RAGAgent initialized', {
      enableIterativeRefinement: this.enableIterativeRefinement,
      maxIterations: this.maxIterations,
      confidenceThreshold: this.confidenceThreshold,
      retrievalStrategy: this.retrievalStrategy,
      useLLM: this.useLLM,
    });
  }

  /**
   * Initialize agent with vector store
   */
  public async initialize(vectorStore: VectorStore): Promise<void> {
    this.logger.info('Initializing RAGAgent with vector store');

    this.vectorStore = vectorStore;
    this.retriever = new HybridRetriever(vectorStore);

    this.logger.info('RAGAgent initialized successfully');
  }

  /**
   * Execute RAG query with agentic capabilities
   */
  public async query(
    query: string,
    options: RAGAgentOptions = {}
  ): Promise<RAGResult> {
    const startTime = Date.now();

    this.logger.info('Starting RAG query', {
      query: query.substring(0, 100),
      options,
    });

    try {
      // Ensure initialized
      if (!this.retriever || !this.vectorStore) {
        throw new Error('RAGAgent not initialized. Call initialize() first.');
      }

      // Merge options with config
      const mergedOptions = this.mergeOptions(options);

      // Step 1: Create query plan
      const plan = await this.createQueryPlan(query, mergedOptions);

      this.logger.info('Query plan created', {
        complexity: plan.complexity,
        subQueries: plan.subQueries.length,
        strategy: plan.strategy,
      });

      // Step 2: Execute retrieval (with or without iteration)
      let results: RetrievalResult[];
      let iterations = 1;
      let avgConfidence = 0;
      let confidenceMet = false;

      if (mergedOptions.enableIterativeRefinement && plan.complexity !== 'simple') {
        // Iterative retrieval with confidence checking
        const iterativeResult = await this.iterativeRetrieval(
          plan,
          mergedOptions
        );
        results = iterativeResult.results;
        iterations = iterativeResult.iterations;
        avgConfidence = iterativeResult.avgConfidence;
        confidenceMet = iterativeResult.confidenceMet;
      } else {
        // Single-shot retrieval
        results = await this.executeRetrieval(plan, mergedOptions);
        avgConfidence = this.calculateAvgConfidence(results);
        confidenceMet = avgConfidence >= mergedOptions.confidenceThreshold!;
      }

      // Step 3: Deduplicate and rank results
      const uniqueResults = this.deduplicateResults(results);
      const rankedResults = this.rankResults(uniqueResults);

      // Step 4: Limit to topK
      const topK = mergedOptions.topK || 5;
      const finalResults = rankedResults.slice(0, topK);

      const executionTime = Date.now() - startTime;

      const ragResult: RAGResult = {
        query,
        plan,
        results: finalResults,
        iterations,
        avgConfidence,
        confidenceMet,
        executionTime,
        metadata: {
          totalResults: results.length,
          uniqueDocuments: uniqueResults.length,
          strategy: mergedOptions.retrievalStrategy!,
          subQueriesExecuted: plan.subQueries.length,
        },
      };

      this.logger.info('RAG query completed', {
        resultCount: finalResults.length,
        iterations,
        avgConfidence,
        confidenceMet,
        executionTime,
      });

      return ragResult;
    } catch (error) {
      this.logger.error('RAG query failed', {
        error: error instanceof Error ? error.message : String(error),
        query: query.substring(0, 100),
      });
      throw error;
    }
  }

  /**
   * Execute simple query without planning (fast path)
   */
  public async simpleQuery(
    query: string,
    topK: number = 5
  ): Promise<RetrievalResult[]> {
    this.logger.debug('Executing simple query', { query, topK });

    if (!this.retriever) {
      throw new Error('RAGAgent not initialized');
    }

    try {
      const searchResults =
        this.retrievalStrategy === 'hybrid'
          ? await this.retriever.search(query, { k: topK })
          : await this.retriever.vectorSearch(query, topK);

      return searchResults.map((result) => ({
        document: result.document,
        score: result.score,
        source: this.retrievalStrategy,
        explanation: result.explanation,
      }));
    } catch (error) {
      this.logger.error('Simple query failed', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  // ==================== Private Methods ====================

  /**
   * Create query plan using QueryPlannerAgent
   */
  private async createQueryPlan(
    query: string,
    options: RAGAgentOptions
  ): Promise<QueryPlan> {
    return await this.queryPlanner.createPlan(query, {
      topicName: options.topicName,
      workspaceContext: options.workspaceContext,
      maxSubQueries: this.maxIterations,
      defaultTopK: options.topK || 5,
      useLLM: options.useLLM,
    });
  }

  /**
   * Execute retrieval for all sub-queries in the plan
   */
  private async executeRetrieval(
    plan: QueryPlan,
    options: RAGAgentOptions
  ): Promise<RetrievalResult[]> {
    const allResults: RetrievalResult[] = [];

    if (plan.strategy === 'parallel') {
      // Execute all sub-queries in parallel
      const promises = plan.subQueries.map((subQuery: SubQuery) =>
        this.executeSubQuery(subQuery, options)
      );
      const results = await Promise.all(promises);
      allResults.push(...results.flat());
    } else {
      // Execute sub-queries sequentially
      for (const subQuery of plan.subQueries) {
        const results = await this.executeSubQuery(subQuery, options);
        allResults.push(...results);
      }
    }

    return allResults;
  }

  /**
   * Execute a single sub-query
   */
  private async executeSubQuery(
    subQuery: SubQuery,
    options: RAGAgentOptions
  ): Promise<RetrievalResult[]> {
    if (!this.retriever) {
      throw new Error('Retriever not initialized');
    }

    const topK = subQuery.topK || options.topK || 5;

    this.logger.debug('Executing sub-query', {
      query: subQuery.query,
      topK,
      reasoning: subQuery.reasoning,
    });

    try {
      const searchResults =
        options.retrievalStrategy === 'hybrid'
          ? await this.retriever.search(subQuery.query, { k: topK })
          : await this.retriever.vectorSearch(subQuery.query, topK);

      return searchResults.map((result) => ({
        document: result.document,
        score: result.score,
        source: options.retrievalStrategy || 'hybrid',
        subQuery: subQuery.query,
        explanation: result.explanation,
      }));
    } catch (error) {
      this.logger.error('Sub-query execution failed', {
        error: error instanceof Error ? error.message : String(error),
        subQuery: subQuery.query,
      });
      return [];
    }
  }

  /**
   * Iterative retrieval with confidence checking
   */
  private async iterativeRetrieval(
    initialPlan: QueryPlan,
    options: RAGAgentOptions
  ): Promise<{
    results: RetrievalResult[];
    iterations: number;
    avgConfidence: number;
    confidenceMet: boolean;
  }> {
    const allResults: RetrievalResult[] = [];
    let currentPlan = initialPlan;
    let iterations = 0;
    const maxIter = options.maxIterations || this.maxIterations;
    const threshold = options.confidenceThreshold || this.confidenceThreshold;

    this.logger.debug('Starting iterative retrieval', {
      maxIterations: maxIter,
      threshold,
    });

    while (iterations < maxIter) {
      iterations++;

      // Execute current plan
      const iterResults = await this.executeRetrieval(currentPlan, options);
      allResults.push(...iterResults);

      // Calculate confidence
      const avgConfidence = this.calculateAvgConfidence(allResults);

      this.logger.debug('Iteration complete', {
        iteration: iterations,
        resultCount: iterResults.length,
        avgConfidence,
      });

      // Check if confidence threshold met
      if (avgConfidence >= threshold) {
        this.logger.info('Confidence threshold met', {
          avgConfidence,
          threshold,
          iterations,
        });
        return {
          results: allResults,
          iterations,
          avgConfidence,
          confidenceMet: true,
        };
      }

      // Check if we should continue
      if (iterations >= maxIter) {
        this.logger.info('Max iterations reached', {
          iterations,
          avgConfidence,
        });
        break;
      }

      // Refine query plan for next iteration
      // (In a full implementation, this would use LLM to refine based on gaps)
      // For now, we'll just stop after first iteration
      break;
    }

    const avgConfidence = this.calculateAvgConfidence(allResults);

    return {
      results: allResults,
      iterations,
      avgConfidence,
      confidenceMet: avgConfidence >= threshold,
    };
  }

  /**
   * Deduplicate results based on document content
   */
  private deduplicateResults(results: RetrievalResult[]): RetrievalResult[] {
    const seen = new Set<string>();
    const unique: RetrievalResult[] = [];

    for (const result of results) {
      // Use content hash or chunk ID for deduplication
      const key =
        result.document.metadata.chunkId ||
        result.document.pageContent.substring(0, 100);

      if (!seen.has(key)) {
        seen.add(key);
        unique.push(result);
      }
    }

    this.logger.debug('Deduplicated results', {
      original: results.length,
      unique: unique.length,
    });

    return unique;
  }

  /**
   * Rank results by score
   */
  private rankResults(results: RetrievalResult[]): RetrievalResult[] {
    return results.sort((a, b) => b.score - a.score);
  }

  /**
   * Calculate average confidence from results
   */
  private calculateAvgConfidence(results: RetrievalResult[]): number {
    if (results.length === 0) {
      return 0;
    }

    const sum = results.reduce((acc, r) => acc + r.score, 0);
    return sum / results.length;
  }

  /**
   * Merge options with configuration
   */
  private mergeOptions(options: RAGAgentOptions): Required<RAGAgentOptions> {
    return {
      topicName: options.topicName || '',
      workspaceContext: options.workspaceContext || '',
      enableIterativeRefinement:
        options.enableIterativeRefinement ?? this.enableIterativeRefinement,
      maxIterations: options.maxIterations ?? this.maxIterations,
      confidenceThreshold:
        options.confidenceThreshold ?? this.confidenceThreshold,
      useLLM: options.useLLM ?? this.useLLM,
      retrievalStrategy: options.retrievalStrategy ?? this.retrievalStrategy,
      topK: options.topK ?? 5,
    };
  }

  /**
   * Update vector store (useful for switching topics)
   */
  public setVectorStore(vectorStore: VectorStore): void {
    this.vectorStore = vectorStore;
    this.retriever = new HybridRetriever(vectorStore);
    this.logger.debug('Vector store updated');
  }

  /**
   * Get current configuration
   */
  public getConfig(): {
    enableIterativeRefinement: boolean;
    maxIterations: number;
    confidenceThreshold: number;
    retrievalStrategy: string;
    useLLM: boolean;
  } {
    return {
      enableIterativeRefinement: this.enableIterativeRefinement,
      maxIterations: this.maxIterations,
      confidenceThreshold: this.confidenceThreshold,
      retrievalStrategy: this.retrievalStrategy,
      useLLM: this.useLLM,
    };
  }
}
