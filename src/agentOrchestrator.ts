/**
 * Agent Orchestrator for Agentic RAG
 * Manages multi-step retrieval, query planning, and result evaluation
 */

import { EmbeddingService } from './embeddingService';
import { VectorDatabaseService } from './vectorDatabase';
import { QueryPlanner } from './queryPlanner';
import { QueryPlan, IQueryPlanner } from './queryPlannerBase';
import { ResultEvaluator } from './resultEvaluator';
import { IResultEvaluator } from './resultEvaluatorBase';
import { LLMQueryPlanner } from './llmQueryPlanner';
import { LLMResultEvaluator } from './llmResultEvaluator';
import { getChatModel } from './llmProvider';
import { RetrievalStrategy, VectorSearchStrategy, HybridSearchStrategy } from './retrievalStrategy';
import { SearchResult } from './types';
import { WorkspaceContext } from './workspaceContext';

export interface AgenticRAGConfig {
  maxIterations: number;
  confidenceThreshold: number;
  enableQueryDecomposition: boolean;
  enableIterativeRefinement: boolean;
  retrievalStrategy: 'vector' | 'hybrid';
  useLLM: boolean;  // Use LLM for planning and evaluation
}

export interface AgenticSearchStep {
  stepNumber: number;
  query: string;
  strategy: string;
  resultsCount: number;
  confidence: number;
  reasoning: string;
}

export interface AgenticSearchResult {
  finalResults: SearchResult[];
  steps: AgenticSearchStep[];
  totalIterations: number;
  queryPlan: QueryPlan;
  confidence: number;
}

export class AgentOrchestrator {
  private embeddingService: EmbeddingService;
  private vectorDb: VectorDatabaseService;
  private heuristicQueryPlanner: QueryPlanner;
  private heuristicResultEvaluator: ResultEvaluator;
  private _llmQueryPlanner?: LLMQueryPlanner;
  private _llmResultEvaluator?: LLMResultEvaluator;
  private activePlanner?: IQueryPlanner;
  private activeEvaluator?: IResultEvaluator;
  private retrievalStrategies: Map<string, RetrievalStrategy>;

  constructor() {
    this.embeddingService = EmbeddingService.getInstance();
    this.vectorDb = VectorDatabaseService.getInstance();

    // Initialize both heuristic and LLM-based components
    this.heuristicQueryPlanner = new QueryPlanner();
    this.heuristicResultEvaluator = new ResultEvaluator();

    this.retrievalStrategies = new Map();

    // Register available retrieval strategies
    this.retrievalStrategies.set('vector', new VectorSearchStrategy(this.vectorDb, this.embeddingService));
    this.retrievalStrategies.set('hybrid', new HybridSearchStrategy(this.vectorDb, this.embeddingService));
  }

  /** Lazily instantiate LLM Query Planner on first access */
  private get llmQueryPlanner(): LLMQueryPlanner {
    if (!this._llmQueryPlanner) {
      this._llmQueryPlanner = new LLMQueryPlanner();
    }
    return this._llmQueryPlanner;
  }

  /** Lazily instantiate LLM Result Evaluator on first access */
  private get llmResultEvaluator(): LLMResultEvaluator {
    if (!this._llmResultEvaluator) {
      this._llmResultEvaluator = new LLMResultEvaluator();
    }
    return this._llmResultEvaluator;
  }

  /**
   * Execute an agentic RAG query with multi-step planning and retrieval
   */
  public async executeAgenticQuery(
    topicId: string,
    query: string,
    config: AgenticRAGConfig,
    context?: {
      topicName?: string;
      topicDescription?: string;
      documentCount?: number;
      recentQueries?: string[]; // TODO how to populate this
    },
    workspaceContext?: WorkspaceContext
  ): Promise<AgenticSearchResult> {
    const steps: AgenticSearchStep[] = [];
    let allResults: SearchResult[] = [];
    let currentIteration = 0;

    // Select planner and evaluator based on config. If LLM use is requested
    // but no chat model is available, fall back to the heuristic planner/evaluator
    if (config.useLLM) {
      const model = await getChatModel();
      if (model) {
        this.activePlanner = this.llmQueryPlanner;
        this.activeEvaluator = this.llmResultEvaluator;
      } else {
        console.warn('Agentic LLM requested but no chat model available; using heuristic planner/evaluator');
        this.activePlanner = this.heuristicQueryPlanner;
        this.activeEvaluator = this.heuristicResultEvaluator;
      }
    } else {
      this.activePlanner = this.heuristicQueryPlanner;
      this.activeEvaluator = this.heuristicResultEvaluator;
    }

    // Step 1: Query Planning (with context if using LLM)
    let queryPlan: QueryPlan;
    if (config.enableQueryDecomposition) {
      try {
        queryPlan = await this.activePlanner!.createPlan(query, context, workspaceContext);
      } catch (planErr) {
        console.warn('Planner failed, falling back to single-query plan:', planErr);
        queryPlan = this.activePlanner!.fallbackSingleQueryPlan(query);
      }
    } else {
        queryPlan = this.activePlanner!.fallbackSingleQueryPlan(query);
    }

    // Step 2: Execute retrieval for each sub-query
    for (const subQuery of queryPlan.subQueries) {
      if (currentIteration >= config.maxIterations) {
        break;
      }

      currentIteration++;

      // Choose retrieval strategy
      const strategy = this.retrievalStrategies.get(config.retrievalStrategy);
      if (!strategy) {
        throw new Error(`Unknown retrieval strategy: ${config.retrievalStrategy}`);
      }

      // Execute search
      const searchResults = await strategy.search(topicId, subQuery.query, subQuery.topK || 5);

      // Evaluate results (with context if using LLM)
      const evalContext = config.useLLM && context ? {
        topicName: context.topicName,
        previousSteps: steps.map(s => `Step ${s.stepNumber}: ${s.query} (${s.resultsCount} results)`),
      } : undefined;

      const evaluation = await this.activeEvaluator!.evaluate(subQuery.query, searchResults, config.confidenceThreshold, evalContext);

      // Record step
      steps.push({
        stepNumber: currentIteration,
        query: subQuery.query,
        strategy: config.retrievalStrategy,
        resultsCount: searchResults.length,
        confidence: evaluation.confidence,
        reasoning: subQuery.reasoning,
      });

      // Add results to collection
      allResults.push(...searchResults);

      // Iterative refinement: check if we need more information
        if (config.enableIterativeRefinement && !evaluation.isComplete) {
        // Generate follow-up query based on gaps
          let followUpQuery;
          try {
              followUpQuery = await this.activePlanner!.generateFollowUpQuery(
              query,
              allResults,
              evaluation.gaps
            );
            } catch (followErr) {
              console.warn('Planner follow-up generation failed; using fallback:', followErr);
              // Use planner's public fallback helper
              followUpQuery = this.activePlanner!.fallbackFollowUpQuery(query, evaluation.gaps);
          }

        if (followUpQuery && currentIteration < config.maxIterations) {
          currentIteration++;
          const followUpResults = await strategy.search(topicId, followUpQuery.query, 5);

          const followUpEvalContext = config.useLLM && context ? {
            topicName: context.topicName,
            previousSteps: steps.map(s => `Step ${s.stepNumber}: ${s.query} (${s.resultsCount} results)`),
          } : undefined;

          const followUpEval = await this.activeEvaluator!.evaluate(followUpQuery.query, followUpResults, config.confidenceThreshold, followUpEvalContext);

          steps.push({
            stepNumber: currentIteration,
            query: followUpQuery.query,
            strategy: config.retrievalStrategy,
            resultsCount: followUpResults.length,
            confidence: followUpEval.confidence,
            reasoning: 'Follow-up query to address information gaps',
          });

          allResults.push(...followUpResults);
        }
      }

      // Check if we have sufficient information
      if (evaluation.isComplete && evaluation.confidence >= config.confidenceThreshold) {
        break;
      }
    }

    // Step 3: Deduplicate and re-rank results
    const finalResults = this.deduplicateAndRerank(allResults, query);

    // Step 4: Final evaluation
    const finalEvalContext = config.useLLM && context ? {
      topicName: context.topicName,
      previousSteps: steps.map(s => `Step ${s.stepNumber}: ${s.query} (${s.resultsCount} results, confidence: ${s.confidence})`),
    } : undefined;

    const finalEvaluation = await this.activeEvaluator!.evaluate(query, finalResults, config.confidenceThreshold, finalEvalContext);

    return {
      finalResults: finalResults.slice(0, 10), // Top 10 overall
      steps,
      totalIterations: currentIteration,
      queryPlan,
      confidence: finalEvaluation.confidence,
    };
  }

  /**
   * Deduplicate results and re-rank based on relevance to original query
   */
  private deduplicateAndRerank(results: SearchResult[], originalQuery: string): SearchResult[] {
    // Deduplicate by chunk ID.
    // NOTE: `originalQuery` is currently kept as a placeholder for future re-ranking
    // (e.g., computing an embedding for the original query and re-scoring chunks).
    // For now we use the provided per-search similarity scores, but when multiple
    // searches return the same chunk with different similarity values we want to
    // keep the best-scoring instance. The previous implementation kept the first
    // encountered instance which could drop a higher-scoring duplicate that
    // appeared later (from a different sub-query/iteration).

    const bestByChunk = new Map<string, SearchResult>();

    for (const result of results) {
      const id = result.chunk.id;
      const prev = bestByChunk.get(id);
      if (!prev || (typeof result.similarity === 'number' && result.similarity > prev.similarity)) {
        bestByChunk.set(id, result);
      }
    }

    const uniqueResults = Array.from(bestByChunk.values());

    // Re-rank by similarity (descending). In future we may re-rank using the
    // `originalQuery` by computing a fresh similarity score (embedding or
    // cross-encoder) which would likely improve final ordering.
    return uniqueResults.sort((a, b) => b.similarity - a.similarity);
  }

  /**
   * Simple non-agentic query for backward compatibility
   */
  public async executeSimpleQuery(
    topicId: string,
    query: string,
    topK: number
  ): Promise<SearchResult[]> {
    const strategy = this.retrievalStrategies.get('vector');
    if (!strategy) {
      throw new Error('Vector search strategy not available');
    }
    return strategy.search(topicId, query, topK);
  }
}

