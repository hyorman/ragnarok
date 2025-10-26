/**
 * Agent Orchestrator for Agentic RAG
 * Manages multi-step retrieval, query planning, and result evaluation
 */

import * as vscode from 'vscode';
import { EmbeddingService } from './embeddingService';
import { VectorDatabaseService } from './vectorDatabase';
import { QueryPlanner, QueryPlan } from './queryPlanner';
import { ResultEvaluator, EvaluationResult } from './resultEvaluator';
import { LLMQueryPlanner } from './llmQueryPlanner';
import { LLMResultEvaluator } from './llmResultEvaluator';
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
  private llmQueryPlanner: LLMQueryPlanner;
  private llmResultEvaluator: LLMResultEvaluator;
  private retrievalStrategies: Map<string, RetrievalStrategy>;

  constructor() {
    this.embeddingService = EmbeddingService.getInstance();
    this.vectorDb = VectorDatabaseService.getInstance();

    // Initialize both heuristic and LLM-based components
    this.heuristicQueryPlanner = new QueryPlanner();
    this.heuristicResultEvaluator = new ResultEvaluator();
    this.llmQueryPlanner = new LLMQueryPlanner();
    this.llmResultEvaluator = new LLMResultEvaluator();

    this.retrievalStrategies = new Map();

    // Register available retrieval strategies
    this.retrievalStrategies.set('vector', new VectorSearchStrategy(this.vectorDb, this.embeddingService));
    this.retrievalStrategies.set('hybrid', new HybridSearchStrategy(this.vectorDb, this.embeddingService));
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
      recentQueries?: string[];
    },
    workspaceContext?: WorkspaceContext
  ): Promise<AgenticSearchResult> {
    const steps: AgenticSearchStep[] = [];
    let allResults: SearchResult[] = [];
    let currentIteration = 0;

    // Select planner and evaluator based on config
    const queryPlanner = config.useLLM ? this.llmQueryPlanner : this.heuristicQueryPlanner;
    const resultEvaluator = config.useLLM ? this.llmResultEvaluator : this.heuristicResultEvaluator;

    // Step 1: Query Planning (with context if using LLM)
    const queryPlan = config.useLLM
      ? await this.llmQueryPlanner.createPlan(query, config.enableQueryDecomposition, context, workspaceContext)
      : await this.heuristicQueryPlanner.createPlan(query, config.enableQueryDecomposition);

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

      const evaluation = await (config.useLLM
        ? this.llmResultEvaluator.evaluate(subQuery.query, searchResults, config.confidenceThreshold, evalContext)
        : this.heuristicResultEvaluator.evaluate(subQuery.query, searchResults, config.confidenceThreshold)
      );

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
        const followUpQuery = await queryPlanner.generateFollowUpQuery(
          query,
          allResults,
          evaluation.gaps
        );

        if (followUpQuery && currentIteration < config.maxIterations) {
          currentIteration++;
          const followUpResults = await strategy.search(topicId, followUpQuery.query, 5);

          const followUpEvalContext = config.useLLM && context ? {
            topicName: context.topicName,
            previousSteps: steps.map(s => `Step ${s.stepNumber}: ${s.query} (${s.resultsCount} results)`),
          } : undefined;

          const followUpEval = await (config.useLLM
            ? this.llmResultEvaluator.evaluate(followUpQuery.query, followUpResults, config.confidenceThreshold, followUpEvalContext)
            : this.heuristicResultEvaluator.evaluate(followUpQuery.query, followUpResults, config.confidenceThreshold)
          );

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

    const finalEvaluation = await (config.useLLM
      ? this.llmResultEvaluator.evaluate(query, finalResults, config.confidenceThreshold, finalEvalContext)
      : this.heuristicResultEvaluator.evaluate(query, finalResults, config.confidenceThreshold)
    );

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
    // Deduplicate by chunk ID
    const seen = new Set<string>();
    const uniqueResults: SearchResult[] = [];

    for (const result of results) {
      if (!seen.has(result.chunk.id)) {
        seen.add(result.chunk.id);
        uniqueResults.push(result);
      }
    }

    // Re-rank by similarity (already sorted from individual searches)
    // Could add more sophisticated re-ranking here (e.g., using cross-encoder)
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

