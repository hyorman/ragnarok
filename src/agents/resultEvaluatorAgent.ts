/**
 * Result Evaluator Agent - Evaluates retrieval results and decides if retry is needed
 * Uses LLM or heuristics to assess result quality and determine if query refinement is needed
 *
 * Architecture: LLM-powered result evaluation with structured output
 * Determines: Whether results are sufficient or if a refined query should be created
 */

import * as vscode from 'vscode';
import { z } from 'zod';
import { Logger } from '../utils/logger';
import { RetrievalResult } from './ragAgent';
import { QueryPlan } from './queryPlannerAgent';

// Zod schema for evaluation result
const EvaluationResultSchema = z.object({
  isSufficient: z.boolean().describe('Whether the current results are sufficient to answer the query'),
  confidence: z.number().min(0).max(1).describe('Overall confidence in the results (0-1)'),
  shouldRetry: z.boolean().describe('Whether a new query should be created and retried'),
  reasoning: z.string().describe('Explanation of the evaluation'),
  gaps: z.array(z.string()).optional().describe('Identified gaps or missing information in the results'),
  suggestedQuery: z.string().optional().describe('Suggested refined query if retry is needed'),
  improvementStrategy: z.enum(['refine', 'expand', 'narrow', 'none']).optional().describe('Strategy for improving results'),
});

export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;

export interface ResultEvaluatorOptions {
  /** Original query */
  originalQuery: string;

  /** Query plan used */
  queryPlan: QueryPlan;

  /** Confidence threshold (0-1) */
  confidenceThreshold?: number;

  /** Minimum number of results expected */
  minResults?: number;

  /** Enable LLM-based evaluation (requires LM API access) */
  useLLM?: boolean;

  /** Topic name for context */
  topicName?: string;

  /** Workspace context */
  workspaceContext?: string;

  /** Current iteration number */
  currentIteration?: number;

  /** Maximum iterations allowed */
  maxIterations?: number;
}

/**
 * Result Evaluator Agent using Zod for validation
 */
export class ResultEvaluatorAgent {
  private logger: Logger;

  constructor() {
    this.logger = new Logger('ResultEvaluatorAgent');
    this.logger.info('ResultEvaluatorAgent initialized');
  }

  /**
   * Evaluate retrieval results and decide if retry is needed
   */
  public async evaluate(
    results: RetrievalResult[],
    options: ResultEvaluatorOptions
  ): Promise<EvaluationResult> {
    this.logger.info('Evaluating retrieval results', {
      resultCount: results.length,
      originalQuery: options.originalQuery.substring(0, 100),
      currentIteration: options.currentIteration,
    });

    try {
      // Use LLM-based evaluation if enabled and available
      if (options.useLLM !== false) {
        const llmEvaluation = await this.evaluateWithLLM(results, options);
        if (llmEvaluation) {
          return llmEvaluation;
        }
      }

      // Fallback to heuristic-based evaluation
      this.logger.debug('Using heuristic-based evaluation (LLM not available)');
      return this.evaluateWithHeuristics(results, options);
    } catch (error) {
      this.logger.error('Failed to evaluate results', {
        error: error instanceof Error ? error.message : String(error),
      });

      // Fallback to heuristic evaluation
      return this.evaluateWithHeuristics(results, options);
    }
  }

  /**
   * Build the prompt for LLM evaluation
   */
  private buildPrompt(
    results: RetrievalResult[],
    options: ResultEvaluatorOptions
  ): string {
    const context = this.buildContextString(options);
    const resultsSummary = this.summarizeResults(results);

    return `You are a result evaluation assistant for a RAG (Retrieval-Augmented Generation) system.
Your task is to evaluate retrieval results and determine if they are sufficient to answer the user's query.

${context}

Original Query: "${options.originalQuery}"

Query Plan:
- Complexity: ${options.queryPlan.complexity}
- Strategy: ${options.queryPlan.strategy}
- Sub-queries executed: ${options.queryPlan.subQueries.length}
- Current iteration: ${options.currentIteration || 1} / ${options.maxIterations || 3}

Retrieval Results Summary:
${resultsSummary}

Evaluation Criteria:
1. Relevance: Do the results directly address the query?
2. Completeness: Are there gaps in the information?
3. Confidence: Are the similarity scores high enough?
4. Coverage: Do the results cover all aspects of the query?

Guidelines:
- If results are highly relevant (avg similarity > 0.8) and cover the query well, mark isSufficient=true
- If results are partially relevant but missing key information, suggest shouldRetry=true with a refined query
- If results are irrelevant or very low confidence, suggest shouldRetry=true with a completely different approach
- Consider the iteration count - don't retry indefinitely if we're near max iterations

Response Format: Provide a JSON object with this exact structure:
{
  "isSufficient": true | false,
  "confidence": 0.0-1.0,
  "shouldRetry": true | false,
  "reasoning": "explanation of your evaluation",
  "gaps": ["gap1", "gap2"] (optional),
  "suggestedQuery": "refined query if retry needed" (optional),
  "improvementStrategy": "refine" | "expand" | "narrow" | "none" (optional)
}

Provide your evaluation as valid JSON:`;
  }

  /**
   * Summarize results for LLM prompt
   */
  private summarizeResults(results: RetrievalResult[]): string {
    if (results.length === 0) {
      return 'No results retrieved.';
    }

    const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
    const maxScore = Math.max(...results.map((r) => r.score));
    const minScore = Math.min(...results.map((r) => r.score));

    const uniqueSources = new Set(results.map((r) => r.source));
    const uniqueDocs = new Set(
      results.map((r) => r.document.metadata.source || 'Unknown')
    );

    const topResults = results
      .slice(0, 3)
      .map((r, idx) => {
        const preview = r.document.pageContent.substring(0, 150).replace(/\n/g, ' ');
        return `  ${idx + 1}. Score: ${r.score.toFixed(3)}, Source: ${r.source}, Preview: "${preview}..."`;
      })
      .join('\n');

    return `Total Results: ${results.length}
Average Similarity Score: ${avgScore.toFixed(3)}
Score Range: ${minScore.toFixed(3)} - ${maxScore.toFixed(3)}
Unique Sources: ${Array.from(uniqueSources).join(', ')}
Unique Documents: ${uniqueDocs.size}

Top Results:
${topResults}`;
  }

  /**
   * Evaluate results using VS Code Language Model API
   */
  private async evaluateWithLLM(
    results: RetrievalResult[],
    options: ResultEvaluatorOptions
  ): Promise<EvaluationResult | null> {
    try {
      // Get VS Code Language Model
      const models = await vscode.lm.selectChatModels({ vendor: 'copilot', family: 'gpt-4o' });
      if (models.length === 0) {
        this.logger.debug('No language models available');
        return null;
      }

      const model = models[0];

      // Build prompt
      const prompt = this.buildPrompt(results, options);

      // Send request to LLM
      const messages = [
        { role: 1, content: prompt } as any, // UserMessage role
      ];

      const response = await model.sendRequest(
        messages,
        {},
        new vscode.CancellationTokenSource().token
      );

      // Collect response
      let responseText = '';
      for await (const chunk of response.text) {
        responseText += chunk;
      }

      this.logger.debug('LLM evaluation response received', {
        responseLength: responseText.length,
      });

      // Parse JSON response
      // Extract JSON from markdown code blocks if present
      const jsonMatch =
        responseText.match(/```json\n([\s\S]*?)\n```/) ||
        responseText.match(/```\n([\s\S]*?)\n```/);
      const jsonText = jsonMatch ? jsonMatch[1] : responseText;

      const parsedJSON = JSON.parse(jsonText.trim());
      const evaluation = EvaluationResultSchema.parse(parsedJSON) as EvaluationResult;

      this.logger.info('LLM evaluation completed', {
        isSufficient: evaluation.isSufficient,
        confidence: evaluation.confidence,
        shouldRetry: evaluation.shouldRetry,
      });

      return evaluation;
    } catch (error) {
      this.logger.warn('LLM evaluation failed, will use fallback', {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  /**
   * Evaluate results using heuristic rules (no LLM required)
   */
  private evaluateWithHeuristics(
    results: RetrievalResult[],
    options: ResultEvaluatorOptions
  ): EvaluationResult {
    this.logger.debug('Creating heuristic evaluation');

    const confidenceThreshold = options.confidenceThreshold ?? 0.7;
    const minResults = options.minResults ?? 3;
    const currentIteration = options.currentIteration ?? 1;
    const maxIterations = options.maxIterations ?? 3;

    // Calculate metrics
    const avgConfidence = this.calculateAvgConfidence(results);
    const maxConfidence = results.length > 0 ? Math.max(...results.map((r) => r.score)) : 0;
    const hasEnoughResults = results.length >= minResults;
    const isAtMaxIterations = currentIteration >= maxIterations;

    // Determine if sufficient
    const isSufficient =
      avgConfidence >= confidenceThreshold &&
      hasEnoughResults &&
      maxConfidence >= confidenceThreshold * 0.9;

    // Determine if should retry
    let shouldRetry = false;
    let improvementStrategy: 'refine' | 'expand' | 'narrow' | 'none' = 'none';
    let suggestedQuery: string | undefined;
    const gaps: string[] = [];

    if (!isSufficient && !isAtMaxIterations) {
      shouldRetry = true;

      if (results.length === 0) {
        // No results - try broader query
        gaps.push('No results retrieved');
        improvementStrategy = 'expand';
        suggestedQuery = this.createExpandedQuery(options.originalQuery);
      } else if (avgConfidence < confidenceThreshold * 0.6) {
        // Very low confidence - refine query
        gaps.push('Low relevance scores');
        improvementStrategy = 'refine';
        suggestedQuery = this.createRefinedQuery(options.originalQuery, results);
      } else if (!hasEnoughResults) {
        // Not enough results - expand query
        gaps.push(`Only ${results.length} results, need at least ${minResults}`);
        improvementStrategy = 'expand';
        suggestedQuery = this.createExpandedQuery(options.originalQuery);
      } else if (maxConfidence < confidenceThreshold) {
        // Best result still below threshold - narrow query
        gaps.push('Top result confidence below threshold');
        improvementStrategy = 'narrow';
        suggestedQuery = this.createNarrowedQuery(options.originalQuery);
      }
    }

    const reasoning = this.buildReasoning(
      isSufficient,
      shouldRetry,
      avgConfidence,
      results.length,
      confidenceThreshold,
      gaps,
      improvementStrategy
    );

    const evaluation: EvaluationResult = {
      isSufficient,
      confidence: avgConfidence,
      shouldRetry,
      reasoning,
      gaps: gaps.length > 0 ? gaps : undefined,
      suggestedQuery,
      improvementStrategy: improvementStrategy !== 'none' ? improvementStrategy : undefined,
    };

    this.logger.info('Heuristic evaluation completed', {
      isSufficient: evaluation.isSufficient,
      confidence: evaluation.confidence,
      shouldRetry: evaluation.shouldRetry,
    });

    return evaluation;
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
   * Create an expanded query (broader search)
   */
  private createExpandedQuery(originalQuery: string): string {
    // Add broader context or remove specific constraints
    // Simple heuristic: add related terms or make it more general
    return originalQuery; // For now, return original - could be enhanced with LLM
  }

  /**
   * Create a refined query based on results
   */
  private createRefinedQuery(originalQuery: string, results: RetrievalResult[]): string {
    // Analyze results to identify what worked and refine
    // Simple heuristic: return original for now - could be enhanced with LLM
    return originalQuery;
  }

  /**
   * Create a narrowed query (more specific)
   */
  private createNarrowedQuery(originalQuery: string): string {
    // Make query more specific
    // Simple heuristic: return original for now - could be enhanced with LLM
    return originalQuery;
  }

  /**
   * Build reasoning string for evaluation
   */
  private buildReasoning(
    isSufficient: boolean,
    shouldRetry: boolean,
    avgConfidence: number,
    resultCount: number,
    threshold: number,
    gaps: string[],
    strategy: string
  ): string {
    const parts: string[] = [];

    if (isSufficient) {
      parts.push(
        `Results are sufficient: average confidence ${avgConfidence.toFixed(3)} meets threshold ${threshold.toFixed(3)}, ${resultCount} results retrieved.`
      );
    } else {
      parts.push(
        `Results are insufficient: average confidence ${avgConfidence.toFixed(3)} below threshold ${threshold.toFixed(3)}, ${resultCount} results retrieved.`
      );

      if (gaps.length > 0) {
        parts.push(`Identified gaps: ${gaps.join('; ')}.`);
      }

      if (shouldRetry) {
        parts.push(`Recommend retry with ${strategy} strategy.`);
      } else {
        parts.push('Max iterations reached, stopping retry.');
      }
    }

    return parts.join(' ');
  }

  /**
   * Build context string from options
   */
  private buildContextString(options: ResultEvaluatorOptions): string {
    const parts: string[] = [];

    if (options.topicName) {
      parts.push(`Topic: ${options.topicName}`);
    }

    if (options.workspaceContext) {
      parts.push(`Workspace Context: ${options.workspaceContext}`);
    }

    if (parts.length === 0) {
      return 'No additional context provided.';
    }

    return parts.join('\n');
  }

  /**
   * Validate an evaluation result
   */
  public validateEvaluation(evaluation: EvaluationResult): boolean {
    if (evaluation.confidence < 0 || evaluation.confidence > 1) {
      this.logger.warn('Invalid evaluation: confidence out of range');
      return false;
    }

    if (evaluation.shouldRetry && !evaluation.suggestedQuery) {
      this.logger.warn('Invalid evaluation: shouldRetry=true but no suggestedQuery');
      // This is not necessarily invalid, but could be improved
    }

    return true;
  }
}
