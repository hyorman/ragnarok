/**
 * Shared evaluator types and interface
 */

import { SearchResult } from './types';

export interface EvaluationResult {
  confidence: number; // 0-1 score
  isComplete: boolean; // Whether we have sufficient information
  gaps: string[]; // Identified information gaps
  reasoning: string;
}

export interface IResultEvaluator {
  /**
   * Evaluate search results quality and completeness.
   * Implementations may use `context` to provide extra information (topic name, previous steps, etc.)
   */
  evaluate(
    query: string,
    results: SearchResult[],
    confidenceThreshold: number,
    context?: { topicName?: string; previousSteps?: string[] }
  ): Promise<EvaluationResult>;
}
