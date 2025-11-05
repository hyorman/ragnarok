/**
 * Shared types and abstract base for query planners
 */

import { SearchResult } from './types';

export interface SubQuery {
  query: string;
  reasoning: string;
  topK?: number;
  dependencies?: number[]; // Indices of other sub-queries this depends on
}

export interface QueryPlan {
  originalQuery: string;
  subQueries: SubQuery[];
  strategy: 'sequential' | 'parallel';
  complexity: 'simple' | 'moderate' | 'complex';
}

export interface FollowUpQuery {
  query: string;
  reasoning: string;
}

export abstract class IQueryPlanner {
  /**
   * Create a plan from a user query.
   *
   * The optional `context` parameter can be used by implementations (for
   * example the LLM-based planner) to pass extra topic/workspace metadata.
   * The optional `baseTopK` parameter allows planners to scale sub-query topK
   * values based on user preference (e.g., if user wants 10 results, sub-queries
   * should use higher topK than the default 5).
   */
  public abstract createPlan(
    query: string,
    baseTopK: number,
    context?: Record<string, any>,
    workspaceContext?: any
  ): Promise<QueryPlan>;

  /**
   * Optionally generate follow-up queries given existing results and gaps.
   */
  public abstract generateFollowUpQuery(
    originalQuery: string,
    existingResults: SearchResult[],
    gaps: string[]
  ): Promise<FollowUpQuery | null>;

  /**
   * Fallback plan for when planning fails - returns single query
   */
  public fallbackSingleQueryPlan(query: string, baseTopK: number): QueryPlan {
    return {
      originalQuery: query,
      subQueries: [{
        query,
        reasoning: 'Fallback single query',
        topK: baseTopK,
      }],
      strategy: 'parallel',
      complexity: 'simple',
    };
  }

  /**
   * Default fallback follow-up query generator used when a planner can't
   * produce a follow-up via more advanced means (LLM, etc.).
   */
  public fallbackFollowUpQuery(originalQuery: string, gaps: string[]): FollowUpQuery | null {
    if (!gaps || gaps.length === 0) return null;
    const mainGap = gaps[0];

    // Use simple heuristics to pick a main concept from the original query.
    const queryWords = originalQuery.split(/\s+/).filter((w) => w.length > 3);
    const mainConcept = queryWords[0] || 'information';

    return {
      query: `${mainConcept} ${mainGap}`,
      reasoning: `Addressing gap: ${mainGap}`,
    };
  }

  /**
   * Determine execution strategy based on sub-query dependencies.
   */
  protected determineStrategy(subQueries: SubQuery[]): 'sequential' | 'parallel' {
    const hasDependencies = subQueries.some((sq) => sq.dependencies && sq.dependencies.length > 0);
    return hasDependencies ? 'sequential' : 'parallel';
  }
}
