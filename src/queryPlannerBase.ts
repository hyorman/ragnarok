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
   */
  public abstract createPlan(
    query: string,
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
   * Public helper that returns a conservative single-query plan used as a
   * centralized fallback. Exposed publicly so orchestrator can call it when
   * planner execution fails.
   */
  public fallbackSingleQueryPlan(originalQuery: string): QueryPlan {
    const lowerQuery = originalQuery.toLowerCase();

    let complexity: 'simple' | 'moderate' | 'complex' = 'simple';
    if (/\b(compare|versus|difference)\b/.test(lowerQuery)) {
      complexity = 'complex';
    } else if (/\b(and|also|how|why)\b/.test(lowerQuery)) {
      complexity = 'moderate';
    }

    return {
      originalQuery,
      subQueries: [{ query: originalQuery, reasoning: 'Direct query execution', topK: 5 }],
      strategy: 'sequential',
      complexity,
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
