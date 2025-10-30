/**
 * Query Planner for Agentic RAG
 * Decomposes complex queries into sub-queries and plans retrieval strategy
 */

import { SearchResult } from './types';
import {
  SubQuery,
  QueryPlan,
  FollowUpQuery,
  IQueryPlanner,
} from './queryPlannerBase';

export class QueryPlanner extends IQueryPlanner {
  /**
   * Analyze query and create execution plan
   */
  public async createPlan(
    query: string,
    context?: Record<string, any>,
    workspaceContext?: any
  ): Promise<QueryPlan> {

    // Analyze query complexity
    const complexity = this.analyzeComplexity(query);
    // Decompose based on complexity
    const subQueries = await this.decompose(query, complexity);

    return {
      originalQuery: query,
      subQueries,
      strategy: this.determineStrategy(subQueries),
      complexity,
    };
  }

  /**
   * Analyze query complexity
   */
  private analyzeComplexity(query: string): 'simple' | 'moderate' | 'complex' {
    const lowerQuery = query.toLowerCase();

    // Complex indicators: multiple questions, comparisons, temporal references, causal relationships
    const complexIndicators = [
      /\b(compare|difference|versus|vs|contrast)\b/,
      /\b(before|after|during|since|until)\b/,
      /\b(why|how|explain|describe)\b.*\b(and|also|additionally)\b/,
      /\b(first.*then|step.*step)\b/,
      /\b(impact|effect|result|consequence)\b.*\b(of|from)\b/,
      /\?.*\?/, // Multiple question marks
    ];

    // Moderate indicators: conjunctions, multiple concepts
    const moderateIndicators = [
      /\b(and|or|but|also|additionally|furthermore)\b/,
      /\b(what|when|where|who|which)\b/,
    ];

    const complexCount = complexIndicators.filter((regex) => regex.test(lowerQuery)).length;
    const moderateCount = moderateIndicators.filter((regex) => regex.test(lowerQuery)).length;

    if (complexCount >= 2) {
      return 'complex';
    } else if (complexCount >= 1 || moderateCount >= 2) {
      return 'moderate';
    }
    return 'simple';
  }

  /**
   * Decompose query into sub-queries
   */
  private async decompose(query: string, complexity: 'simple' | 'moderate' | 'complex'): Promise<SubQuery[]> {
    if (complexity === 'simple') {
      return [
        {
          query,
          reasoning: 'Simple query requiring single retrieval',
          topK: 5,
        },
      ];
    }

    const subQueries: SubQuery[] = [];
    const lowerQuery = query.toLowerCase();

    // Handle comparison queries
    if (/\b(compare|difference|versus|vs|contrast)\b/.test(lowerQuery)) {
      const entities = this.extractComparisonEntities(query);
      if (entities.length >= 2) {
        subQueries.push({
          query: `Information about ${entities[0]}`,
          reasoning: `First entity in comparison: ${entities[0]}`,
          topK: 5,
        });
        subQueries.push({
          query: `Information about ${entities[1]}`,
          reasoning: `Second entity in comparison: ${entities[1]}`,
          topK: 5,
          dependencies: [0],
        });
        subQueries.push({
          query: `${entities[0]} versus ${entities[1]} differences similarities`,
          reasoning: 'Direct comparison information',
          topK: 3,
          dependencies: [0, 1],
        });
        return subQueries;
      }
    }

    // Handle temporal/causal queries
    if (/\b(before|after|impact|effect|result|consequence)\b/.test(lowerQuery)) {
      // Extract main concepts
      const concepts = this.extractKeyPhrases(query);
      if (concepts.length > 0) {
        subQueries.push({
          query: concepts[0],
          reasoning: 'Primary concept or event',
          topK: 5,
        });

        // Look for related/consequent information
        if (/\b(after|impact|effect|result|consequence)\b/.test(lowerQuery)) {
          subQueries.push({
            query: `${concepts[0]} impact effect result`,
            reasoning: 'Consequences or effects',
            topK: 5,
            dependencies: [0],
          });
        }

        if (/\b(before|cause|reason)\b/.test(lowerQuery)) {
          subQueries.push({
            query: `${concepts[0]} cause reason background`,
            reasoning: 'Causes or background',
            topK: 5,
            dependencies: [0],
          });
        }

        return subQueries;
      }
    }

    // Handle "how" or "why" questions (explanatory)
    if (/\b(how|why|explain)\b/.test(lowerQuery)) {
      const mainTopic = this.extractMainTopic(query);
      subQueries.push({
        query: mainTopic,
        reasoning: 'Core concept definition and overview',
        topK: 5,
      });
      subQueries.push({
        query: `${mainTopic} examples use cases`,
        reasoning: 'Practical examples and applications',
        topK: 3,
        dependencies: [0],
      });
      return subQueries;
    }

    // Handle conjunctive queries (multiple aspects)
    if (/\b(and|also|additionally)\b/.test(lowerQuery)) {
      const parts = query.split(/\b(and|also|additionally)\b/i).filter((p) => p.trim().length > 3);
      parts.slice(0, 3).forEach((part, idx) => {
        // Limit to 3 sub-queries
        subQueries.push({
          query: part.trim(),
          reasoning: `Sub-question ${idx + 1}`,
          topK: 5,
        });
      });
      return subQueries;
    }

    // Default: treat as single query
    return [
      {
        query,
        reasoning: 'Direct query execution',
        topK: 5,
      },
    ];
  }


  /**
   * Extract entities being compared
   */
  private extractComparisonEntities(query: string): string[] {
    // Simple heuristic: look for patterns like "X vs Y" or "compare X and Y"
    const vsPattern = /([^vs]+)\s+(?:vs|versus)\s+([^?.!]+)/i;
    const vsMatch = query.match(vsPattern);
    if (vsMatch) {
      return [vsMatch[1].trim(), vsMatch[2].trim()];
    }

    const comparePattern = /compare\s+([^and]+)\s+and\s+([^?.!]+)/i;
    const compareMatch = query.match(comparePattern);
    if (compareMatch) {
      return [compareMatch[1].trim(), compareMatch[2].trim()];
    }

    const betweenPattern = /(?:difference|comparison)\s+between\s+([^and]+)\s+and\s+([^?.!]+)/i;
    const betweenMatch = query.match(betweenPattern);
    if (betweenMatch) {
      return [betweenMatch[1].trim(), betweenMatch[2].trim()];
    }

    return [];
  }

  /**
   * Extract key phrases from query
   */
  private extractKeyPhrases(query: string): string[] {
    // Remove question words and focus on content
    const cleaned = query
      .replace(/\b(what|when|where|who|why|how|is|are|was|were|do|does|did|can|could|would|should)\b/gi, '')
      .replace(/[?!.]/g, '')
      .trim();

    // Split on conjunctions and take first meaningful phrase
    const parts = cleaned.split(/\b(and|or|but)\b/i);
    return parts.filter((p) => p.trim().length > 5).map((p) => p.trim());
  }

  /**
   * Extract main topic from query
   */
  private extractMainTopic(query: string): string {
    // Remove question words
    const cleaned = query
      .replace(/\b(what|when|where|who|why|how|is|are|was|were|do|does|did|can|could|would|should)\b/gi, '')
      .replace(/[?!.]/g, '')
      .trim();

    // Take first significant phrase (up to 50 chars)
    return cleaned.substring(0, 50).trim();
  }

  /**
   * Generate follow-up query based on information gaps
   */
  public async generateFollowUpQuery(
    originalQuery: string,
    existingResults: SearchResult[],
    gaps: string[]
  ): Promise<FollowUpQuery | null> {
    if (gaps.length === 0) {
      return null;
    }

    // Create follow-up query targeting the first gap
    const mainGap = gaps[0];

    // Extract key terms from original query
    const keyTerms = this.extractKeyPhrases(originalQuery);

    if (keyTerms.length === 0) {
      return null;
    }

    return {
      query: `${keyTerms[0]} ${mainGap}`,
      reasoning: `Addressing identified gap: ${mainGap}`,
    };
  }
}

