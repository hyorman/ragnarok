/**
 * Result Evaluator for Agentic RAG
 * Evaluates search result quality and determines if more retrieval is needed
 */

import { SearchResult } from './types';

export interface EvaluationResult {
  confidence: number; // 0-1 score
  isComplete: boolean; // Whether we have sufficient information
  gaps: string[]; // Identified information gaps
  reasoning: string;
}

export class ResultEvaluator {
  /**
   * Evaluate search results quality and completeness
   */
  public async evaluate(
    query: string,
    results: SearchResult[],
    confidenceThreshold: number
  ): Promise<EvaluationResult> {
    if (results.length === 0) {
      return {
        confidence: 0,
        isComplete: false,
        gaps: ['No results found'],
        reasoning: 'No search results returned',
      };
    }

    // Calculate various quality metrics
    const similarityScore = this.calculateSimilarityScore(results);
    const diversityScore = this.calculateDiversityScore(results);
    const coverageScore = this.calculateCoverageScore(query, results);

    // Weighted combination of scores
    const confidence = similarityScore * 0.5 + diversityScore * 0.2 + coverageScore * 0.3;

    // Identify gaps
    const gaps = this.identifyGaps(query, results);

    // Determine if complete
    const isComplete = confidence >= confidenceThreshold && gaps.length === 0;

    return {
      confidence: Math.round(confidence * 100) / 100,
      isComplete,
      gaps,
      reasoning: this.generateReasoning(confidence, similarityScore, diversityScore, coverageScore, gaps),
    };
  }

  /**
   * Calculate average similarity score of top results
   */
  private calculateSimilarityScore(results: SearchResult[]): number {
    if (results.length === 0) {
      return 0;
    }

    // Focus on top 5 results
    const topResults = results.slice(0, 5);
    const avgSimilarity = topResults.reduce((sum, r) => sum + r.similarity, 0) / topResults.length;

    // Normalize to 0-1 (assuming similarities are already 0-1 from cosine similarity)
    return Math.min(1, avgSimilarity);
  }

  /**
   * Calculate diversity score - how diverse are the results
   */
  private calculateDiversityScore(results: SearchResult[]): number {
    if (results.length <= 1) {
      return 0.5; // Neutral score for single result
    }

    // Check document diversity
    const uniqueDocuments = new Set(results.map((r) => r.documentName)).size;
    const documentDiversity = uniqueDocuments / Math.min(results.length, 5);

    // Check section diversity (if available)
    const uniqueSections = new Set(
      results.map((r) => r.chunk.metadata.sectionTitle || 'unknown')
    ).size;
    const sectionDiversity = uniqueSections / Math.min(results.length, 5);

    // Average of diversities
    return (documentDiversity + sectionDiversity) / 2;
  }

  /**
   * Calculate coverage score - do results cover key aspects of query
   */
  private calculateCoverageScore(query: string, results: SearchResult[]): number {
    // Extract key terms from query
    const keyTerms = this.extractKeyTerms(query);

    if (keyTerms.length === 0) {
      return 0.5; // Neutral score
    }

    // Count how many key terms appear in results
    let coveredTerms = 0;

    for (const term of keyTerms) {
      const termLower = term.toLowerCase();
      const foundInResults = results.some((r) => r.chunk.text.toLowerCase().includes(termLower));
      if (foundInResults) {
        coveredTerms++;
      }
    }

    return coveredTerms / keyTerms.length;
  }

  /**
   * Extract key terms from query
   */
  private extractKeyTerms(query: string): string[] {
    // Remove common stop words and extract meaningful terms
    const stopWords = new Set([
      'what',
      'when',
      'where',
      'who',
      'why',
      'how',
      'is',
      'are',
      'was',
      'were',
      'do',
      'does',
      'did',
      'the',
      'a',
      'an',
      'and',
      'or',
      'but',
      'in',
      'on',
      'at',
      'to',
      'for',
      'of',
      'with',
      'by',
      'from',
      'about',
    ]);

    const words = query
      .toLowerCase()
      .replace(/[?!.,;:]/g, '')
      .split(/\s+/)
      .filter((word) => word.length > 3 && !stopWords.has(word));

    // Return unique terms
    return Array.from(new Set(words));
  }

  /**
   * Identify information gaps
   */
  private identifyGaps(query: string, results: SearchResult[]): string[] {
    const gaps: string[] = [];
    const keyTerms = this.extractKeyTerms(query);

    // Check if query asks for specific types of information
    const queryLower = query.toLowerCase();

    // Gap detection patterns
    const gapPatterns = [
      {
        pattern: /\b(example|examples|instance|instances)\b/,
        gap: 'specific examples',
        check: (results: SearchResult[]) =>
          !results.some((r) => /\b(example|instance|such as|for example|e\.g\.)\b/i.test(r.chunk.text)),
      },
      {
        pattern: /\b(how|process|step|steps|procedure)\b/,
        gap: 'step-by-step process',
        check: (results: SearchResult[]) =>
          !results.some((r) => /\b(first|second|third|step|then|next|finally)\b/i.test(r.chunk.text)),
      },
      {
        pattern: /\b(why|reason|cause|because)\b/,
        gap: 'explanatory reasoning',
        check: (results: SearchResult[]) =>
          !results.some((r) => /\b(because|reason|cause|due to|therefore)\b/i.test(r.chunk.text)),
      },
      {
        pattern: /\b(benefit|advantage|pro|cons|disadvantage)\b/,
        gap: 'benefits or drawbacks',
        check: (results: SearchResult[]) =>
          !results.some(
            (r) =>
              /\b(benefit|advantage|pro|cons|disadvantage|drawback|limitation)\b/i.test(r.chunk.text)
          ),
      },
      {
        pattern: /\b(compare|difference|versus|vs|contrast)\b/,
        gap: 'comparative information',
        check: (results: SearchResult[]) =>
          !results.some(
            (r) =>
              /\b(compare|difference|versus|vs|contrast|similar|different|unlike)\b/i.test(r.chunk.text)
          ),
      },
    ];

    for (const { pattern, gap, check } of gapPatterns) {
      if (pattern.test(queryLower) && check(results)) {
        gaps.push(gap);
      }
    }

    // Check for missing key terms
    const missingTerms = keyTerms.filter(
      (term) => !results.some((r) => r.chunk.text.toLowerCase().includes(term.toLowerCase()))
    );

    if (missingTerms.length > keyTerms.length / 2) {
      gaps.push(`information about: ${missingTerms.slice(0, 3).join(', ')}`);
    }

    return gaps;
  }

  /**
   * Generate reasoning explanation
   */
  private generateReasoning(
    confidence: number,
    similarityScore: number,
    diversityScore: number,
    coverageScore: number,
    gaps: string[]
  ): string {
    const parts: string[] = [];

    if (similarityScore >= 0.7) {
      parts.push('High relevance match');
    } else if (similarityScore >= 0.5) {
      parts.push('Moderate relevance match');
    } else {
      parts.push('Low relevance match');
    }

    if (diversityScore >= 0.6) {
      parts.push('diverse sources');
    } else if (diversityScore >= 0.4) {
      parts.push('moderate source diversity');
    } else {
      parts.push('limited source diversity');
    }

    if (coverageScore >= 0.7) {
      parts.push('good query coverage');
    } else if (coverageScore >= 0.5) {
      parts.push('partial query coverage');
    } else {
      parts.push('limited query coverage');
    }

    if (gaps.length > 0) {
      parts.push(`gaps: ${gaps.join(', ')}`);
    }

    return `${parts.join('; ')}. Overall confidence: ${Math.round(confidence * 100)}%`;
  }
}

