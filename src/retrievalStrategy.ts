/**
 * Retrieval Strategies for Agentic RAG
 * Different approaches to searching the vector database
 */

import { VectorDatabaseService } from './vectorDatabase';
import { EmbeddingService } from './embeddingService';
import { SearchResult } from './types';

/**
 * Abstract base class for retrieval strategies
 */
export abstract class RetrievalStrategy {
  constructor(
    protected vectorDb: VectorDatabaseService,
    protected embeddingService: EmbeddingService
  ) {}

  abstract search(topicId: string, query: string, topK: number): Promise<SearchResult[]>;
}

/**
 * Vector Search Strategy - pure semantic similarity search
 */
export class VectorSearchStrategy extends RetrievalStrategy {
  public async search(topicId: string, query: string, topK: number): Promise<SearchResult[]> {
    // Generate embedding for query
    const queryEmbedding = await this.embeddingService.embed(query);

    // Perform vector similarity search
    return this.vectorDb.search(topicId, queryEmbedding, topK);
  }
}

/**
 * Hybrid Search Strategy - combines semantic and keyword-based search
 */
export class HybridSearchStrategy extends RetrievalStrategy {
  private readonly KEYWORD_WEIGHT = 0.3;
  private readonly VECTOR_WEIGHT = 0.7;

  public async search(topicId: string, query: string, topK: number): Promise<SearchResult[]> {
    // 1. Get vector search results (semantic)
    const queryEmbedding = await this.embeddingService.embed(query);
    const vectorResults = await this.vectorDb.search(topicId, queryEmbedding, topK * 2);

    // 2. Extract keywords from query
    const keywords = this.extractKeywords(query);

    // 3. Calculate keyword matching scores
    const keywordScores = new Map<string, number>();
    for (const result of vectorResults) {
      const keywordScore = this.calculateKeywordScore(result.chunk.text, keywords);
      keywordScores.set(result.chunk.id, keywordScore);
    }

    // 4. Combine scores using weighted average
    const hybridResults = vectorResults.map((result) => {
      const vectorScore = result.similarity;
      const keywordScore = keywordScores.get(result.chunk.id) || 0;

      const hybridScore =
        this.VECTOR_WEIGHT * vectorScore + this.KEYWORD_WEIGHT * keywordScore;

      return {
        ...result,
        similarity: hybridScore,
      };
    });

    // 5. Re-sort by hybrid score and return top-k
    hybridResults.sort((a, b) => b.similarity - a.similarity);
    return hybridResults.slice(0, topK);
  }

  /**
   * Extract keywords from query
   */
  private extractKeywords(query: string): string[] {
    // Stop words to filter out
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
      'as',
      'into',
      'through',
      'during',
      'before',
      'after',
      'above',
      'below',
      'between',
      'under',
      'again',
      'further',
      'then',
      'once',
    ]);

    // Extract words, filter stop words, and normalize
    const words = query
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .split(/\s+/)
      .filter((word) => word.length > 2 && !stopWords.has(word));

    return Array.from(new Set(words)); // Return unique keywords
  }

  /**
   * Calculate keyword matching score
   */
  private calculateKeywordScore(text: string, keywords: string[]): number {
    if (keywords.length === 0) {
      return 0;
    }

    const textLower = text.toLowerCase();
    let matchCount = 0;
    let totalWeight = 0;

    for (const keyword of keywords) {
      // Count occurrences of keyword
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      const matches = textLower.match(regex);

      if (matches) {
        // Weight longer keywords more heavily
        const weight = Math.log(keyword.length + 1);
        matchCount += matches.length * weight;
        totalWeight += weight;
      }
    }

    // Normalize score to 0-1 range
    if (totalWeight === 0) {
      return 0;
    }

    // Use log to dampen very high match counts
    const normalizedScore = Math.log(matchCount + 1) / Math.log(keywords.length * 10 + 1);
    return Math.min(1, normalizedScore);
  }
}

/**
 * MMR (Maximal Marginal Relevance) Strategy - balances relevance and diversity
 */
export class MMRSearchStrategy extends RetrievalStrategy {
  private readonly LAMBDA = 0.5; // Balance between relevance and diversity

  public async search(topicId: string, query: string, topK: number): Promise<SearchResult[]> {
    // Get initial candidates (more than needed)
    const queryEmbedding = await this.embeddingService.embed(query);
    const candidates = await this.vectorDb.search(topicId, queryEmbedding, topK * 3);

    if (candidates.length === 0) {
      return [];
    }

    // Apply MMR selection
    const selected: SearchResult[] = [];
    const remaining = [...candidates];

    // Always select the top result first
    selected.push(remaining.shift()!);

    // Iteratively select diverse results
    while (selected.length < topK && remaining.length > 0) {
      let bestIndex = 0;
      let bestScore = -Infinity;

      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i];

        // Calculate relevance score (similarity to query)
        const relevanceScore = candidate.similarity;

        // Calculate max similarity to already selected results
        let maxSimilarity = 0;
        for (const selectedResult of selected) {
          const similarity = this.embeddingService.cosineSimilarity(
            candidate.chunk.embedding,
            selectedResult.chunk.embedding
          );
          maxSimilarity = Math.max(maxSimilarity, similarity);
        }

        // MMR score: balance relevance and diversity
        const mmrScore = this.LAMBDA * relevanceScore - (1 - this.LAMBDA) * maxSimilarity;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIndex = i;
        }
      }

      // Add best result and remove from candidates
      selected.push(remaining.splice(bestIndex, 1)[0]);
    }

    return selected;
  }
}

