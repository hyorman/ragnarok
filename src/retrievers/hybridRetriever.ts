/**
 * Hybrid Retriever - Combines semantic (vector) and keyword (BM25-like) search
 * Implements weighted scoring and result fusion for optimal retrieval
 *
 * Architecture: Multi-strategy retrieval with configurable weights
 * Replaces: HybridSearchStrategy with LangChain integration
 */

import { VectorStore } from '@langchain/core/vectorstores';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { Logger } from '../utils/logger';

export interface HybridSearchOptions {
  /** Number of results to return */
  k?: number;

  /** Weight for vector similarity (0-1) */
  vectorWeight?: number;

  /** Weight for keyword matching (0-1) */
  keywordWeight?: number;

  /** Minimum similarity threshold (0-1) */
  minSimilarity?: number;

  /** Enable keyword boosting */
  keywordBoosting?: boolean;

  /** Custom stop words to filter */
  customStopWords?: string[];
}

export interface HybridSearchResult {
  document: LangChainDocument;
  score: number;
  vectorScore: number;
  keywordScore: number;
  explanation?: string;
}

/**
 * Hybrid retriever combining vector and keyword search
 */
export class HybridRetriever {
  private logger: Logger;
  private vectorStore: VectorStore;

  // Default weights
  private readonly DEFAULT_VECTOR_WEIGHT = 0.7;
  private readonly DEFAULT_KEYWORD_WEIGHT = 0.3;
  private readonly DEFAULT_K = 5;
  private readonly DEFAULT_MIN_SIMILARITY = 0.0;

  // Stop words for keyword extraction (common English words)
  private readonly STOP_WORDS = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'how',
    'this', 'these', 'those', 'they', 'their', 'there', 'which', 'can',
    'could', 'would', 'should', 'do', 'does', 'did', 'have', 'had', 'been',
  ]);

  constructor(vectorStore: VectorStore) {
    this.logger = new Logger('HybridRetriever');
    this.vectorStore = vectorStore;

    this.logger.info('HybridRetriever initialized');
  }

  /**
   * Perform hybrid search combining vector and keyword search
   */
  public async search(
    query: string,
    options: HybridSearchOptions = {}
  ): Promise<HybridSearchResult[]> {
    const startTime = Date.now();

    const k = options.k || this.DEFAULT_K;
    const vectorWeight = options.vectorWeight ?? this.DEFAULT_VECTOR_WEIGHT;
    const keywordWeight = options.keywordWeight ?? this.DEFAULT_KEYWORD_WEIGHT;
    const minSimilarity = options.minSimilarity ?? this.DEFAULT_MIN_SIMILARITY;

    this.logger.info('Starting hybrid search', {
      query: query.substring(0, 100),
      k,
      vectorWeight,
      keywordWeight,
    });

    try {
      // Step 1: Perform vector similarity search
      // Fetch more candidates than needed for re-ranking
      const candidateCount = Math.max(k * 3, 20);
      const vectorResults = await this.vectorStore.similaritySearchWithScore(
        query,
        candidateCount
      );

      this.logger.debug('Vector search complete', {
        candidateCount: vectorResults.length,
      });

      // Step 2: Extract keywords from query
      const keywords = this.extractKeywords(query, options.customStopWords);

      this.logger.debug('Keywords extracted', {
        keywords,
        count: keywords.length,
      });

      // Step 3: Calculate keyword scores for all candidates
      const hybridResults: HybridSearchResult[] = vectorResults.map(([doc, vectorScore]) => {
        const keywordScore = this.calculateKeywordScore(
          doc.pageContent,
          keywords,
          options.keywordBoosting
        );

        // Handle LanceDB distance/similarity score normalization
        // LangChain's LanceDB returns undefined score but stores L2 distance in metadata
        let normalizedVectorScore: number;
        
        // First, get the actual distance value
        let distance: number | undefined = vectorScore;
        if (distance === undefined || isNaN(distance)) {
          // LanceDB stores the actual distance in metadata._distance
          distance = doc.metadata?._distance;
        }
        
        if (distance === undefined || isNaN(distance)) {
          // No distance available - use neutral score for keyword-only ranking
          normalizedVectorScore = 0.5;
          this.logger.debug('No distance score available, using neutral score');
        } else if (distance < 0) {
          // Negative distances can occur with dot product similarity
          // Convert to similarity: smaller absolute distance = higher similarity
          normalizedVectorScore = 1 / (1 + Math.abs(distance));
        } else if (distance <= 2.0) {
          // Small positive values likely indicate cosine similarity (0-2 range)
          // where 0 = identical, 2 = opposite
          // Convert to similarity score: 0 distance → 1.0 similarity, 2 distance → 0.0 similarity
          normalizedVectorScore = Math.max(0, 1 - (distance / 2));
        } else {
          // Large positive values indicate L2/Euclidean distance
          // Convert distance to similarity: smaller distance = higher similarity
          // Using formula: similarity = 1 / (1 + distance)
          normalizedVectorScore = 1 / (1 + distance);
        }

        // Calculate hybrid score as weighted combination
        const hybridScore =
          vectorWeight * normalizedVectorScore +
          keywordWeight * keywordScore;

        return {
          document: doc,
          score: hybridScore,
          vectorScore: normalizedVectorScore,
          keywordScore,
        };
      });

      // Step 4: Re-rank by hybrid score
      hybridResults.sort((a, b) => b.score - a.score);

      // Step 5: Filter by minimum similarity and limit results
      const filteredResults = hybridResults
        .filter((result) => result.score >= minSimilarity)
        .slice(0, k);

      const searchTime = Date.now() - startTime;

      this.logger.info('Hybrid search complete', {
        resultCount: filteredResults.length,
        searchTime,
        avgScore: this.calculateAvgScore(filteredResults),
      });

      // Add explanations if needed
      if (filteredResults.length > 0) {
        this.addExplanations(filteredResults, keywords);
      }

      return filteredResults;
    } catch (error) {
      this.logger.error('Hybrid search failed', {
        error: error instanceof Error ? error.message : String(error),
        query: query.substring(0, 100),
      });
      throw error;
    }
  }

  /**
   * Perform vector-only search (semantic similarity)
   */
  public async vectorSearch(
    query: string,
    k: number = this.DEFAULT_K
  ): Promise<HybridSearchResult[]> {
    this.logger.debug('Performing vector-only search', { query, k });

    try {
      const results = await this.vectorStore.similaritySearchWithScore(query, k);

      return results.map(([doc, score]) => ({
        document: doc,
        score,
        vectorScore: score,
        keywordScore: 0,
      }));
    } catch (error) {
      this.logger.error('Vector search failed', {
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      });
      throw error;
    }
  }

  /**
   * Perform keyword-only search (BM25-like)
   */
  public async keywordSearch(
    query: string,
    k: number = this.DEFAULT_K
  ): Promise<HybridSearchResult[]> {
    this.logger.debug('Performing keyword-only search', { query, k });

    try {
      // Fetch more candidates for keyword filtering
      const candidateCount = Math.max(k * 5, 50);
      const allDocs = await this.vectorStore.similaritySearchWithScore(
        query,
        candidateCount
      );

      const keywords = this.extractKeywords(query);

      // Score documents by keyword matching only
      const keywordResults: HybridSearchResult[] = allDocs.map(([doc]) => {
        const keywordScore = this.calculateKeywordScore(doc.pageContent, keywords);

        return {
          document: doc,
          score: keywordScore,
          vectorScore: 0,
          keywordScore,
        };
      });

      // Sort by keyword score and return top-k
      keywordResults.sort((a, b) => b.keywordScore - a.keywordScore);
      return keywordResults.slice(0, k);
    } catch (error) {
      this.logger.error('Keyword search failed', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  // ==================== Private Methods ====================

  /**
   * Extract keywords from query text
   */
  private extractKeywords(query: string, customStopWords?: string[]): string[] {
    // Combine default and custom stop words
    const stopWords = customStopWords
      ? new Set([...this.STOP_WORDS, ...customStopWords])
      : this.STOP_WORDS;

    // Tokenize and filter
    const tokens = query
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .split(/\s+/)
      .filter((word) => word.length > 2 && !stopWords.has(word));

    // Remove duplicates
    return [...new Set(tokens)];
  }

  /**
   * Calculate keyword matching score for a document
   * Uses BM25-like scoring with term frequency and document length normalization
   */
  private calculateKeywordScore(
    text: string,
    keywords: string[],
    boosting: boolean = true
  ): number {
    if (keywords.length === 0) {
      return 0;
    }

    const textLower = text.toLowerCase();
    const textWords = textLower.split(/\s+/);
    const textLength = textWords.length;

    let score = 0;

    for (const keyword of keywords) {
      // Count occurrences
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      const matches = textLower.match(regex);
      const termFrequency = matches ? matches.length : 0;

      if (termFrequency > 0) {
        // BM25-like scoring
        // TF component: log-scaled term frequency
        const tfScore = Math.log(1 + termFrequency);

        // Length normalization (penalize very long documents)
        const lengthNorm = 1 / (1 + Math.log(1 + textLength / 100));

        // Position boosting (keyword near start of document is weighted more)
        let positionBoost = 1;
        if (boosting) {
          const firstOccurrence = textLower.indexOf(keyword);
          if (firstOccurrence >= 0) {
            // Higher boost for keywords appearing earlier
            positionBoost = 1 + (1 - firstOccurrence / textLength);
          }
        }

        score += tfScore * lengthNorm * positionBoost;
      }
    }

    // Normalize by number of keywords (0-1 range)
    return Math.min(1, score / keywords.length);
  }

  /**
   * Add human-readable explanations to results
   */
  private addExplanations(results: HybridSearchResult[], keywords: string[]): void {
    results.forEach((result) => {
      const parts: string[] = [];

      if (result.vectorScore > 0) {
        parts.push(`Semantic: ${(result.vectorScore * 100).toFixed(1)}%`);
      }

      if (result.keywordScore > 0) {
        const matchedKeywords = keywords.filter((kw) =>
          result.document.pageContent.toLowerCase().includes(kw)
        );
        if (matchedKeywords.length > 0) {
          parts.push(
            `Keywords: ${(result.keywordScore * 100).toFixed(1)}% (${matchedKeywords.join(', ')})`
          );
        }
      }

      parts.push(`Overall: ${(result.score * 100).toFixed(1)}%`);

      result.explanation = parts.join(' | ');
    });
  }

  /**
   * Calculate average score for results
   */
  private calculateAvgScore(results: HybridSearchResult[]): number {
    if (results.length === 0) {
      return 0;
    }

    const sum = results.reduce((acc, r) => acc + r.score, 0);
    return sum / results.length;
  }

  /**
   * Update the vector store (useful for testing or switching topics)
   */
  public setVectorStore(vectorStore: VectorStore): void {
    this.vectorStore = vectorStore;
    this.logger.debug('Vector store updated');
  }
}
