/**
 * RAG Query Tool for Copilot/LLM Agent integration
 */

import * as vscode from 'vscode';
import { EmbeddingService } from './embeddingService';
import { VectorDatabaseService } from './vectorDatabase';
import { RAGQueryParams, RAGQueryResult } from './types';
import { TOOLS, CONFIG } from './constants';

export class RAGTool {
  private embeddingService: EmbeddingService;
  private vectorDb: VectorDatabaseService;

  constructor() {
    this.embeddingService = EmbeddingService.getInstance();
    this.vectorDb = VectorDatabaseService.getInstance();
  }

  /**
   * Register the RAG query tool with VSCode
   */
  public static register(context: vscode.ExtensionContext): vscode.Disposable {
    const tool = new RAGTool();

    // Register as a language model tool
    const ragTool = vscode.lm.registerTool(TOOLS.RAG_QUERY, {
      invoke: async (options: vscode.LanguageModelToolInvocationOptions<RAGQueryParams>) => {
        const params = options.input;
        const result = await tool.executeQuery(params);
        return new vscode.LanguageModelToolResult([
          new vscode.LanguageModelTextPart(JSON.stringify(result, null, 2))
        ]);
      },
      prepareInvocation: async (
        options: vscode.LanguageModelToolInvocationPrepareOptions<RAGQueryParams>
      ) => {
        const params = options.input;
        return {
          invocationMessage: `Searching RAG database for topic "${params.topic}" with query: "${params.query}"`
        };
      }
    });

    context.subscriptions.push(ragTool);
    return ragTool;
  }

  /**
   * Execute a RAG query
   */
  public async executeQuery(params: RAGQueryParams): Promise<RAGQueryResult> {
    try {
      // Get configuration
      const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
      const defaultTopK = config.get<number>('topK', 5);
      const topK = params.topK || defaultTopK;

      // Initialize embedding service if needed
      await this.embeddingService.initialize();

      // Find the topic with intelligent matching
      const topicMatch = await this.findBestMatchingTopic(params.topic);

      // Check if topic has any documents
      if (topicMatch.topic.documentCount === 0) {
        throw new Error(
          `Topic "${topicMatch.topic.name}" exists but has no documents. Add documents using the "RAG: Add Document to Topic" command.`
        );
      }

      // Generate embedding for the query
      const queryEmbedding = await this.embeddingService.embed(params.query);

      // Search the vector database
      const searchResults = await this.vectorDb.search(topicMatch.topic.id, queryEmbedding, topK);

      // Format results with heading context
      const results: RAGQueryResult = {
        query: params.query,
        topicName: topicMatch.topic.name,
        topicMatched: topicMatch.matchType,
        requestedTopic: topicMatch.matchType !== 'exact' ? params.topic : undefined,
        availableTopics: topicMatch.availableTopics,
        results: searchResults.map((result) => ({
          text: result.chunk.text,
          documentName: result.documentName,
          similarity: Math.round(result.similarity * 100) / 100,
          metadata: {
            chunkIndex: result.chunk.metadata.chunkIndex,
            position: `chars ${result.chunk.metadata.startPosition}-${result.chunk.metadata.endPosition}`,
            headingPath: result.chunk.metadata.headingPath
              ? result.chunk.metadata.headingPath.join(' → ')
              : undefined,
            sectionTitle: result.chunk.metadata.sectionTitle,
          },
        })),
      };

      return results;
    } catch (error) {
      // Return error information in a structured way
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`RAG Query Failed: ${errorMessage}`);
    }
  }

  /**
   * Find the best matching topic using semantic similarity
   */
  private async findBestMatchingTopic(requestedTopic: string): Promise<{
    topic: any;
    matchType: 'exact' | 'similar' | 'fallback';
    availableTopics?: string[];
  }> {
    // Try exact match first
    const exactMatch = await this.vectorDb.getTopicByName(requestedTopic);
    if (exactMatch) {
      return { topic: exactMatch, matchType: 'exact' };
    }

    // Get all available topics
    const allTopics = await this.vectorDb.getTopics();

    if (allTopics.length === 0) {
      throw new Error(
        'No topics found in the RAG database. Create a topic using the "RAG: Create New Topic" command first.'
      );
    }

    const topicNames = allTopics.map(t => t.name);

    // If only one topic exists, use it as fallback
    if (allTopics.length === 1) {
      return {
        topic: allTopics[0],
        matchType: 'fallback',
        availableTopics: topicNames,
      };
    }

    // Compute semantic similarity between requested topic and all available topics
    const requestedEmbedding = await this.embeddingService.embed(requestedTopic);

    const topicSimilarities = await Promise.all(
      allTopics.map(async (topic) => {
        const topicEmbedding = await this.embeddingService.embed(topic.name);
        const similarity = this.embeddingService.cosineSimilarity(
          requestedEmbedding,
          topicEmbedding
        );
        return { topic, similarity };
      })
    );

    // Sort by similarity and get the best match
    topicSimilarities.sort((a, b) => b.similarity - a.similarity);
    const bestMatch = topicSimilarities[0];

    return {
      topic: bestMatch.topic,
      matchType: 'similar',
      availableTopics: topicNames,
    };
  }
}

