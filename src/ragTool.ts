/**
 * RAG Query Tool for Copilot/LLM Agent integration
 * Refactored to use new LangChain-based architecture with RAGAgent
 */

import * as vscode from 'vscode';
import { EmbeddingService } from './embeddingService';
import { TopicManager } from './managers/topicManager';
import { RAGAgent } from './agents/ragAgent';
import { RAGQueryParams, RAGQueryResult } from './types';
import { TOOLS, CONFIG } from './constants';
import { WorkspaceContextProvider } from './workspaceContext';
import { Logger } from './utils/logger';

const logger = new Logger('RAGTool');

export class RAGTool {
  private embeddingService: EmbeddingService;
  private topicManager: Promise<TopicManager>;
  private ragAgents: Map<string, RAGAgent>; // One agent per topic

  constructor() {
    this.embeddingService = EmbeddingService.getInstance();
    this.topicManager = TopicManager.getInstance();
    this.ragAgents = new Map();
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
   * Execute a RAG query (supports both simple and agentic modes)
   */
  public async executeQuery(params: RAGQueryParams): Promise<RAGQueryResult> {
    try {
      logger.info(`Executing RAG query: "${params.query}" for topic: "${params.topic}"`);

      // Get configuration
      const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
      const configTopK = config.get<number>(CONFIG.TOP_K, 5);
      const topK = params.topK !== undefined ? params.topK : configTopK;

      // Initialize services
      await this.embeddingService.initialize();
      const topicManager = await this.topicManager;

      // Find the topic with intelligent matching
      const topicMatch = await this.findBestMatchingTopic(params.topic);

      // Check if topic has any documents
      const stats = await topicManager.getTopicStats(topicMatch.topic.id);
      if (!stats || stats.documentCount === 0) {
        throw new Error(
          `Topic "${topicMatch.topic.name}" exists but has no documents. Add documents using the "RAG: Add Document to Topic" command.`
        );
      }

      logger.info(`Topic matched: ${topicMatch.topic.name} (${topicMatch.matchType}), ${stats.documentCount} documents, ${stats.chunkCount} chunks`);

      // Determine if we should use agentic mode
      const useAgenticMode = params.useAgenticMode ?? config.get<boolean>(CONFIG.USE_AGENTIC_MODE, false);

      // Get or create RAG agent for this topic
      const agent = await this.getOrCreateAgent(topicMatch.topic.id);

      let ragResult: any; // Will be typed by RAGResult or simpleQuery result
      let agenticMetadata;

      if (useAgenticMode) {
        logger.info('Using agentic RAG mode');

        // Build agentic options from params and config
        const agenticOptions = this.buildAgenticOptions(params, config);

        // Get workspace context (if LLM enabled and includeWorkspaceContext is true)
        const includeWorkspace = config.get<boolean>(CONFIG.AGENTIC_INCLUDE_WORKSPACE, true);
        if (agenticOptions.useLLM && includeWorkspace) {
          const wsContext = await WorkspaceContextProvider.getContext({
            includeSelection: true,
            includeActiveFile: true,
            includeWorkspace: true,
            maxCodeLength: 1000,
          });
          // Convert workspace context to string for the agent
          agenticOptions.workspaceContext = JSON.stringify(wsContext, null, 2);
        }

        // Execute agentic query with RAGAgent
        ragResult = await agent.query(params.query, agenticOptions);

        // Convert RAGAgent result to agenticMetadata format
        agenticMetadata = {
          mode: 'agentic' as const,
          steps: ragResult.plan.subQueries.map((sq: any, idx: number) => ({
            stepNumber: idx + 1,
            query: sq.query,
            strategy: ragResult.plan.strategy,
            resultsCount: ragResult.results.filter((r: any) => r.document?.metadata?.subQueryIndex === idx).length,
            confidence: ragResult.avgConfidence,
            reasoning: sq.reasoning,
          })),
          totalIterations: ragResult.iterations,
          queryComplexity: ragResult.plan.complexity,
          confidence: ragResult.avgConfidence,
        };
      } else {
        logger.info('Using simple RAG mode');

        // Use simple query (bypasses planning)
        const simpleResults = await agent.simpleQuery(params.query, topK);

        ragResult = {
          query: params.query,
          results: simpleResults,
          iterations: 1,
          avgConfidence: simpleResults.length > 0
            ? simpleResults.reduce((sum, r) => sum + r.score, 0) / simpleResults.length
            : 0,
          plan: {
            originalQuery: params.query,
            complexity: 'simple' as const,
            subQueries: [{ query: params.query, reasoning: 'Direct retrieval', priority: 'high' as const }],
            strategy: 'sequential' as const,
            explanation: 'Simple single-shot retrieval',
          },
        };

        agenticMetadata = {
          mode: 'simple' as const,
        };
      }

      // Format results for RAGQueryResult
      const formattedResults: RAGQueryResult = {
        query: params.query,
        topicName: topicMatch.topic.name,
        topicMatched: topicMatch.matchType,
        requestedTopic: topicMatch.matchType !== 'exact' ? params.topic : undefined,
        availableTopics: topicMatch.availableTopics,
        agenticMetadata,
        results: ragResult.results.map((result: any) => ({
          text: result.document.pageContent,
          documentName: result.document.metadata.source || 'Unknown',
          similarity: Math.round(result.score * 100) / 100,
          metadata: {
            chunkIndex: result.document.metadata.chunkIndex || 0,
            position: result.document.metadata.loc?.lines
              ? `lines ${result.document.metadata.loc.lines.from}-${result.document.metadata.loc.lines.to}`
              : `chars ${result.document.metadata.startPosition || 0}-${result.document.metadata.endPosition || 0}`,
            headingPath: result.document.metadata.headingPath
              ? (Array.isArray(result.document.metadata.headingPath)
                  ? result.document.metadata.headingPath.join(' → ')
                  : result.document.metadata.headingPath)
              : undefined,
            sectionTitle: result.document.metadata.sectionTitle,
          },
        })),
      };

      logger.info(`Query completed: ${formattedResults.results.length} results, confidence: ${ragResult.avgConfidence.toFixed(2)}`);
      return formattedResults;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`RAG Query Failed: ${errorMessage}`);
      throw new Error(`RAG Query Failed: ${errorMessage}`);
    }
  }

  /**
   * Get or create RAG agent for a topic
   */
  private async getOrCreateAgent(topicId: string): Promise<RAGAgent> {
    // Check cache first
    if (this.ragAgents.has(topicId)) {
      return this.ragAgents.get(topicId)!;
    }

    // Create new agent
    logger.debug(`Creating new RAGAgent for topic: ${topicId}`);
    const agent = new RAGAgent();

    // Get vector store from TopicManager
    const topicManager = await this.topicManager;
    const vectorStore = await topicManager.getVectorStore(topicId);
    if (!vectorStore) {
      throw new Error(`Failed to load vector store for topic: ${topicId}`);
    }

    // Initialize agent with vector store
    await agent.initialize(vectorStore);

    // Cache the agent
    this.ragAgents.set(topicId, agent);

    return agent;
  }

  /**
   * Build agentic options from parameters and settings
   */
  private buildAgenticOptions(
    params: RAGQueryParams,
    config: vscode.WorkspaceConfiguration
  ): {
    topK?: number;
    enableIterativeRefinement?: boolean;
    maxIterations?: number;
    confidenceThreshold?: number;
    useLLM?: boolean;
    retrievalStrategy?: 'vector' | 'hybrid';
    workspaceContext?: string;
  } {
    const userConfig = params.agenticConfig || {};

    return {
      topK: params.topK ?? config.get<number>(CONFIG.TOP_K, 5),
      enableIterativeRefinement: userConfig.enableIterativeRefinement ?? config.get<boolean>(CONFIG.AGENTIC_ITERATIVE_REFINEMENT, true),
      maxIterations: userConfig.maxIterations ?? config.get<number>(CONFIG.AGENTIC_MAX_ITERATIONS, 3),
      confidenceThreshold: userConfig.confidenceThreshold ?? config.get<number>(CONFIG.AGENTIC_CONFIDENCE_THRESHOLD, 0.7),
      useLLM: userConfig.useLLM ?? config.get<boolean>(CONFIG.AGENTIC_USE_LLM, false),
      retrievalStrategy: userConfig.retrievalStrategy ?? config.get<'vector' | 'hybrid'>(CONFIG.AGENTIC_RETRIEVAL_STRATEGY, 'hybrid'),
    };
  }

  /**
   * Find the best matching topic using semantic similarity
   */
  private async findBestMatchingTopic(requestedTopic: string): Promise<{
    topic: any;
    matchType: 'exact' | 'similar' | 'fallback';
    availableTopics?: string[];
  }> {
    // Get all available topics
    const topicManager = await this.topicManager;
    const allTopics = await topicManager.getAllTopics();

    if (allTopics.length === 0) {
      throw new Error(
        'No topics found in the RAG database. Create a topic using the "RAG: Create New Topic" command first.'
      );
    }

    // Try exact match first (case-insensitive)
    const exactMatch = allTopics.find(
      t => t.name.toLowerCase() === requestedTopic.toLowerCase()
    );
    if (exactMatch) {
      logger.debug(`Exact topic match found: ${exactMatch.name}`);
      return { topic: exactMatch, matchType: 'exact' };
    }

    const topicNames = allTopics.map((t: any) => t.name);

    // If only one topic exists, use it as fallback
    if (allTopics.length === 1) {
      logger.debug(`Single topic fallback: ${allTopics[0].name}`);
      return {
        topic: allTopics[0],
        matchType: 'fallback',
        availableTopics: topicNames,
      };
    }

    // Compute semantic similarity between requested topic and all available topics
    logger.debug(`Computing semantic similarity for topic: ${requestedTopic}`);
    const requestedEmbedding = await this.embeddingService.embed(requestedTopic);

    const topicSimilarities = await Promise.all(
      allTopics.map(async (topic: any) => {
        const topicEmbedding = await this.embeddingService.embed(topic.name);
        const similarity = this.embeddingService.cosineSimilarity(
          requestedEmbedding,
          topicEmbedding
        );
        return { topic, similarity };
      })
    );

    // Sort by similarity and get the best match
    topicSimilarities.sort((a: any, b: any) => b.similarity - a.similarity);
    const bestMatch = topicSimilarities[0];

    logger.debug(`Best matching topic: ${bestMatch.topic.name} (similarity: ${bestMatch.similarity.toFixed(3)})`);
    return {
      topic: bestMatch.topic,
      matchType: 'similar',
      availableTopics: topicNames,
    };
  }
}

