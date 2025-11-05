/**
 * RAG Query Tool for Copilot/LLM Agent integration
 */

import * as vscode from 'vscode';
import { EmbeddingService } from './embeddingService';
import { VectorDatabaseService } from './vectorDatabase';
import { RAGQueryParams, RAGQueryResult } from './types';
import { TOOLS, CONFIG } from './constants';
import { AgentOrchestrator, AgenticRAGConfig } from './agentOrchestrator';
import { WorkspaceContextProvider } from './workspaceContext';

export class RAGTool {
  private embeddingService: EmbeddingService;
  private vectorDb: VectorDatabaseService;
  private agentOrchestrator: AgentOrchestrator;

  constructor() {
    this.embeddingService = EmbeddingService.getInstance();
    this.vectorDb = VectorDatabaseService.getInstance();
    this.agentOrchestrator = new AgentOrchestrator();
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
      // Get configuration
      const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
      const configTopK = config.get<number>('topK', 5);
      // Only use tool parameter if explicitly provided, otherwise use config value
      const topK = params.topK !== undefined ? params.topK : configTopK;

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

      // Determine if we should use agentic mode
      const useAgenticMode = params.useAgenticMode ?? config.get<boolean>(CONFIG.USE_AGENTIC_MODE, false);

      let searchResults;
      let agenticMetadata;

      if (useAgenticMode) {
        // Use Agentic RAG with multi-step retrieval
        const agenticConfig = this.buildAgenticConfig(params, config);

        // Build context for LLM (if enabled)
        const context = agenticConfig.useLLM ? {
          topicName: topicMatch.topic.name,
          topicDescription: topicMatch.topic.description,
          documentCount: topicMatch.topic.documentCount,
          // Could add recent queries here if we track them
        } : undefined;

        // Get workspace context (if LLM enabled and includeWorkspaceContext is true)
        const includeWorkspace = config.get<boolean>(CONFIG.AGENTIC_INCLUDE_WORKSPACE, true);
        const workspaceContext = agenticConfig.useLLM && includeWorkspace
          ? await WorkspaceContextProvider.getContext({
              includeSelection: true,
              includeActiveFile: true,
              includeWorkspace: true,
              maxCodeLength: 1000,
            })
          : undefined;

        const agenticResult = await this.agentOrchestrator.executeAgenticQuery(
          topicMatch.topic.id,
          params.query,
          agenticConfig,
          topK,
          context,
          workspaceContext
        );

        searchResults = agenticResult.finalResults;
        agenticMetadata = {
          mode: 'agentic' as const,
          steps: agenticResult.steps,
          totalIterations: agenticResult.totalIterations,
          queryComplexity: agenticResult.queryPlan.complexity,
          confidence: agenticResult.confidence,
        };
      } else {
        // Use simple single-shot retrieval
        const queryEmbedding = await this.embeddingService.embed(params.query);
        searchResults = await this.vectorDb.search(topicMatch.topic.id, queryEmbedding, topK);
        agenticMetadata = {
          mode: 'simple' as const,
        };
      }

      // Format results with heading context
      const results: RAGQueryResult = {
        query: params.query,
        topicName: topicMatch.topic.name,
        topicMatched: topicMatch.matchType,
        requestedTopic: topicMatch.matchType !== 'exact' ? params.topic : undefined,
        availableTopics: topicMatch.availableTopics,
        agenticMetadata,
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
   * Build agentic configuration from parameters and settings
   */
  private buildAgenticConfig(params: RAGQueryParams, config: vscode.WorkspaceConfiguration): AgenticRAGConfig {
    const userConfig = params.agenticConfig || {};

    return {
      maxIterations: userConfig.maxIterations ?? config.get<number>(CONFIG.AGENTIC_MAX_ITERATIONS, 3),
      confidenceThreshold: userConfig.confidenceThreshold ?? config.get<number>(CONFIG.AGENTIC_CONFIDENCE_THRESHOLD, 0.7),
      enableIterativeRefinement: userConfig.enableIterativeRefinement ?? config.get<boolean>(CONFIG.AGENTIC_ITERATIVE_REFINEMENT, true),
      retrievalStrategy: userConfig.retrievalStrategy ?? config.get<'vector' | 'hybrid'>(CONFIG.AGENTIC_RETRIEVAL_STRATEGY, 'hybrid'),
      useLLM: userConfig.useLLM ?? config.get<boolean>(CONFIG.AGENTIC_USE_LLM, false),
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

