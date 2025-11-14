/**
 * Query Planner Agent - Decomposes complex queries into sub-queries
 * Uses LangChain StructuredOutputParser with Zod schemas
 *
 * Architecture: LLM-powered query analysis and decomposition
 * Determines optimal search strategy (sequential vs parallel)
 */

import * as vscode from 'vscode';
import { z } from 'zod';
import { Logger } from '../utils/logger';

// Zod schema for query plan
const SubQuerySchema = z.object({
  query: z.string().describe('The sub-query to search for'),
  reasoning: z.string().describe('Why this sub-query is needed'),
  topK: z.number().optional().describe('Number of results for this sub-query'),
  priority: z.enum(['high', 'medium', 'low']).optional().describe('Search priority'),
});

const QueryPlanSchema = z.object({
  originalQuery: z.string().describe('The original user query'),
  complexity: z.enum(['simple', 'moderate', 'complex']).describe('Query complexity'),
  subQueries: z.array(SubQuerySchema).describe('Decomposed sub-queries'),
  strategy: z.enum(['sequential', 'parallel']).describe('Execution strategy'),
  explanation: z.string().describe('Brief explanation of the search strategy'),
});

export type SubQuery = z.infer<typeof SubQuerySchema>;
export type QueryPlan = z.infer<typeof QueryPlanSchema>;

export interface QueryPlannerOptions {
  /** Topic name for context */
  topicName?: string;

  /** Workspace context (open files, etc.) */
  workspaceContext?: string;

  /** Maximum number of sub-queries */
  maxSubQueries?: number;

  /** Default topK for sub-queries */
  defaultTopK?: number;

  /** Enable LLM-based planning (requires LM API access) */
  useLLM?: boolean;
}

export interface RefinementContext {
  /** Current results from previous queries */
  currentResults: Array<{
    content: string;
    score: number;
    metadata?: Record<string, any>;
  }>;

  /** Previously executed sub-queries */
  executedQueries: string[];

  /** Average confidence of current results */
  avgConfidence: number;

  /** Number of unique documents retrieved */
  uniqueDocCount: number;

  /** Confidence threshold to meet */
  confidenceThreshold: number;
}

/**
 * Query Planner Agent using Zod for validation
 */
export class QueryPlannerAgent {
  private logger: Logger;

  constructor() {
    this.logger = new Logger('QueryPlannerAgent');
    this.logger.info('QueryPlannerAgent initialized');
  }

  /**
   * Build the prompt for LLM query planning
   */
  private buildPrompt(query: string, context: string): string {
    return `You are a query planning assistant for a RAG (Retrieval-Augmented Generation) system.
Your task is to analyze user queries and create an optimal search strategy.

${context}

User Query: "${query}"

Guidelines:
1. Simple queries (single concept): Use ONE sub-query
2. Moderate queries (2-3 concepts): Break into 2-3 focused sub-queries
3. Complex queries (comparisons, multi-part): Break into multiple specific sub-queries

Strategies:
- Sequential: When results of one query inform the next
- Parallel: When sub-queries are independent

Response Format: Provide a JSON object with this exact structure:
{
  "originalQuery": "the original query",
  "complexity": "simple" | "moderate" | "complex",
  "subQueries": [
    {
      "query": "sub-query text",
      "reasoning": "why this sub-query is needed",
      "topK": 5,
      "priority": "high" | "medium" | "low"
    }
  ],
  "strategy": "sequential" | "parallel",
  "explanation": "brief explanation of the strategy"
}

Provide your analysis as valid JSON:`;
  }

  /**
   * Create a query plan using LLM (if available)
   */
  public async createPlan(
    query: string,
    options: QueryPlannerOptions = {}
  ): Promise<QueryPlan> {
    this.logger.info('Creating query plan', {
      query: query.substring(0, 100),
      useLLM: options.useLLM,
    });

    try {
      // Use LLM-based planning if enabled and available
      if (options.useLLM !== false) {
        const llmPlan = await this.createLLMPlan(query, options);
        if (llmPlan) {
          return llmPlan;
        }
      }

      // Fallback to heuristic-based planning
      this.logger.debug('Using heuristic-based planning (LLM not available)');
      return this.createHeuristicPlan(query, options);
    } catch (error) {
      this.logger.error('Failed to create query plan', {
        error: error instanceof Error ? error.message : String(error),
        query: query.substring(0, 100),
      });

      // Fallback to simple plan
      return this.createSimplePlan(query, options);
    }
  }

  /**
   * Create a plan using VS Code Language Model API
   */
  private async createLLMPlan(
    query: string,
    options: QueryPlannerOptions
  ): Promise<QueryPlan | null> {
    try {
      // Get VS Code Language Model
      const models = await vscode.lm.selectChatModels({ vendor: 'copilot', family: 'gpt-4o' });
      if (models.length === 0) {
        this.logger.debug('No language models available');
        return null;
      }

      const model = models[0];

      // Build context
      const context = this.buildContextString(options);

      // Build prompt
      const prompt = this.buildPrompt(query, context);

      // Send request to LLM
      const messages = [
        { role: 1, content: prompt } as any, // UserMessage role
      ];

      const response = await model.sendRequest(messages, {}, new vscode.CancellationTokenSource().token);

      // Collect response
      let responseText = '';
      for await (const chunk of response.text) {
        responseText += chunk;
      }

      this.logger.debug('LLM response received', {
        responseLength: responseText.length,
      });

      // Parse JSON response
      // Extract JSON from markdown code blocks if present
      const jsonMatch = responseText.match(/```json\n([\s\S]*?)\n```/) || responseText.match(/```\n([\s\S]*?)\n```/);
      const jsonText = jsonMatch ? jsonMatch[1] : responseText;

      const parsedJSON = JSON.parse(jsonText.trim());
      const plan = QueryPlanSchema.parse(parsedJSON) as QueryPlan;

      // Apply constraints
      if (options.maxSubQueries && plan.subQueries.length > options.maxSubQueries) {
        plan.subQueries = plan.subQueries.slice(0, options.maxSubQueries);
      }

      // Set default topK if not specified
      const defaultTopK = options.defaultTopK || 5;
      plan.subQueries.forEach((sq: SubQuery) => {
        if (!sq.topK) {
          sq.topK = defaultTopK;
        }
      });

      this.logger.info('LLM query plan created', {
        complexity: plan.complexity,
        subQueryCount: plan.subQueries.length,
        strategy: plan.strategy,
      });

      return plan;
    } catch (error) {
      this.logger.warn('LLM planning failed, will use fallback', {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  /**
   * Create a plan using heuristic rules (no LLM required)
   */
  private createHeuristicPlan(
    query: string,
    options: QueryPlannerOptions
  ): QueryPlan {
    this.logger.debug('Creating heuristic query plan');

    const defaultTopK = options.defaultTopK || 5;

    // Analyze query for complexity indicators
    const hasComparison = /\b(vs|versus|compare|difference|between|better|worse)\b/i.test(query);
    const hasMultipleQuestions = (query.match(/\?/g) || []).length > 1;
    const hasMultipleConcepts = query.split(/\band\b|\bor\b/i).length > 2;
    const isLongQuery = query.split(/\s+/).length > 15;

    let complexity: 'simple' | 'moderate' | 'complex';
    let subQueries: SubQuery[];
    let strategy: 'sequential' | 'parallel';
    let explanation: string;

    if (hasComparison) {
      // Comparison query
      complexity = 'complex';
      strategy = 'parallel';

      const parts = query.split(/\b(vs|versus|compare|difference|between)\b/i);
      subQueries = parts
        .filter((p) => p.trim().length > 3)
        .filter((p) => !/^(vs|versus|compare|difference|between|and)$/i.test(p.trim()))
        .map((part, index) => ({
          query: part.trim(),
          reasoning: `Search for information about ${part.trim()}`,
          topK: defaultTopK,
          priority: 'high' as const,
        }));

      explanation = 'Comparison query broken into parallel searches for each concept';
    } else if (hasMultipleQuestions || hasMultipleConcepts) {
      // Multiple concepts - moderate complexity
      complexity = 'moderate';
      strategy = 'parallel';

      // Split by common delimiters
      const parts = query.split(/[.!?;]\s+|\band\b/i);
      subQueries = parts
        .filter((p) => p.trim().length > 5)
        .slice(0, options.maxSubQueries || 3)
        .map((part) => ({
          query: part.trim(),
          reasoning: `Search for ${part.trim()}`,
          topK: defaultTopK,
          priority: 'medium' as const,
        }));

      explanation = 'Multi-concept query split into parallel searches';
    } else if (isLongQuery) {
      // Long query - extract key phrases
      complexity = 'moderate';
      strategy = 'sequential';

      // Use the full query plus extract key noun phrases
      subQueries = [
        {
          query: query,
          reasoning: 'Full query for comprehensive search',
          topK: defaultTopK,
          priority: 'high' as const,
        },
      ];

      explanation = 'Long query searched as-is with follow-up capability';
    } else {
      // Simple query - single sub-query
      complexity = 'simple';
      strategy = 'parallel';

      subQueries = [
        {
          query: query,
          reasoning: 'Direct search for the query',
          topK: defaultTopK,
          priority: 'high' as const,
        },
      ];

      explanation = 'Simple, focused query requires single search';
    }

    // Ensure we have at least one sub-query
    if (subQueries.length === 0) {
      subQueries = [
        {
          query: query,
          reasoning: 'Fallback to full query search',
          topK: defaultTopK,
          priority: 'high' as const,
        },
      ];
    }

    const plan: QueryPlan = {
      originalQuery: query,
      complexity,
      subQueries,
      strategy,
      explanation,
    };

    this.logger.info('Heuristic query plan created', {
      complexity: plan.complexity,
      subQueryCount: plan.subQueries.length,
      strategy: plan.strategy,
    });

    return plan;
  }

  /**
   * Create a simple fallback plan (single query)
   */
  private createSimplePlan(query: string, options: QueryPlannerOptions): QueryPlan {
    this.logger.debug('Creating simple fallback plan');

    const defaultTopK = options.defaultTopK || 5;

    return {
      originalQuery: query,
      complexity: 'simple',
      subQueries: [
        {
          query: query,
          reasoning: 'Direct search',
          topK: defaultTopK,
          priority: 'high',
        },
      ],
      strategy: 'parallel',
      explanation: 'Simple single-query search',
    };
  }

  /**
   * Build context string from options
   */
  private buildContextString(options: QueryPlannerOptions): string {
    const parts: string[] = [];

    if (options.topicName) {
      parts.push(`Topic: ${options.topicName}`);
    }

    if (options.workspaceContext) {
      parts.push(`Workspace Context: ${options.workspaceContext}`);
    }

    if (parts.length === 0) {
      return 'No additional context provided.';
    }

    return parts.join('\n');
  }

  /**
   * Validate a query plan
   */
  public validatePlan(plan: QueryPlan): boolean {
    if (!plan.originalQuery || plan.originalQuery.trim().length === 0) {
      this.logger.warn('Invalid plan: empty original query');
      return false;
    }

    if (!plan.subQueries || plan.subQueries.length === 0) {
      this.logger.warn('Invalid plan: no sub-queries');
      return false;
    }

    for (const sq of plan.subQueries) {
      if (!sq.query || sq.query.trim().length === 0) {
        this.logger.warn('Invalid plan: empty sub-query');
        return false;
      }
    }

    return true;
  }

  /**
   * Refine query plan based on current results and gaps
   */
  public async refinePlan(
    currentPlan: QueryPlan,
    refinementContext: RefinementContext,
    options: QueryPlannerOptions = {}
  ): Promise<QueryPlan | null> {
    this.logger.info('Refining query plan based on results', {
      currentResultCount: refinementContext.currentResults.length,
      avgConfidence: refinementContext.avgConfidence,
      useLLM: options.useLLM,
    });

    try {
      if (options.useLLM !== false) {
        // Use LLM to analyze gaps and generate refined queries
        const llmPlan = await this.refinePlanWithLLM(currentPlan, refinementContext, options);
        if (llmPlan) {
          return llmPlan;
        }
      }

      // Fallback to heuristic-based refinement
      this.logger.debug('Using heuristic-based refinement');
      return this.refinePlanHeuristically(currentPlan, refinementContext, options);
    } catch (error) {
      this.logger.error('Query plan refinement failed', {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  /**
   * Refine query plan using LLM to analyze gaps
   */
  private async refinePlanWithLLM(
    currentPlan: QueryPlan,
    context: RefinementContext,
    options: QueryPlannerOptions
  ): Promise<QueryPlan | null> {
    try {
      // Get VS Code Language Model
      const config = vscode.workspace.getConfiguration('ragnarok');
      const modelFamily = config.get<string>('agenticLLMModel', 'gpt-4o');

      const models = await vscode.lm.selectChatModels({
        vendor: 'copilot',
        family: modelFamily
      });

      if (models.length === 0) {
        this.logger.debug('No language models available for refinement');
        return null;
      }

      const model = models[0];

      // Build results summary
      const resultsSummary = context.currentResults
        .slice(0, 5)
        .map((r, i) => {
          const excerpt = r.content.substring(0, 150).replace(/\n/g, ' ');
          return `${i + 1}. [Score: ${r.score.toFixed(2)}] ${excerpt}...`;
        })
        .join('\n');

      // Build context string
      const contextStr = this.buildContextString(options);

      // Build prompt for gap analysis
      const prompt = `You are analyzing search results to identify information gaps.

${contextStr}

Original Query: "${currentPlan.originalQuery}"

Current Sub-Queries Executed:
${context.executedQueries.map((q, i) => `${i + 1}. ${q}`).join('\n')}

Results Summary:
- Total results: ${context.currentResults.length}
- Average relevance: ${context.avgConfidence.toFixed(2)}
- Unique documents: ${context.uniqueDocCount}
- Confidence threshold: ${context.confidenceThreshold}

Top Result Excerpts:
${resultsSummary}

Your task: Identify information gaps and generate 1-3 refined follow-up queries to fill these gaps.

Guidelines:
1. Only suggest queries if there are clear gaps or low confidence
2. Focus on aspects NOT well-covered in current results
3. Make queries specific and targeted
4. If results are sufficient, return an empty array

Response Format (JSON):
{
  "hasGaps": true/false,
  "gapAnalysis": "brief explanation of what's missing",
  "refinedQueries": [
    {
      "query": "refined query text",
      "reasoning": "why this query fills a gap",
      "topK": 5,
      "priority": "high" | "medium" | "low"
    }
  ]
}

Provide your analysis as valid JSON:`;

      // Send request to LLM
      const messages = [{ role: 1, content: prompt } as any];
      const response = await model.sendRequest(
        messages,
        {},
        new vscode.CancellationTokenSource().token
      );

      // Collect response
      let responseText = '';
      for await (const chunk of response.text) {
        responseText += chunk;
      }

      this.logger.debug('LLM refinement response received', {
        responseLength: responseText.length,
      });

      // Parse JSON response
      const jsonMatch = responseText.match(/```json\n([\s\S]*?)\n```/) ||
                       responseText.match(/```\n([\s\S]*?)\n```/);
      const jsonText = jsonMatch ? jsonMatch[1] : responseText;
      const refinementData = JSON.parse(jsonText.trim());

      // Check if refinement is needed
      if (!refinementData.hasGaps || !refinementData.refinedQueries ||
          refinementData.refinedQueries.length === 0) {
        this.logger.info('LLM determined no refinement needed', {
          analysis: refinementData.gapAnalysis,
        });
        return null;
      }

      this.logger.info('LLM identified gaps and generated refined queries', {
        analysis: refinementData.gapAnalysis,
        queryCount: refinementData.refinedQueries.length,
      });

      // Apply constraints
      let refinedQueries = refinementData.refinedQueries;
      if (options.maxSubQueries && refinedQueries.length > options.maxSubQueries) {
        refinedQueries = refinedQueries.slice(0, options.maxSubQueries);
      }

      // Set default topK if not specified
      const defaultTopK = options.defaultTopK || 5;
      refinedQueries.forEach((sq: SubQuery) => {
        if (!sq.topK) {
          sq.topK = defaultTopK;
        }
      });

      // Create refined plan
      const refinedPlan: QueryPlan = {
        originalQuery: currentPlan.originalQuery,
        complexity: 'moderate',
        subQueries: refinedQueries,
        strategy: 'parallel',
        explanation: `Refinement: ${refinementData.gapAnalysis}`,
      };

      return refinedPlan;
    } catch (error) {
      this.logger.error('LLM-based refinement failed', {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  /**
   * Refine query plan using heuristic rules
   */
  private refinePlanHeuristically(
    currentPlan: QueryPlan,
    context: RefinementContext,
    options: QueryPlannerOptions
  ): QueryPlan | null {
    this.logger.debug('Heuristic gap analysis', {
      avgConfidence: context.avgConfidence,
      uniqueDocs: context.uniqueDocCount,
      resultCount: context.currentResults.length,
    });

    // Check if refinement is needed based on heuristics
    const needsRefinement =
      context.avgConfidence < context.confidenceThreshold * 1.2 || // Close to threshold
      context.uniqueDocCount < 3 || // Too few unique documents
      context.currentResults.length < (options.defaultTopK || 5); // Fewer results than expected

    if (!needsRefinement) {
      this.logger.debug('Heuristics indicate no refinement needed');
      return null;
    }

    // Generate refined queries based on the original query
    const refinedSubQueries: SubQuery[] = [];
    const defaultTopK = options.defaultTopK || 5;

    // Strategy 1: Add more specific terms if confidence is low
    if (context.avgConfidence < context.confidenceThreshold) {
      // Extract key terms and create targeted queries
      const words = currentPlan.originalQuery.split(/\s+/)
        .filter(w => w.length > 4)
        .slice(0, 3);

      for (const word of words) {
        refinedSubQueries.push({
          query: `${word} ${currentPlan.originalQuery}`,
          reasoning: `Focus on "${word}" aspect of the query`,
          topK: defaultTopK,
          priority: 'medium',
        });
      }
    }

    // Strategy 2: Try alternative phrasings if few results
    if (context.currentResults.length < defaultTopK) {
      // Create synonym-based variations
      const alternativeQuery = currentPlan.originalQuery
        .replace(/\bhow\b/gi, 'what way')
        .replace(/\bwhat\b/gi, 'which')
        .replace(/\bwhy\b/gi, 'reason for');

      if (alternativeQuery !== currentPlan.originalQuery) {
        refinedSubQueries.push({
          query: alternativeQuery,
          reasoning: 'Alternative phrasing of the query',
          topK: defaultTopK,
          priority: 'medium',
        });
      }
    }

    // Strategy 3: Broaden search if too specific
    if (context.uniqueDocCount < 3) {
      // Extract core concepts (remove modifiers)
      const coreQuery = currentPlan.originalQuery
        .replace(/\b(very|extremely|highly|most|best|worst)\b/gi, '')
        .trim();

      if (coreQuery !== currentPlan.originalQuery && coreQuery.length > 5) {
        refinedSubQueries.push({
          query: coreQuery,
          reasoning: 'Broader search for core concepts',
          topK: defaultTopK,
          priority: 'low',
        });
      }
    }

    // Limit to max 3 refined queries
    const limitedQueries = refinedSubQueries.slice(0, 3);

    if (limitedQueries.length === 0) {
      this.logger.debug('No heuristic refinements generated');
      return null;
    }

    this.logger.info('Generated heuristic refinements', {
      refinementCount: limitedQueries.length,
    });

    return {
      originalQuery: currentPlan.originalQuery,
      complexity: 'moderate',
      subQueries: limitedQueries,
      strategy: 'parallel',
      explanation: 'Heuristic refinement based on low confidence or sparse results',
    };
  }
}
