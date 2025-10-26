/**
 * LLM-Powered Query Planner for True Agentic RAG
 * Uses VS Code's Language Model API to intelligently plan queries
 */

import * as vscode from 'vscode';
import { SearchResult } from './types';
import { WorkspaceContext, WorkspaceContextProvider } from './workspaceContext';
import { CONFIG } from './constants';

export interface SubQuery {
  query: string;
  reasoning: string;
  topK?: number;
  dependencies?: number[];
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

export class LLMQueryPlanner {
  private model: vscode.LanguageModelChat | null = null;
  private modelFamily: string | null = null;

  /**
   * Initialize and get available language model
   */
  private async getModel(): Promise<vscode.LanguageModelChat | null> {
    // Get configured model family
    const config = vscode.workspace.getConfiguration();
    const configuredModel = config.get<string>(CONFIG.AGENTIC_LLM_MODEL, 'gpt-4o');

    // If model family changed, reset cached model
    if (this.modelFamily && this.modelFamily !== configuredModel) {
      this.model = null;
    }
    this.modelFamily = configuredModel;

    if (this.model) {
      return this.model;
    }

    try {
      // Get available models with configured family
      const models = await vscode.lm.selectChatModels({
        vendor: 'copilot',
        family: configuredModel,
      });

      if (models.length > 0) {
        this.model = models[0];
        return this.model;
      }
    } catch (error) {
      console.error('Failed to get language model:', error);
    }

    return null;
  }

  /**
   * Create query plan using LLM reasoning
   */
  public async createPlan(
    query: string,
    enableDecomposition: boolean,
    context?: {
      topicName?: string;
      topicDescription?: string;
      documentCount?: number;
      recentQueries?: string[];
    },
    workspaceContext?: WorkspaceContext
  ): Promise<QueryPlan> {
    if (!enableDecomposition) {
      return {
        originalQuery: query,
        subQueries: [{ query, reasoning: 'Direct query execution', topK: 5 }],
        strategy: 'sequential',
        complexity: 'simple',
      };
    }

    const model = await this.getModel();
    if (!model) {
      // Fallback to heuristic approach if LLM not available
      console.warn('LLM not available, using heuristic query planning');
      return this.fallbackHeuristicPlan(query);
    }

    try {
      const prompt = this.buildQueryPlanningPrompt(query, context);
      const messages = this.buildContextualMessages(context, workspaceContext);
      messages.push(vscode.LanguageModelChatMessage.User(prompt));

      const response = await model.sendRequest(messages, {}, new vscode.CancellationTokenSource().token);

      // Parse LLM response
      let fullResponse = '';
      for await (const chunk of response.text) {
        fullResponse += chunk;
      }

      return this.parseQueryPlan(query, fullResponse);
    } catch (error) {
      console.error('LLM query planning failed:', error);
      return this.fallbackHeuristicPlan(query);
    }
  }

  /**
   * Build contextual messages to prime the LLM with domain knowledge
   */
  private buildContextualMessages(
    context?: {
      topicName?: string;
      topicDescription?: string;
      documentCount?: number;
      recentQueries?: string[];
    },
    workspaceContext?: WorkspaceContext
  ): vscode.LanguageModelChatMessage[] {
    const messages: vscode.LanguageModelChatMessage[] = [];

    let contextInfo = 'You are a query planning assistant for a RAG system.';

    // Add workspace context first (if available)
    if (workspaceContext) {
      contextInfo += '\n\n' + WorkspaceContextProvider.formatContextForLLM(workspaceContext);
      contextInfo += '\n\nUse this workspace context to understand what the user is working on and make more relevant search queries.';
    }

    // Add RAG topic context
    if (context) {
      contextInfo += '\n\n**RAG Database Context:**';

      if (context.topicName) {
        contextInfo += `\nCurrent Topic: "${context.topicName}"`;
      }

      if (context.topicDescription) {
        contextInfo += `\nTopic Description: ${context.topicDescription}`;
      }

      if (context.documentCount) {
        contextInfo += `\nAvailable Documents: ${context.documentCount} documents indexed`;
      }

      if (context.recentQueries && context.recentQueries.length > 0) {
        contextInfo += `\n\nRecent Queries in this topic:\n${context.recentQueries.map(q => `- "${q}"`).join('\n')}`;
        contextInfo += '\n\nConsider these recent queries when planning - users may be building on previous knowledge.';
      }
    }

    // Add context as assistant message (simulates system context)
    if (contextInfo.length > 0) {
      messages.push(vscode.LanguageModelChatMessage.Assistant(contextInfo));
    }

    return messages;
  }

  /**
   * Build prompt for query planning
   */
  private buildQueryPlanningPrompt(
    query: string,
    context?: {
      topicName?: string;
      topicDescription?: string;
      documentCount?: number;
      recentQueries?: string[];
    }
  ): string {
    let prompt = `Analyze the user's query and create a retrieval plan.

User Query: "${query}"

Analyze this query and provide:
1. Complexity level (simple/moderate/complex)
2. Sub-queries needed to fully answer this question
3. Reasoning for each sub-query
4. Whether sub-queries can be executed in parallel or must be sequential

Guidelines:
- Simple queries: Single concept, can be answered with one search
- Moderate queries: Multiple concepts, 2-3 searches needed
- Complex queries: Comparisons, temporal relationships, multi-step reasoning

Respond in this JSON format:
{
  "complexity": "simple|moderate|complex",
  "subQueries": [
    {
      "query": "specific search query",
      "reasoning": "why this query is needed",
      "topK": 5
    }
  ],
  "strategy": "parallel|sequential"
}

Respond ONLY with valid JSON, no other text.`;

    return prompt;
  }

  /**
   * Parse LLM response into query plan
   */
  private parseQueryPlan(originalQuery: string, llmResponse: string): QueryPlan {
    try {
      // Extract JSON from response (in case LLM adds markdown formatting)
      const jsonMatch = llmResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);

      return {
        originalQuery,
        subQueries: parsed.subQueries || [{ query: originalQuery, reasoning: 'Default', topK: 5 }],
        strategy: parsed.strategy || 'sequential',
        complexity: parsed.complexity || 'moderate',
      };
    } catch (error) {
      console.error('Failed to parse LLM response:', error);
      return this.fallbackHeuristicPlan(originalQuery);
    }
  }

  /**
   * Fallback heuristic plan when LLM unavailable
   */
  private fallbackHeuristicPlan(query: string): QueryPlan {
    const lowerQuery = query.toLowerCase();

    // Simple heuristic complexity
    let complexity: 'simple' | 'moderate' | 'complex' = 'simple';
    if (/\b(compare|versus|difference)\b/.test(lowerQuery)) {
      complexity = 'complex';
    } else if (/\b(and|also|how|why)\b/.test(lowerQuery)) {
      complexity = 'moderate';
    }

    return {
      originalQuery: query,
      subQueries: [{ query, reasoning: 'Heuristic-based single query', topK: 5 }],
      strategy: 'sequential',
      complexity,
    };
  }

  /**
   * Generate follow-up query using LLM
   */
  public async generateFollowUpQuery(
    originalQuery: string,
    existingResults: SearchResult[],
    gaps: string[]
  ): Promise<FollowUpQuery | null> {
    if (gaps.length === 0) {
      return null;
    }

    const model = await this.getModel();
    if (!model) {
      return this.fallbackFollowUpQuery(originalQuery, gaps);
    }

    try {
      const prompt = this.buildFollowUpPrompt(originalQuery, existingResults, gaps);
      const messages = [vscode.LanguageModelChatMessage.User(prompt)];

      const response = await model.sendRequest(messages, {}, new vscode.CancellationTokenSource().token);

      let fullResponse = '';
      for await (const chunk of response.text) {
        fullResponse += chunk;
      }

      return this.parseFollowUpQuery(fullResponse);
    } catch (error) {
      console.error('LLM follow-up query generation failed:', error);
      return this.fallbackFollowUpQuery(originalQuery, gaps);
    }
  }

  /**
   * Build prompt for follow-up query generation
   */
  private buildFollowUpPrompt(
    originalQuery: string,
    existingResults: SearchResult[],
    gaps: string[]
  ): string {
    const resultsSummary = existingResults
      .slice(0, 3)
      .map((r, i) => `${i + 1}. ${r.chunk.text.substring(0, 100)}...`)
      .join('\n');

    return `You are helping refine a RAG search that hasn't fully answered the user's question.

Original Query: "${originalQuery}"

Current Results Summary:
${resultsSummary}

Identified Gaps:
${gaps.map((g, i) => `${i + 1}. ${g}`).join('\n')}

Generate a follow-up search query that would fill the most important gap.

Respond in this JSON format:
{
  "query": "specific follow-up search query",
  "reasoning": "why this query addresses the gaps"
}

Respond ONLY with valid JSON, no other text.`;
  }

  /**
   * Parse follow-up query from LLM response
   */
  private parseFollowUpQuery(llmResponse: string): FollowUpQuery | null {
    try {
      const jsonMatch = llmResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        return null;
      }

      const parsed = JSON.parse(jsonMatch[0]);
      return {
        query: parsed.query,
        reasoning: parsed.reasoning,
      };
    } catch (error) {
      console.error('Failed to parse follow-up query:', error);
      return null;
    }
  }

  /**
   * Fallback follow-up query generation
   */
  private fallbackFollowUpQuery(originalQuery: string, gaps: string[]): FollowUpQuery | null {
    if (gaps.length === 0) {
      return null;
    }

    const mainGap = gaps[0];
    const queryWords = originalQuery.split(' ').filter((w) => w.length > 3);
    const mainConcept = queryWords[0] || 'information';

    return {
      query: `${mainConcept} ${mainGap}`,
      reasoning: `Addressing gap: ${mainGap}`,
    };
  }
}

