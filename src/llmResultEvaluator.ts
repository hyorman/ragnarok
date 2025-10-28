/**
 * LLM-Powered Result Evaluator for True Agentic RAG
 * Uses VS Code's Language Model API to intelligently evaluate search results
 */

import * as vscode from 'vscode';
import { SearchResult } from './types';
import { CONFIG } from './constants';
import { EvaluationResult, IResultEvaluator } from './resultEvaluatorBase';

export class LLMResultEvaluator implements IResultEvaluator {
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
   * Evaluate search results using LLM reasoning
   */
  public async evaluate(
    query: string,
    results: SearchResult[],
    confidenceThreshold: number,
    context?: {
      topicName?: string;
      previousSteps?: string[];
    }
  ): Promise<EvaluationResult> {
    if (results.length === 0) {
      return {
        confidence: 0,
        isComplete: false,
        gaps: ['No results found'],
        reasoning: 'No search results returned',
      };
    }

    const model = await this.getModel();
    if (!model) {
      console.warn('LLM not available, using heuristic evaluation');
      return this.fallbackHeuristicEvaluation(query, results, confidenceThreshold);
    }

    try {
      const prompt = this.buildEvaluationPrompt(query, results, context);
      const messages: vscode.LanguageModelChatMessage[] = [];

      // Add context if available
      if (context) {
        let contextInfo = 'You are evaluating search results from a RAG system.';
        if (context.topicName) {
          contextInfo += `\n\nTopic: "${context.topicName}"`;
        }
        if (context.previousSteps && context.previousSteps.length > 0) {
          contextInfo += `\n\nPrevious search steps in this session:\n${context.previousSteps.join('\n')}`;
          contextInfo += '\n\nConsider what information has already been gathered.';
        }
        messages.push(vscode.LanguageModelChatMessage.Assistant(contextInfo));
      }

      messages.push(vscode.LanguageModelChatMessage.User(prompt));

      const response = await model.sendRequest(messages, {}, new vscode.CancellationTokenSource().token);

      let fullResponse = '';
      for await (const chunk of response.text) {
        fullResponse += chunk;
      }

      return this.parseEvaluationResult(fullResponse, confidenceThreshold);
    } catch (error) {
      console.error('LLM evaluation failed:', error);
      return this.fallbackHeuristicEvaluation(query, results, confidenceThreshold);
    }
  }

  /**
   * Build prompt for result evaluation
   */
  private buildEvaluationPrompt(
    query: string,
    results: SearchResult[],
    context?: {
      topicName?: string;
      previousSteps?: string[];
    }
  ): string {
    const topResults = results.slice(0, 5);
    const resultsSummary = topResults
      .map((r, i) => {
        return `Result ${i + 1} (similarity: ${r.similarity.toFixed(2)}):
Document: ${r.documentName}
Text: ${r.chunk.text.substring(0, 200)}${r.chunk.text.length > 200 ? '...' : ''}
`;
      })
      .join('\n');

    return `You are evaluating search results from a RAG system to determine if they adequately answer the user's query.

User Query: "${query}"

Search Results:
${resultsSummary}

Evaluate these results and determine:
1. Confidence score (0-1): How well do these results answer the query?
2. Is complete (true/false): Are the results sufficient to answer the query?
3. Information gaps: What key information is missing (if any)?
4. Reasoning: Brief explanation of your evaluation

Consider:
- Relevance: Do results directly address the query?
- Completeness: Is all requested information present?
- Quality: Are results from authoritative sources?
- Specificity: Do results provide specific details or just general info?

Respond in this JSON format:
{
  "confidence": 0.85,
  "isComplete": true,
  "gaps": ["missing examples", "no step-by-step instructions"],
  "reasoning": "Results provide good overview but lack practical examples"
}

Respond ONLY with valid JSON, no other text.`;
  }

  /**
   * Parse LLM evaluation response
   */
  private parseEvaluationResult(
    llmResponse: string,
    confidenceThreshold: number
  ): EvaluationResult {
    try {
      const jsonMatch = llmResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      const parsed = JSON.parse(jsonMatch[0]);

      return {
        confidence: parsed.confidence ?? 0.5,
        isComplete: parsed.isComplete ?? parsed.confidence >= confidenceThreshold,
        gaps: parsed.gaps ?? [],
        reasoning: parsed.reasoning ?? 'LLM evaluation completed',
      };
    } catch (error) {
      console.error('Failed to parse evaluation result:', error);
      return {
        confidence: 0.5,
        isComplete: false,
        gaps: ['Evaluation parsing failed'],
        reasoning: 'Could not parse LLM response',
      };
    }
  }

  /**
   * Fallback heuristic evaluation when LLM unavailable
   */
  private fallbackHeuristicEvaluation(
    query: string,
    results: SearchResult[],
    confidenceThreshold: number
  ): EvaluationResult {
    // Simple heuristic: average similarity score
    const avgSimilarity =
      results.slice(0, 5).reduce((sum, r) => sum + r.similarity, 0) /
      Math.min(results.length, 5);

    // Check for diversity
    const uniqueDocs = new Set(results.map((r) => r.documentName)).size;
    const diversityScore = uniqueDocs / Math.min(results.length, 5);

    // Check keyword coverage
    const queryWords = query
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 3);
    const coveredWords = queryWords.filter((word) =>
      results.some((r) => r.chunk.text.toLowerCase().includes(word))
    ).length;
    const coverageScore = queryWords.length > 0 ? coveredWords / queryWords.length : 0.5;

    // Weighted confidence
    const confidence = avgSimilarity * 0.5 + diversityScore * 0.2 + coverageScore * 0.3;

    // Detect gaps heuristically
    const gaps: string[] = [];
    const queryLower = query.toLowerCase();
    const allText = results.map((r) => r.chunk.text.toLowerCase()).join(' ');

    if (/\b(example|examples)\b/.test(queryLower) && !/\b(example|for example|such as)\b/.test(allText)) {
      gaps.push('specific examples');
    }
    if (/\b(how|steps|procedure)\b/.test(queryLower) && !/\b(first|second|step)\b/.test(allText)) {
      gaps.push('step-by-step instructions');
    }
    if (/\b(why|reason)\b/.test(queryLower) && !/\b(because|reason|due to)\b/.test(allText)) {
      gaps.push('explanatory reasoning');
    }

    return {
      confidence: Math.round(confidence * 100) / 100,
      isComplete: confidence >= confidenceThreshold && gaps.length === 0,
      gaps,
      reasoning: `Heuristic evaluation: similarity=${avgSimilarity.toFixed(2)}, diversity=${diversityScore.toFixed(2)}, coverage=${coverageScore.toFixed(2)}`,
    };
  }
}

