import * as vscode from 'vscode';
import { CONFIG } from './constants';

let cachedModel: vscode.LanguageModelChat | null = null;
let cachedModelFamily: string | null = null;

/**
 * Return a selected chat model for the configured agentic LLM family.
 * Caches the selected model per-family to avoid repeated selection calls.
 */
export async function getChatModel(): Promise<vscode.LanguageModelChat | null> {
  const config = vscode.workspace.getConfiguration();
  const configuredModel = config.get<string>(CONFIG.AGENTIC_LLM_MODEL, 'gpt-4o');

  // Reset cache when the configured family changes
  if (cachedModelFamily && cachedModelFamily !== configuredModel) {
    cachedModel = null;
  }
  cachedModelFamily = configuredModel;

  if (cachedModel) return cachedModel;

  try {
    const models = await vscode.lm.selectChatModels({ vendor: 'copilot', family: configuredModel });
    if (models.length > 0) {
      cachedModel = models[0];
      return cachedModel;
    }
  } catch (error) {
    console.error('Failed to get language model:', error);
  }

  return null;
}
