/**
 * Main extension entry point
 */

import * as vscode from 'vscode';
import { VectorDatabaseService } from './vectorDatabase';
import { EmbeddingService } from './embeddingService';
import { RAGTool } from './ragTool';
import { CommandHandler } from './commands';
import { TopicTreeDataProvider } from './topicTreeView';
import { VIEWS, STATE, COMMANDS } from './constants';

export async function activate(context: vscode.ExtensionContext) {
  console.log('RAGnarōk extension is now active');

  try {
    // Initialize services
    VectorDatabaseService.initialize(context);
    const vectorDb = VectorDatabaseService.getInstance();
    await vectorDb.loadDatabase();

    // Initialize embedding service instance (will load model on first use)
    EmbeddingService.getInstance();

    // Register tree view
    const treeDataProvider = new TopicTreeDataProvider();
    const treeView = vscode.window.createTreeView(VIEWS.RAG_TOPICS, {
      treeDataProvider,
      showCollapseAll: true,
    });
    context.subscriptions.push(treeView);

    // Register commands
    CommandHandler.registerCommands(context, treeDataProvider);

    // Load database with error handling for corrupted data
    try {
      await vectorDb.loadDatabase();
    } catch (dbError) {
      console.error('Database load error:', dbError);
      // If database is corrupted, offer to reset it
      const response = await vscode.window.showErrorMessage(
        'Failed to load RAG database. Would you like to reset it?',
        'Reset Database',
        'Cancel'
      );

      if (response === 'Reset Database') {
        await vectorDb.clearDatabase();
        vscode.window.showInformationMessage('Database has been reset successfully.');
      }
      // Don't throw - let the extension continue working
    }

    // Register RAG tool for Copilot/LLM agents
    try {
      // Check if Language Model API is available
      if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {
        console.warn('Language Model API not available. Requires VS Code 1.90+ and GitHub Copilot Chat.');
        vscode.window.showWarningMessage(
          'RAG Tool requires VS Code 1.90+ and GitHub Copilot Chat extension to be visible.',
          'Learn More'
        ).then(selection => {
          if (selection === 'Learn More') {
            vscode.env.openExternal(vscode.Uri.parse('https://code.visualstudio.com/docs/copilot/copilot-chat'));
          }
        });
      } else {
        const ragToolDisposable = RAGTool.register(context);
        console.log('RAG query tool registered successfully');
        console.log('Tool name: ragQuery');
        console.log('To use: Open Copilot Chat and look for "RAG Query" in the tools list');
      }
    } catch (error) {
      console.error('Failed to register RAG tool:', error);
      vscode.window.showWarningMessage(
        `RAG tool registration failed: ${error}`
      );
    }

    // Show welcome message on first activation
    const hasShownWelcome = context.globalState.get(STATE.HAS_SHOWN_WELCOME, false);
    if (!hasShownWelcome) {
      const response = await vscode.window.showInformationMessage(
        'Welcome to RAGnarōk! Create topics and add documents to enable RAG queries.',
        'Create Topic',
        'Learn More'
      );

      if (response === 'Create Topic') {
        vscode.commands.executeCommand(COMMANDS.CREATE_TOPIC);
      } else if (response === 'Learn More') {
        vscode.env.openExternal(vscode.Uri.parse('https://github.com/yourusername/ragnarok'));
      }

      await context.globalState.update(STATE.HAS_SHOWN_WELCOME, true);
    }

    vscode.window.showInformationMessage('RAGnarōk extension activated successfully!');
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to activate RAGnarōk: ${error}`);
    console.error('Activation error:', error);
    throw error; // Re-throw to signal activation failure
  }
}

export function deactivate() {
  console.log('RAGnarōk extension is now deactivated');
}

