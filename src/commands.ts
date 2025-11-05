/**
 * Command handlers for the RAG extension
 * Refactored to use new LangChain-based architecture with TopicManager
 */

import * as vscode from 'vscode';
import { TopicManager } from './managers/topicManager';
import { EmbeddingService } from './embeddingService';
import { TopicTreeDataProvider } from './topicTreeView';
import { COMMANDS } from './constants';
import { Logger } from './utils/logger';

const logger = new Logger('CommandHandler');

export class CommandHandler {
  private topicManager: TopicManager;
  private embeddingService: EmbeddingService;
  private treeDataProvider: TopicTreeDataProvider;

  private constructor(topicManager: TopicManager, treeDataProvider: TopicTreeDataProvider) {
    this.topicManager = topicManager;
    this.embeddingService = EmbeddingService.getInstance();
    this.treeDataProvider = treeDataProvider;
  }

  /**
   * Register all commands
   */
  public static async registerCommands(
    context: vscode.ExtensionContext,
    treeDataProvider: TopicTreeDataProvider
  ): Promise<void> {
    const topicManager = await TopicManager.getInstance();
    const handler = new CommandHandler(topicManager, treeDataProvider);

    context.subscriptions.push(
      vscode.commands.registerCommand(COMMANDS.CREATE_TOPIC, () =>
        handler.createTopic()
      ),
      vscode.commands.registerCommand(COMMANDS.DELETE_TOPIC, (item?: any) =>
        handler.deleteTopic(item)
      ),
      vscode.commands.registerCommand(COMMANDS.LIST_TOPICS, () =>
        handler.listTopics()
      ),
      vscode.commands.registerCommand(COMMANDS.ADD_DOCUMENT, (item?: any) =>
        handler.addDocument(item)
      ),
      vscode.commands.registerCommand(COMMANDS.REFRESH_TOPICS, () =>
        handler.refreshTopics()
      ),
      vscode.commands.registerCommand(COMMANDS.CLEAR_MODEL_CACHE, () =>
        handler.clearModelCache()
      ),
      vscode.commands.registerCommand(COMMANDS.CLEAR_DATABASE, () =>
        handler.clearDatabase()
      )
    );
  }

  /**
   * Create a new topic
   */
  private async createTopic(): Promise<void> {
    try {
      const name = await vscode.window.showInputBox({
        prompt: 'Enter topic name',
        placeHolder: 'e.g., React Documentation, Company Policies',
        validateInput: (value) => {
          if (!value || value.trim().length === 0) {
            return 'Topic name cannot be empty';
          }
          return null;
        },
      });

      if (!name) {
        return;
      }

      const description = await vscode.window.showInputBox({
        prompt: 'Enter topic description (optional)',
        placeHolder: 'Brief description of this topic',
      });

      logger.info(`Creating topic: ${name}`);
      const topic = await this.topicManager.createTopic({
        name: name.trim(),
        description: description?.trim(),
      });

      vscode.window.showInformationMessage(`Topic "${topic.name}" created successfully!`);
      this.treeDataProvider.refresh();
      logger.info(`Topic created: ${topic.id}`);
    } catch (error) {
      logger.error(`Failed to create topic: ${error}`);
      vscode.window.showErrorMessage(`Failed to create topic: ${error}`);
    }
  }

  /**
   * Delete a topic
   */
  private async deleteTopic(item?: any): Promise<void> {
    try {
      let topicToDelete;

      // If called from tree view with item
      if (item && item.topic) {
        topicToDelete = item.topic;
      } else {
        // Called from command palette - show picker
        const topics = await this.topicManager.getAllTopics();

        if (topics.length === 0) {
          vscode.window.showInformationMessage('No topics available to delete.');
          return;
        }

        const selected = await vscode.window.showQuickPick(
          topics.map((t: any) => ({
            label: t.name,
            description: `${t.documentCount} document(s)`,
            detail: t.description,
            topic: t,
          })),
          {
            placeHolder: 'Select a topic to delete',
          }
        );

        if (!selected) {
          return;
        }

        topicToDelete = selected.topic;
      }

      const confirmation = await vscode.window.showWarningMessage(
        `Are you sure you want to delete topic "${topicToDelete.name}"? This will also delete all associated documents and embeddings.`,
        { modal: true },
        'Delete'
      );

      if (confirmation === 'Delete') {
        logger.info(`Deleting topic: ${topicToDelete.name} (${topicToDelete.id})`);
        await this.topicManager.deleteTopic(topicToDelete.id);
        vscode.window.showInformationMessage(`Topic "${topicToDelete.name}" deleted.`);
        this.treeDataProvider.refresh();
        logger.info(`Topic deleted successfully`);
      }
    } catch (error) {
      logger.error(`Failed to delete topic: ${error}`);
      vscode.window.showErrorMessage(`Failed to delete topic: ${error}`);
    }
  }

  /**
   * List all topics
   */
  private async listTopics(): Promise<void> {
    try {
      const topics = await this.topicManager.getAllTopics();

      if (topics.length === 0) {
        vscode.window.showInformationMessage(
          'No topics found. Create a topic using "RAG: Create New Topic" command.'
        );
        return;
      }

      const items = topics.map((t: any) => ({
        label: t.name,
        description: `${t.documentCount} document(s)`,
        detail: t.description || 'No description',
      }));

      await vscode.window.showQuickPick(items, {
        placeHolder: 'Available RAG Topics',
      });
    } catch (error) {
      logger.error(`Failed to list topics: ${error}`);
      vscode.window.showErrorMessage(`Failed to list topics: ${error}`);
    }
  }

  /**
   * Add a document to a topic
   */
  private async addDocument(item?: any): Promise<void> {
    try {
      let selectedTopic: any;

      // If called from tree view with item
      if (item && item.topic) {
        selectedTopic = item.topic;
      } else {
        // Called from command palette - show picker
        const topics = await this.topicManager.getAllTopics();

        if (topics.length === 0) {
          const create = await vscode.window.showInformationMessage(
            'No topics available. Would you like to create one?',
            'Create Topic'
          );

          if (create === 'Create Topic') {
            await this.createTopic();
            return this.addDocument(); // Retry after creating topic
          }
          return;
        }

        const selected = await vscode.window.showQuickPick(
          topics.map((t: any) => ({
            label: t.name,
            description: `${t.documentCount} document(s)`,
            topic: t,
          })),
          {
            placeHolder: 'Select a topic',
          }
        );

        if (!selected) {
          return;
        }

        selectedTopic = selected.topic;
      }

      // Select files (can select multiple)
      const fileUris = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: true, // Allow multiple file selection
        filters: {
          'Supported Documents': ['pdf', 'md', 'markdown', 'html', 'htm', 'txt'],
          'PDF': ['pdf'],
          'Markdown': ['md', 'markdown'],
          'HTML': ['html', 'htm'],
          'Text': ['txt'],
        },
        openLabel: 'Add Document(s)',
      });

      if (!fileUris || fileUris.length === 0) {
        return;
      }

      const filePaths = fileUris.map(uri => uri.fsPath);
      logger.info(`Adding ${filePaths.length} document(s) to topic: ${selectedTopic.name}`);

      // Process documents using TopicManager
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Processing ${filePaths.length} document(s)...`,
          cancellable: false,
        },
        async (progress) => {
          let currentFile = 0;

          const results = await this.topicManager.addDocuments(
            selectedTopic.id,
            filePaths,
            {
              onProgress: (pipelineProgress) => {
                currentFile = Math.floor((pipelineProgress.progress / 100) * filePaths.length);
                progress.report({
                  message: `${pipelineProgress.message} (${currentFile + 1}/${filePaths.length})`,
                  increment: pipelineProgress.progress / filePaths.length,
                });
              }
            }
          );

          progress.report({ message: 'Complete!' });

          const totalChunks = results.reduce((sum, r) => sum + r.pipelineResult.metadata.chunksStored, 0);
          logger.info(`Documents added: ${results.length} files, ${totalChunks} chunks`);
          currentFile++;
        }
      );

      const stats = await this.topicManager.getTopicStats(selectedTopic.id);
      vscode.window.showInformationMessage(
        `${filePaths.length} document(s) added to "${selectedTopic.name}" successfully! Total: ${stats?.documentCount} documents, ${stats?.chunkCount} chunks.`
      );
      this.treeDataProvider.refresh();
    } catch (error) {
      logger.error(`Failed to add document: ${error}`);
      vscode.window.showErrorMessage(`Failed to add document: ${error}`);
    }
  }

  /**
   * Refresh topics view
   */
  private refreshTopics(): void {
    this.treeDataProvider.refresh();
    vscode.window.showInformationMessage('Topics refreshed.');
  }

  /**
   * Clear model cache
   */
  private async clearModelCache(): Promise<void> {
    try {
      await this.embeddingService.clearCache();
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to clear cache: ${error}`);
    }
  }

  /**
   * Clear database
   */
  private async clearDatabase(): Promise<void> {
    try {
      const confirmation = await vscode.window.showWarningMessage(
        'Are you sure you want to clear the entire database? This will delete all topics, documents, and embeddings. This action cannot be undone.',
        { modal: true },
        'Clear Database'
      );

      if (confirmation === 'Clear Database') {
        logger.warn('Clearing entire database');

        // Delete all topics (this will also clear their vector stores)
        const topics = await this.topicManager.getAllTopics();
        for (const topic of topics) {
          await this.topicManager.deleteTopic(topic.id);
        }

        // Clear embedding service cache
        await this.embeddingService.clearCache();

        vscode.window.showInformationMessage('Database cleared successfully.');
        this.treeDataProvider.refresh();
        logger.info('Database cleared');
      }
    } catch (error) {
      logger.error(`Failed to clear database: ${error}`);
      vscode.window.showErrorMessage(`Failed to clear database: ${error}`);
    }
  }
}

