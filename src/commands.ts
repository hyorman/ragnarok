/**
 * Command handlers for the RAG extension
 */

import * as vscode from 'vscode';
import { VectorDatabaseService } from './vectorDatabase';
import { EmbeddingService } from './embeddingService';
import { DocumentProcessor } from './documentProcessor';
import { TextChunk } from './types';
import { TopicTreeDataProvider } from './topicTreeView';
import { COMMANDS, CONFIG } from './constants';

export class CommandHandler {
  private vectorDb: VectorDatabaseService;
  private embeddingService: EmbeddingService;
  private treeDataProvider: TopicTreeDataProvider;

  constructor(treeDataProvider: TopicTreeDataProvider) {
    this.vectorDb = VectorDatabaseService.getInstance();
    this.embeddingService = EmbeddingService.getInstance();
    this.treeDataProvider = treeDataProvider;
  }

  /**
   * Register all commands
   */
  public static registerCommands(
    context: vscode.ExtensionContext,
    treeDataProvider: TopicTreeDataProvider
  ): void {
    const handler = new CommandHandler(treeDataProvider);

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

      const topic = await this.vectorDb.createTopic(name.trim(), description?.trim());
      vscode.window.showInformationMessage(`Topic "${topic.name}" created successfully!`);
      this.treeDataProvider.refresh();
    } catch (error) {
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
        const topics = await this.vectorDb.getTopics();

        if (topics.length === 0) {
          vscode.window.showInformationMessage('No topics available to delete.');
          return;
        }

        const selected = await vscode.window.showQuickPick(
          topics.map((t) => ({
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
        await this.vectorDb.deleteTopic(topicToDelete.id);
        vscode.window.showInformationMessage(`Topic "${topicToDelete.name}" deleted.`);
        this.treeDataProvider.refresh();
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to delete topic: ${error}`);
    }
  }

  /**
   * List all topics
   */
  private async listTopics(): Promise<void> {
    try {
      const topics = await this.vectorDb.getTopics();

      if (topics.length === 0) {
        vscode.window.showInformationMessage(
          'No topics found. Create a topic using "RAG: Create New Topic" command.'
        );
        return;
      }

      const items = topics.map((t) => ({
        label: t.name,
        description: `${t.documentCount} document(s)`,
        detail: t.description || 'No description',
      }));

      await vscode.window.showQuickPick(items, {
        placeHolder: 'Available RAG Topics',
      });
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to list topics: ${error}`);
    }
  }

  /**
   * Add a document to a topic
   */
  private async addDocument(item?: any): Promise<void> {
    try {
      let selectedTopic;

      // If called from tree view with item
      if (item && item.topic) {
        selectedTopic = { topic: item.topic };
      } else {
        // Called from command palette - show picker
        const topics = await this.vectorDb.getTopics();

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

        selectedTopic = await vscode.window.showQuickPick(
          topics.map((t) => ({
            label: t.name,
            description: `${t.documentCount} document(s)`,
            topic: t,
          })),
          {
            placeHolder: 'Select a topic',
          }
        );

        if (!selectedTopic) {
          return;
        }
      }

      // Select file
      const fileUris = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: false,
        filters: {
          'Supported Documents': ['pdf', 'md', 'markdown', 'html', 'htm'],
          'PDF': ['pdf'],
          'Markdown': ['md', 'markdown'],
          'HTML': ['html', 'htm'],
        },
        openLabel: 'Add Document',
      });

      if (!fileUris || fileUris.length === 0) {
        return;
      }

      const fileUri = fileUris[0];
      const filePath = fileUri.fsPath;

      // Process document
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'Processing document...',
          cancellable: false,
        },
        async (progress) => {
          progress.report({ message: 'Reading file...' });

          // Process the document
          const processed = await DocumentProcessor.processDocument(filePath);

          progress.report({ message: 'Splitting into semantic chunks...' });

          // Get configuration
          const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
          const chunkSize = config.get<number>('chunkSize', 512);
          const chunkOverlap = config.get<number>('chunkOverlap', 50);

          // Use semantic chunking based on markdown structure
          const semanticChunks = DocumentProcessor.splitIntoSemanticChunks(
            processed.markdown,
            chunkSize,
            chunkOverlap
          );

          progress.report({
            message: `Creating embeddings for ${semanticChunks.length} semantic chunks...`,
          });

          // Initialize embedding service
          await this.embeddingService.initialize();

          // Extract text from semantic chunks
          const chunkTexts = semanticChunks.map(c => c.text);

          // Generate embeddings
          const embeddings = await this.embeddingService.embedBatch(chunkTexts);

          progress.report({ message: 'Saving to database...' });

          // Create chunks with embeddings and semantic metadata
          const chunks: TextChunk[] = semanticChunks.map((semanticChunk, index) => ({
            id: `${Date.now()}-${index}`,
            documentId: '', // Will be set by vectorDb
            topicId: selectedTopic.topic.id,
            text: semanticChunk.text,
            embedding: embeddings[index],
            metadata: {
              documentName: processed.metadata.fileName,
              chunkIndex: index,
              startPosition: semanticChunk.startPosition,
              endPosition: semanticChunk.endPosition,
              headingPath: semanticChunk.headingPath,
              headingLevel: semanticChunk.headingLevel,
              sectionTitle: semanticChunk.sectionTitle,
            },
          }));

          // Determine file type
          let fileType: 'pdf' | 'markdown' | 'html' = 'markdown';
          if (processed.metadata.fileType === 'pdf') {
            fileType = 'pdf';
          } else if (['html', 'htm'].includes(processed.metadata.fileType)) {
            fileType = 'html';
          }

          // Add to database
          await this.vectorDb.addDocument(
            selectedTopic.topic.id,
            processed.metadata.fileName,
            filePath,
            fileType,
            chunks
          );

          progress.report({ message: 'Complete!' });
        }
      );

      vscode.window.showInformationMessage(
        `Document added to "${selectedTopic.topic.name}" successfully!`
      );
      this.treeDataProvider.refresh();
    } catch (error) {
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
        await this.vectorDb.clearDatabase();
        vscode.window.showInformationMessage('Database cleared successfully.');
        this.treeDataProvider.refresh();
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to clear database: ${error}`);
    }
  }
}

