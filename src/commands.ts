/**
 * Command handlers for the RAG extension
 * Refactored to use new LangChain-based architecture with TopicManager
 */

import * as vscode from "vscode";
import { TopicManager } from "./managers/topicManager";
import { EmbeddingService } from "./embeddings/embeddingService";
import { TopicTreeDataProvider } from "./topicTreeView";
import { COMMANDS } from "./utils/constants";
import { Logger } from "./utils/logger";
import { GitHubTokenManager } from "./utils/githubTokenManager";

const logger = new Logger("CommandHandler");

export class CommandHandler {
  private topicManager: TopicManager;
  private embeddingService: EmbeddingService;
  private treeDataProvider: TopicTreeDataProvider;
  private context: vscode.ExtensionContext;
  private tokenManager: GitHubTokenManager;

  private constructor(
    context: vscode.ExtensionContext,
    topicManager: TopicManager,
    treeDataProvider: TopicTreeDataProvider
  ) {
    this.context = context;
    this.topicManager = topicManager;
    this.embeddingService = EmbeddingService.getInstance();
    this.treeDataProvider = treeDataProvider;
    this.tokenManager = GitHubTokenManager.getInstance();
  }

  /**
   * Register all commands
   */
  public static async registerCommands(
    context: vscode.ExtensionContext,
    treeDataProvider: TopicTreeDataProvider
  ): Promise<void> {
    const topicManager = await TopicManager.getInstance();
    const handler = new CommandHandler(context, topicManager, treeDataProvider);

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
      vscode.commands.registerCommand(COMMANDS.ADD_GITHUB_REPO, (item?: any) =>
        handler.addGithubRepo(item)
      ),
      vscode.commands.registerCommand(COMMANDS.REFRESH_TOPICS, () =>
        handler.refreshTopics()
      ),
      vscode.commands.registerCommand(COMMANDS.CLEAR_MODEL_CACHE, () =>
        handler.clearModelCache()
      ),
      vscode.commands.registerCommand(COMMANDS.CLEAR_DATABASE, () =>
        handler.clearDatabase()
      ),
      // GitHub token management commands
      vscode.commands.registerCommand(COMMANDS.ADD_GITHUB_TOKEN, () =>
        handler.addGithubToken()
      ),
      vscode.commands.registerCommand(COMMANDS.LIST_GITHUB_TOKENS, () =>
        handler.listGithubTokens()
      ),
      vscode.commands.registerCommand(COMMANDS.REMOVE_GITHUB_TOKEN, () =>
        handler.removeGithubToken()
      )
    );
  }

  /**
   * Create a new topic
   */
  private async createTopic(): Promise<void> {
    try {
      const name = await vscode.window.showInputBox({
        prompt: "Enter topic name",
        placeHolder: "e.g., React Documentation, Company Policies",
        validateInput: (value) => {
          if (!value || value.trim().length === 0) {
            return "Topic name cannot be empty";
          }
          return null;
        },
      });

      if (!name) {
        return;
      }

      const description = await vscode.window.showInputBox({
        prompt: "Enter topic description (optional)",
        placeHolder: "Brief description of this topic",
      });

      logger.info(`Creating topic: ${name}`);
      const topic = await this.topicManager.createTopic({
        name: name.trim(),
        description: description?.trim(),
      });

      vscode.window.showInformationMessage(
        `Topic "${topic.name}" created successfully!`
      );
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
          vscode.window.showInformationMessage(
            "No topics available to delete."
          );
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
            placeHolder: "Select a topic to delete",
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
        "Delete"
      );

      if (confirmation === "Delete") {
        logger.info(
          `Deleting topic: ${topicToDelete.name} (${topicToDelete.id})`
        );
        await this.topicManager.deleteTopic(topicToDelete.id);
        vscode.window.showInformationMessage(
          `Topic "${topicToDelete.name}" deleted.`
        );
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
        detail: t.description || "No description",
      }));

      await vscode.window.showQuickPick(items, {
        placeHolder: "Available RAG Topics",
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
            "No topics available. Would you like to create one?",
            "Create Topic"
          );

          if (create === "Create Topic") {
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
            placeHolder: "Select a topic",
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
          "Supported Documents": [
            "pdf",
            "md",
            "markdown",
            "html",
            "htm",
            "txt",
          ],
          PDF: ["pdf"],
          Markdown: ["md", "markdown"],
          HTML: ["html", "htm"],
          Text: ["txt"],
        },
        openLabel: "Add Document(s)",
      });

      if (!fileUris || fileUris.length === 0) {
        return;
      }

      const filePaths = fileUris.map((uri) => uri.fsPath);
      logger.info(
        `Adding ${filePaths.length} document(s) to topic: ${selectedTopic.name}`
      );

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
                currentFile = Math.floor(
                  (pipelineProgress.progress / 100) * filePaths.length
                );
                progress.report({
                  message: `${pipelineProgress.message} (${currentFile + 1}/${
                    filePaths.length
                  })`,
                  increment: pipelineProgress.progress / filePaths.length,
                });
              },
            }
          );

          progress.report({ message: "Complete!" });

          const totalChunks = results.reduce(
            (sum, r) => sum + r.pipelineResult.metadata.chunksStored,
            0
          );
          logger.info(
            `Documents added: ${results.length} files, ${totalChunks} chunks`
          );
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
   * Add a GitHub repository to a topic
   */
  private async addGithubRepo(item?: any): Promise<void> {
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
            "No topics available. Would you like to create one?",
            "Create Topic"
          );

          if (create === "Create Topic") {
            await this.createTopic();
            return this.addGithubRepo(); // Retry after creating topic
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
            placeHolder: "Select a topic",
          }
        );

        if (!selected) {
          return;
        }

        selectedTopic = selected.topic;
      }

      // Check for saved GitHub hosts
      const savedHosts = await this.tokenManager.listHosts(this.context);
      let selectedHost: string | undefined;
      let accessToken: string | undefined;

      // If there are saved hosts, offer to use them
      if (savedHosts.length > 0) {
        const hostOptions = [
          ...savedHosts.map((host) => ({
            label: host,
            description: "✅ Saved token available",
            value: host,
          })),
          {
            label: "$(add) Enter custom URL",
            description: "Use a different GitHub server",
            value: "custom",
          },
        ];

        const hostChoice = await vscode.window.showQuickPick(hostOptions, {
          placeHolder: "Select GitHub host or enter custom URL",
        });

        if (!hostChoice) {
          return;
        }

        if (hostChoice.value !== "custom") {
          selectedHost = hostChoice.value;
          accessToken = await this.tokenManager.getToken(selectedHost);
          logger.info(`Using saved token for host: ${selectedHost}`);
        }
      }

      // Get repository path (owner/repo)
      let repoUrl: string;

      if (selectedHost) {
        // Simplified: just ask for owner/repo
        const repoPath = await vscode.window.showInputBox({
          ignoreFocusOut: true,
          prompt: `Enter repository (owner/repo) for ${selectedHost}`,
          placeHolder: "facebook/react",
          validateInput: (value) => {
            if (!value || value.trim().length === 0) {
              return "Repository path cannot be empty";
            }
            if (!/^[\w-]+\/[\w.-]+$/.test(value.trim())) {
              return "Invalid format. Use: owner/repo";
            }
            return null;
          },
        });

        if (!repoPath) {
          return;
        }

        repoUrl = `https://${selectedHost}/${repoPath.trim()}`;
      } else {
        // Full URL entry for custom hosts
        repoUrl =
          (await vscode.window.showInputBox({
            ignoreFocusOut: true,
            prompt: "Enter full GitHub repository URL",
            placeHolder: "https://github.com/owner/repo",
            validateInput: (value) => {
              if (!value || value.trim().length === 0) {
                return "Repository URL cannot be empty";
              }
              if (
                !/^https?:\/\/[a-zA-Z0-9.-]+\/[\w-]+\/[\w.-]+/.test(
                  value.trim()
                )
              ) {
                return "Invalid GitHub URL. Format: https://host/owner/repo";
              }
              return null;
            },
          })) || "";

        if (!repoUrl) {
          return;
        }

        // Extract host and check for token
        const host = this.tokenManager.extractHost(repoUrl);
        if (host) {
          accessToken = await this.tokenManager.getToken(host);
          if (accessToken) {
            logger.info(`Found saved token for host: ${host}`);
          } else {
            // Prompt for token if not found
            const needToken = await vscode.window.showQuickPick(
              [
                {
                  label: "Enter Token",
                  description: "For private repositories",
                  value: true,
                },
                {
                  label: "Continue Without Token",
                  description: "Public repositories only",
                  value: false,
                },
              ],
              {
                placeHolder: `No saved token for ${host}`,
              }
            );

            if (needToken?.value) {
              const tokenInput = await vscode.window.showInputBox({
                prompt: "Enter GitHub access token",
                placeHolder: "ghp_xxxxxxxxxxxxxxxxxxxx",
                password: true,
                ignoreFocusOut: true,
              });

              if (tokenInput && tokenInput.trim().length > 0) {
                accessToken = tokenInput.trim();
                await this.tokenManager.promptToSaveToken(
                  this.context,
                  host,
                  accessToken
                );
              }
            }
          }
        }
      }

      // Get branch
      const branch = await vscode.window.showInputBox({
        prompt: "Enter branch name",
        placeHolder: "main",
        value: "main",
        ignoreFocusOut: true,
      });

      if (!branch) {
        return;
      }

      // Optional: ignore patterns
      const defaultIgnore = "*.github*, *makefile*, **/TEST/**, **/tst/**, *.test.*, node_modules/**";
      const ignoreInput = await vscode.window.showInputBox({
        prompt: "Enter ignore patterns (optional, comma-separated)",
        placeHolder: `${defaultIgnore} (press Enter to accept default)`,
        value: defaultIgnore,
        ignoreFocusOut: true,
      });

      const ignorePaths =
        ignoreInput && ignoreInput.trim().length > 0
          ? ignoreInput
              .split(",")
              .map((p) => p.trim())
              .filter((p) => p.length > 0)
          : undefined;

      logger.info(`Adding GitHub repository to topic: ${selectedTopic.name}`, {
        repoUrl,
        branch,
        ignorePaths,
      });

      // Process repository using TopicManager
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Processing GitHub repository...`,
          cancellable: false,
        },
        async (progress) => {
          progress.report({
            message: "Fetching repository structure (this may take a while)...",
            increment: 10,
          });

          const results = await this.topicManager.addDocuments(
            selectedTopic.id,
            [repoUrl],
            {
              onProgress: (pipelineProgress) => {
                progress.report({
                  message: pipelineProgress.message,
                  increment: (pipelineProgress.progress / 100) * 90,
                });
              },
              loaderOptions: {
                fileType: "github",
                branch,
                recursive: true,
                ignorePaths,
                accessToken,
                maxConcurrency: 10, // Increase concurrency for faster loading
              },
            }
          );

          progress.report({ message: "Complete!", increment: 100 });

          const totalChunks = results.reduce(
            (sum, r) => sum + r.pipelineResult.metadata.chunksStored,
            0
          );
          logger.info(`GitHub repository added: ${totalChunks} chunks`);
        }
      );

      const stats = await this.topicManager.getTopicStats(selectedTopic.id);
      vscode.window.showInformationMessage(
        `GitHub repository added to "${selectedTopic.name}" successfully! Total: ${stats?.documentCount} documents, ${stats?.chunkCount} chunks.`
      );
      this.treeDataProvider.refresh();
    } catch (error) {
      logger.error(`Failed to add GitHub repository: ${error}`);
      const errorMessage = error instanceof Error ? error.message : String(error);
      const errorStack = error instanceof Error ? error.stack : undefined;

      vscode.window.showErrorMessage(
        `Failed to add GitHub repository:\n\n${errorMessage}${errorStack ? '\n\nStack:\n' + errorStack : ''}`,
        "OK"
      );
    }
  }

  /**
   * Refresh topics view
   */
  private refreshTopics(): void {
    this.treeDataProvider.refresh();
    vscode.window.showInformationMessage("Topics refreshed.");
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
        "Are you sure you want to clear the entire database? This will delete all topics, documents, and embeddings. This action cannot be undone.",
        { modal: true },
        "Clear Database"
      );

      if (confirmation === "Clear Database") {
        logger.warn("Clearing entire database");

        // Delete all topics (this will also clear their vector stores)
        const topics = await this.topicManager.getAllTopics();
        for (const topic of topics) {
          await this.topicManager.deleteTopic(topic.id);
        }

        // Clear embedding service cache
        await this.embeddingService.clearCache();

        vscode.window.showInformationMessage("Database cleared successfully.");
        this.treeDataProvider.refresh();
        logger.info("Database cleared");
      }
    } catch (error) {
      logger.error(`Failed to clear database: ${error}`);
      vscode.window.showErrorMessage(`Failed to clear database: ${error}`);
    }
  }

  /**
   * Add GitHub token for a specific host
   */
  private async addGithubToken(): Promise<void> {
    try {
      // Ask for host
      const host = await vscode.window.showInputBox({
        prompt: "Enter GitHub host",
        placeHolder: "e.g., github.com, github.company.com",
        validateInput: (value) => {
          if (!value || value.trim().length === 0) {
            return "Host cannot be empty";
          }
          // Basic validation for host format
          if (!/^[a-zA-Z0-9.-]+$/.test(value.trim())) {
            return "Invalid host format";
          }
          return null;
        },
      });

      if (!host) {
        return;
      }

      // Check if token already exists
      const hasToken = await this.tokenManager.hasToken(host.trim());
      if (hasToken) {
        const overwrite = await vscode.window.showWarningMessage(
          `A token already exists for "${host.trim()}". Do you want to overwrite it?`,
          "Overwrite",
          "Cancel"
        );

        if (overwrite !== "Overwrite") {
          return;
        }
      }

      // Ask for token
      const token = await vscode.window.showInputBox({
        prompt: `Enter GitHub access token for ${host.trim()}`,
        placeHolder: "ghp_xxxxxxxxxxxxxxxxxxxx",
        password: true,
        validateInput: (value) => {
          if (!value || value.trim().length === 0) {
            return "Token cannot be empty";
          }
          return null;
        },
      });

      if (!token) {
        return;
      }

      // Save token
      await this.tokenManager.setToken(host.trim(), token.trim());
      await this.tokenManager.addHostToList(this.context, host.trim());

      vscode.window.showInformationMessage(
        `GitHub token saved for host "${host.trim()}"`
      );
      logger.info(`GitHub token added for host: ${host.trim()}`);
    } catch (error) {
      logger.error(`Failed to add GitHub token: ${error}`);
      vscode.window.showErrorMessage(`Failed to add GitHub token: ${error}`);
    }
  }

  /**
   * List all saved GitHub tokens (hosts only, not the actual tokens)
   */
  private async listGithubTokens(): Promise<void> {
    try {
      const hosts = await this.tokenManager.listHosts(this.context);

      if (hosts.length === 0) {
        vscode.window.showInformationMessage(
          'No GitHub tokens saved. Use "RAG: Add GitHub Token" to add one.'
        );
        return;
      }

      const items = hosts.map((host) => ({
        label: host,
        description: "GitHub host with saved token",
      }));

      await vscode.window.showQuickPick(items, {
        placeHolder: "Saved GitHub Tokens",
      });
    } catch (error) {
      logger.error(`Failed to list GitHub tokens: ${error}`);
      vscode.window.showErrorMessage(`Failed to list GitHub tokens: ${error}`);
    }
  }

  /**
   * Remove a saved GitHub token
   */
  private async removeGithubToken(): Promise<void> {
    try {
      const hosts = await this.tokenManager.listHosts(this.context);

      if (hosts.length === 0) {
        vscode.window.showInformationMessage("No GitHub tokens to remove.");
        return;
      }

      const selected = await vscode.window.showQuickPick(
        hosts.map((host) => ({
          label: host,
          description: "GitHub host",
        })),
        {
          placeHolder: "Select a host to remove its token",
        }
      );

      if (!selected) {
        return;
      }

      const confirmation = await vscode.window.showWarningMessage(
        `Are you sure you want to remove the GitHub token for "${selected.label}"?`,
        "Remove",
        "Cancel"
      );

      if (confirmation === "Remove") {
        await this.tokenManager.deleteToken(selected.label);
        await this.tokenManager.removeHostFromList(
          this.context,
          selected.label
        );

        vscode.window.showInformationMessage(
          `GitHub token removed for "${selected.label}"`
        );
        logger.info(`GitHub token removed for host: ${selected.label}`);
      }
    } catch (error) {
      logger.error(`Failed to remove GitHub token: ${error}`);
      vscode.window.showErrorMessage(`Failed to remove GitHub token: ${error}`);
    }
  }
}
