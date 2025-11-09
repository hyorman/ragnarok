/**
 * Main extension entry point
 * Refactored to use new LangChain-based architecture
 */

import * as vscode from "vscode";
import { TopicManager } from "./managers/topicManager";
import { EmbeddingService } from "./embeddings/embeddingService";
import { RAGTool } from "./ragTool";
import { CommandHandler } from "./commands";
import { TopicTreeDataProvider } from "./topicTreeView";
import { VIEWS, STATE, COMMANDS } from "./utils/constants";
import { Logger } from "./utils/logger";
import { GitHubTokenManager } from "./utils/githubTokenManager";

const logger = new Logger("Extension");

export async function activate(context: vscode.ExtensionContext) {
  logger.info("RAGnarōk extension activating...");

  try {
    // Initialize TopicManager (singleton with automatic initialization)
    const topicManager = await TopicManager.getInstance(context);

    // Initialize embedding service instance (will load model on first use)
    EmbeddingService.getInstance();

    // Initialize GitHub token manager
    GitHubTokenManager.initialize(context);
    logger.info("GitHub token manager initialized");

    // Register tree view
    const treeDataProvider = new TopicTreeDataProvider();
    const treeView = vscode.window.createTreeView(VIEWS.RAG_TOPICS, {
      treeDataProvider,
      showCollapseAll: true,
    });
    context.subscriptions.push(treeView);

    // Register commands
    await CommandHandler.registerCommands(context, treeDataProvider);

    // Load topics with error handling
    try {
      const topics = await topicManager.getAllTopics();
      logger.info(`Loaded ${topics.length} topics`);
    } catch (dbError) {
      logger.error("Failed to load topics", { error: dbError });
      // If topics index is corrupted, offer to reset it
      const response = await vscode.window.showErrorMessage(
        "Failed to load RAG topics. Would you like to reset the database?",
        "Reset Database",
        "Cancel"
      );

      if (response === "Reset Database") {
        // Delete all topics to reset
        const topics = await topicManager.getAllTopics();
        for (const topic of topics) {
          await topicManager.deleteTopic(topic.id);
        }
        vscode.window.showInformationMessage(
          "Database has been reset successfully."
        );
        logger.info("Database reset completed");
      }
      // Don't throw - let the extension continue working
    }

    // Register RAG tool for Copilot/LLM agents
    try {
      // Check if Language Model API is available
      if (!vscode.lm || typeof vscode.lm.registerTool !== "function") {
        console.warn(
          "Language Model API not available. Requires VS Code 1.90+ and GitHub Copilot Chat."
        );
        vscode.window
          .showWarningMessage(
            "RAG Tool requires VS Code 1.90+ and GitHub Copilot Chat extension to be visible.",
            "Learn More"
          )
          .then((selection) => {
            if (selection === "Learn More") {
              vscode.env.openExternal(
                vscode.Uri.parse(
                  "https://code.visualstudio.com/docs/copilot/copilot-chat"
                )
              );
            }
          });
      } else {
        const ragToolDisposable = RAGTool.register(context);
        console.log("RAG query tool registered successfully");
      }
    } catch (error) {
      console.error("Failed to register RAG tool:", error);
      vscode.window.showWarningMessage(
        `RAG tool registration failed: ${error}`
      );
    }

    // Show welcome message on first activation
    const hasShownWelcome = context.globalState.get(
      STATE.HAS_SHOWN_WELCOME,
      false
    );
    if (!hasShownWelcome) {
      const response = await vscode.window.showInformationMessage(
        "Welcome to RAGnarōk! Create topics and add documents to enable RAG queries.",
        "Create Topic",
        "Learn More"
      );

      if (response === "Create Topic") {
        vscode.commands.executeCommand(COMMANDS.CREATE_TOPIC);
      } else if (response === "Learn More") {
        vscode.env.openExternal(
          vscode.Uri.parse("https://github.com/hyorman/ragnarok")
        );
      }

      await context.globalState.update(STATE.HAS_SHOWN_WELCOME, true);
    }

    logger.info("Extension activation complete");
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logger.error("Failed to activate extension", { error: errorMessage });
    throw error; // Re-throw to signal activation failure
  }
}

export async function deactivate() {
  logger.info("RAGnarōk extension deactivating...");
  // Clear any caches
  const topicManager = await TopicManager.getInstance();
  topicManager.clearCache();
  logger.info("Extension deactivation complete");
}
