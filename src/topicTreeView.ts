/**
 * Tree view for displaying RAG topics and documents
 * Refactored to use TopicManager and display agentic metadata
 */

import * as vscode from "vscode";
import { TopicManager } from "./managers/topicManager";
import { Topic, Document, RetrievalStrategy } from "./utils/types";
import { Logger } from "./utils/logger";
import { CONFIG } from "./utils/constants";
import { EmbeddingService } from "./embeddings/embeddingService";

const logger = new Logger("TopicTreeView");

export class TopicTreeDataProvider
  implements vscode.TreeDataProvider<TopicTreeItem>
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    TopicTreeItem | undefined | null | void
  > = new vscode.EventEmitter<TopicTreeItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    TopicTreeItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  private topicManager: Promise<TopicManager>;
  private embeddingService: EmbeddingService;

  constructor() {
    this.topicManager = TopicManager.getInstance();
    this.embeddingService = EmbeddingService.getInstance();
  }

  refresh(): void {
    logger.debug("Refreshing topic tree view");
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: TopicTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: TopicTreeItem): Promise<TopicTreeItem[]> {
    try {
      const topicManager = await this.topicManager;

      if (!element) {
        // Root level - show configuration status + topics
        const items: TopicTreeItem[] = [];

        // Add configuration status item
        items.push(new TopicTreeItem(null, "config-status"));

        // Add topics
        const topics = await topicManager.getAllTopics();
        logger.debug(`Loaded ${topics.length} topics for tree view`);

        items.push(
          ...topics.map((topic: any) => new TopicTreeItem(topic, "topic"))
        );
        return items;
      } else if (element.type === "config-status") {
        // Show configuration items
        return this.getConfigurationItems();
      } else if (element.type === "topic" && element.topic) {
        // Show statistics and documents for this topic
        const items: TopicTreeItem[] = [];

        // Add stats item
        const stats = await topicManager.getTopicStats(element.topic.id);
        if (stats) {
          items.push(new TopicTreeItem(stats, "topic-stats"));
        }

        // Add documents
        const documents = topicManager.getTopicDocuments(element.topic.id);
        if (documents.length > 0) {
          items.push(
            ...documents.map((doc: any) => new TopicTreeItem(doc, "document"))
          );
        }

        return items;
      } else if (element.type === "topic-stats" && element.data) {
        // Show detailed statistics
        return this.getStatisticsItems(element.data);
      }
      return [];
    } catch (error) {
      logger.error(`Failed to get tree children: ${error}`);
      return [];
    }
  }

  /**
   * Get configuration status items
   */
  private async getConfigurationItems(): Promise<TopicTreeItem[]> {
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    const items: TopicTreeItem[] = [];

    // Embedding model (actual model currently loaded)
    const currentModel = this.embeddingService.getCurrentModel();
    items.push(
      new TopicTreeItem(
        { key: "embedding-model", value: currentModel },
        "config-item"
      )
    );

    // Retrieval strategy (applies to all modes)
    const strategy = config.get<string>(CONFIG.RETRIEVAL_STRATEGY, "hybrid");
    items.push(
      new TopicTreeItem(
        { key: "retrieval-strategy", value: strategy },
        "config-item"
      )
    );

    // Agentic mode status
    const useAgenticMode = config.get<boolean>(CONFIG.USE_AGENTIC_MODE, false);
    items.push(
      new TopicTreeItem(
        { key: "agentic-mode", value: useAgenticMode },
        "config-item"
      )
    );

    // LLM usage
    if (useAgenticMode) {
      const useLLM = config.get<boolean>(CONFIG.AGENTIC_USE_LLM, false);
      items.push(
        new TopicTreeItem({ key: "use-llm", value: useLLM }, "config-item")
      );

      // Max iterations
      const maxIterations = config.get<number>(
        CONFIG.AGENTIC_MAX_ITERATIONS,
        3
      );
      items.push(
        new TopicTreeItem(
          { key: "max-iterations", value: maxIterations },
          "config-item"
        )
      );

      // Confidence threshold
      const threshold = config.get<number>(
        CONFIG.AGENTIC_CONFIDENCE_THRESHOLD,
        0.7
      );
      items.push(
        new TopicTreeItem(
          { key: "confidence-threshold", value: threshold },
          "config-item"
        )
      );
    }

    return items;
  }

  /**
   * Get detailed statistics items for a topic
   */
  private getStatisticsItems(stats: any): TopicTreeItem[] {
    const items: TopicTreeItem[] = [];

    // Document count
    items.push(
      new TopicTreeItem(
        { key: "document-count", value: stats.documentCount },
        "stat-item"
      )
    );

    // Chunk count
    items.push(
      new TopicTreeItem(
        { key: "chunk-count", value: stats.chunkCount },
        "stat-item"
      )
    );

    // Embedding model
    items.push(
      new TopicTreeItem(
        { key: "embedding-model", value: stats.embeddingModel },
        "stat-item"
      )
    );

    // Last updated
    const lastUpdated = new Date(stats.lastUpdated).toLocaleString();
    items.push(
      new TopicTreeItem(
        { key: "last-updated", value: lastUpdated },
        "stat-item"
      )
    );

    return items;
  }
}

export class TopicTreeItem extends vscode.TreeItem {
  constructor(
    public readonly data: Topic | Document | any,
    public readonly type:
      | "topic"
      | "document"
      | "config-status"
      | "config-item"
      | "topic-stats"
      | "stat-item"
  ) {
    super(
      TopicTreeItem.getLabel(data, type),
      TopicTreeItem.getCollapsibleState(type)
    );

    this.setupTreeItem(data, type);
  }

  private static getLabel(data: any, type: string): string {
    switch (type) {
      case "topic":
        return data.name;
      case "document":
        return `📄 ${data.name}`;
      case "config-status":
        return "⚙️ Configuration";
      case "config-item":
        return TopicTreeItem.formatConfigLabel(data);
      case "topic-stats":
        return "📊 Statistics";
      case "stat-item":
        return TopicTreeItem.formatStatLabel(data);
      default:
        return "Unknown";
    }
  }

  private static getCollapsibleState(
    type: string
  ): vscode.TreeItemCollapsibleState {
    switch (type) {
      case "topic":
      case "config-status":
      case "topic-stats":
        return vscode.TreeItemCollapsibleState.Collapsed;
      default:
        return vscode.TreeItemCollapsibleState.None;
    }
  }

  private static formatConfigLabel(configData: any): string {
    const { key, value } = configData;
    switch (key) {
      case "agentic-mode":
        return `Agentic Mode: ${value ? "✅ Enabled" : "❌ Disabled"}`;
      case "use-llm":
        return `LLM Planning: ${value ? "✅ Enabled" : "❌ Disabled"}`;
      case "retrieval-strategy":
        return `Strategy: ${
          value === RetrievalStrategy.HYBRID
            ? "🔀 Hybrid"
            : value === RetrievalStrategy.VECTOR
            ? "🎯 Vector"
            : value === RetrievalStrategy.ENSEMBLE
            ? "🎭 Ensemble"
            : value === RetrievalStrategy.BM25
            ? "🔍 BM25"
            : "❓ Unknown"
        }`;
      case "embedding-model":
        return `Embedding Model: ${value}`;
      case "max-iterations":
        return `Max Iterations: ${value}`;
      case "confidence-threshold":
        return `Confidence: ${(value * 100).toFixed(0)}%`;
      default:
        return `${key}: ${value}`;
    }
  }

  private static formatStatLabel(statData: any): string {
    const { key, value } = statData;
    switch (key) {
      case "document-count":
        return `📄 Documents: ${value}`;
      case "chunk-count":
        return `📦 Chunks: ${value}`;
      case "embedding-model":
        return `🤖 Model: ${value}`;
      case "last-updated":
        return `🕒 Updated: ${value}`;
      default:
        return `${key}: ${value}`;
    }
  }

  private setupTreeItem(data: any, type: string): void {
    switch (type) {
      case "topic":
        const topic = data as Topic;
        this.tooltip = topic.description || topic.name;
        this.description = `${topic.documentCount} document${
          topic.documentCount !== 1 ? "s" : ""
        }`;
        this.contextValue = "topic";
        this.iconPath = new vscode.ThemeIcon("folder");
        break;

      case "document":
        const doc = data as Document;
        this.tooltip = `${doc.name} (${doc.fileType})`;
        this.description = `${doc.chunkCount} chunks`;
        this.contextValue = "document";
        this.iconPath = new vscode.ThemeIcon("file");
        break;

      case "config-status":
        this.tooltip = "View current RAG configuration";
        this.contextValue = "config-status";
        this.iconPath = new vscode.ThemeIcon("settings-gear");
        break;

      case "config-item":
        this.tooltip = `Click to change this setting`;
        this.contextValue = "config-item";
        this.iconPath = new vscode.ThemeIcon("symbol-property");
        break;

      case "topic-stats":
        this.tooltip = "Topic statistics and metadata";
        this.contextValue = "topic-stats";
        this.iconPath = new vscode.ThemeIcon("graph");
        break;

      case "stat-item":
        this.tooltip = `${data.key}: ${data.value}`;
        this.contextValue = "stat-item";
        this.iconPath = new vscode.ThemeIcon("symbol-numeric");
        break;
    }
  }

  get topic(): Topic | undefined {
    return this.type === "topic" ? (this.data as Topic) : undefined;
  }

  get document(): Document | undefined {
    return this.type === "document" ? (this.data as Document) : undefined;
  }
}
