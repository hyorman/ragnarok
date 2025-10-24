/**
 * Tree view for displaying RAG topics and documents
 */

import * as vscode from 'vscode';
import { VectorDatabaseService } from './vectorDatabase';
import { Topic, Document } from './types';

export class TopicTreeDataProvider implements vscode.TreeDataProvider<TopicTreeItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<TopicTreeItem | undefined | null | void> =
    new vscode.EventEmitter<TopicTreeItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<TopicTreeItem | undefined | null | void> =
    this._onDidChangeTreeData.event;

  private vectorDb: VectorDatabaseService;

  constructor() {
    this.vectorDb = VectorDatabaseService.getInstance();
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: TopicTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: TopicTreeItem): Promise<TopicTreeItem[]> {
    if (!element) {
      // Root level - show topics
      const topics = await this.vectorDb.getTopics();
      return topics.map((topic) => new TopicTreeItem(topic, 'topic'));
    } else if (element.type === 'topic' && element.topic) {
      // Show documents for this topic
      const documents = await this.vectorDb.getDocumentsByTopic(element.topic.id);
      return documents.map((doc) => new TopicTreeItem(doc, 'document'));
    }
    return [];
  }
}

export class TopicTreeItem extends vscode.TreeItem {
  constructor(
    public readonly data: Topic | Document,
    public readonly type: 'topic' | 'document'
  ) {
    super(
      type === 'topic' ? (data as Topic).name : (data as Document).name,
      type === 'topic'
        ? vscode.TreeItemCollapsibleState.Collapsed
        : vscode.TreeItemCollapsibleState.None
    );

    if (type === 'topic') {
      const topic = data as Topic;
      this.tooltip = topic.description || topic.name;
      this.description = `${topic.documentCount} document${topic.documentCount !== 1 ? 's' : ''}`;
      this.contextValue = 'topic';
      this.iconPath = new vscode.ThemeIcon('folder');
    } else {
      const doc = data as Document;
      this.tooltip = `${doc.name} (${doc.fileType})`;
      this.description = `${doc.chunkCount} chunks`;
      this.contextValue = 'document';
      this.iconPath = new vscode.ThemeIcon('file');
    }
  }

  get topic(): Topic | undefined {
    return this.type === 'topic' ? (this.data as Topic) : undefined;
  }

  get document(): Document | undefined {
    return this.type === 'document' ? (this.data as Document) : undefined;
  }
}

