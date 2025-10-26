/**
 * Vector database implementation using per-topic file-based storage
 * Each topic gets its own JSON file with documents and embeddings
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as path from 'path';
import { Topic, Document, TextChunk, SearchResult, TopicData, TopicsIndex } from './types';
import { EmbeddingService } from './embeddingService';
import { EXTENSION } from './constants';

export class VectorDatabaseService {
  private static instance: VectorDatabaseService;
  private context: vscode.ExtensionContext;
  private topicsIndex: TopicsIndex | null = null;
  private embeddingService: EmbeddingService;
  // Cache for loaded topic data
  private topicCache: Map<string, TopicData> = new Map();

  private constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.embeddingService = EmbeddingService.getInstance();
  }

  public static initialize(context: vscode.ExtensionContext): VectorDatabaseService {
    if (!VectorDatabaseService.instance) {
      VectorDatabaseService.instance = new VectorDatabaseService(context);
    }
    return VectorDatabaseService.instance;
  }

  public static getInstance(): VectorDatabaseService {
    if (!VectorDatabaseService.instance) {
      throw new Error('VectorDatabaseService not initialized');
    }
    return VectorDatabaseService.instance;
  }

  /**
   * Get the database directory path
   */
  private getDatabaseDir(): string {
    return path.join(this.context.globalStorageUri.fsPath, EXTENSION.DATABASE_DIR);
  }

  /**
   * Get the topics index file path
   */
  private getTopicsIndexPath(): string {
    return path.join(this.getDatabaseDir(), EXTENSION.TOPICS_INDEX_FILENAME);
  }

  /**
   * Get the file path for a specific topic's data
   */
  private getTopicDataPath(topicId: string): string {
    return path.join(this.getDatabaseDir(), `topic-${topicId}.json`);
  }

  /**
   * Ensure storage directory exists
   */
  private async ensureStorageDirectory(): Promise<void> {
    try {
      await fs.mkdir(this.getDatabaseDir(), { recursive: true });
    } catch (error) {
      // Directory might already exist, that's fine
    }
  }


  /**
   * Load topics index from file
   */
  private async loadTopicsIndex(): Promise<void> {
    try {
      const indexPath = this.getTopicsIndexPath();
      const data = await fs.readFile(indexPath, 'utf-8');
      this.topicsIndex = JSON.parse(data);
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        // Initialize empty index
        this.topicsIndex = {
          topics: {},
          modelName: this.embeddingService.getCurrentModel() || 'unknown',
          lastUpdated: Date.now(),
        };
        await this.saveTopicsIndex();
      } else {
        throw error;
      }
    }
  }

  /**
   * Save topics index to file
   */
  private async saveTopicsIndex(): Promise<void> {
    if (!this.topicsIndex) {
      throw new Error('Topics index not initialized');
    }

    await this.ensureStorageDirectory();
    this.topicsIndex.lastUpdated = Date.now();
    const data = JSON.stringify(this.topicsIndex, null, 2);
    const indexPath = this.getTopicsIndexPath();
    await fs.writeFile(indexPath, data, 'utf-8');
  }

  /**
   * Load a specific topic's data from file
   */
  private async loadTopicData(topicId: string): Promise<TopicData> {
    // Check cache first
    if (this.topicCache.has(topicId)) {
      return this.topicCache.get(topicId)!;
    }

    try {
      const topicPath = this.getTopicDataPath(topicId);
      const data = await fs.readFile(topicPath, 'utf-8');
      const topicData: TopicData = JSON.parse(data);

      // Cache for future use
      this.topicCache.set(topicId, topicData);
      return topicData;
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        // Topic file doesn't exist - this shouldn't happen if index is correct
        throw new Error(`Topic data file not found for topic ID: ${topicId}`);
      }
      throw error;
    }
  }

  /**
   * Save a specific topic's data to file
   */
  private async saveTopicData(topicId: string, topicData: TopicData): Promise<void> {
    await this.ensureStorageDirectory();
    topicData.lastUpdated = Date.now();

    const data = JSON.stringify(topicData, null, 2);
    const topicPath = this.getTopicDataPath(topicId);
    await fs.writeFile(topicPath, data, 'utf-8');

    // Update cache
    this.topicCache.set(topicId, topicData);
  }

  /**
   * Initialize database - load topics index
   */
  public async loadDatabase(): Promise<void> {
    try {
      await this.ensureStorageDirectory();
      await this.loadTopicsIndex();
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to load database: ${error}`);
      throw error;
    }
  }

  /**
   * Create a new topic
   */
  public async createTopic(name: string, description?: string): Promise<Topic> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }

    // Check if topic already exists
    const existingTopic = Object.values(this.topicsIndex!.topics).find(
      (t) => t.name.toLowerCase() === name.toLowerCase()
    );

    if (existingTopic) {
      throw new Error(`Topic "${name}" already exists`);
    }

    const topic: Topic = {
      id: this.generateId(),
      name,
      description,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      documentCount: 0,
    };

    // Add to index
    this.topicsIndex!.topics[topic.id] = topic;
    await this.saveTopicsIndex();

    // Create empty topic data file
    const topicData: TopicData = {
      topic,
      documents: {},
      chunks: {},
      modelName: this.embeddingService.getCurrentModel() || 'unknown',
      lastUpdated: Date.now(),
    };
    await this.saveTopicData(topic.id, topicData);

    return topic;
  }

  /**
   * Get all topics
   */
  public async getTopics(): Promise<Topic[]> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }
    return Object.values(this.topicsIndex!.topics);
  }

  /**
   * Get a topic by name
   */
  public async getTopicByName(name: string): Promise<Topic | null> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }

    const topic = Object.values(this.topicsIndex!.topics).find(
      (t) => t.name.toLowerCase() === name.toLowerCase()
    );

    return topic || null;
  }

  /**
   * Delete a topic and all its documents and chunks
   */
  public async deleteTopic(topicId: string): Promise<void> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }

    // Delete the topic from index
    delete this.topicsIndex!.topics[topicId];
    await this.saveTopicsIndex();

    // Delete the topic data file
    try {
      const topicPath = this.getTopicDataPath(topicId);
      await fs.unlink(topicPath);
    } catch (error) {
      console.error(`Failed to delete topic file for ${topicId}:`, error);
    }

    // Remove from cache
    this.topicCache.delete(topicId);
  }

  /**
   * Add a document to a topic
   */
  public async addDocument(
    topicId: string,
    name: string,
    filePath: string,
    fileType: 'pdf' | 'markdown' | 'html',
    chunks: TextChunk[]
  ): Promise<Document> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }

    if (!this.topicsIndex!.topics[topicId]) {
      throw new Error('Topic not found');
    }

    // Load topic data
    const topicData = await this.loadTopicData(topicId);

    const document: Document = {
      id: this.generateId(),
      topicId,
      name,
      filePath,
      fileType,
      addedAt: Date.now(),
      chunkCount: chunks.length,
    };

    // Add document to topic data
    topicData.documents[document.id] = document;

    // Add chunks to topic data
    chunks.forEach((chunk) => {
      topicData.chunks[chunk.id] = chunk;
    });

    // Update topic in topicData and index
    topicData.topic.documentCount++;
    topicData.topic.updatedAt = Date.now();
    this.topicsIndex!.topics[topicId] = topicData.topic;

    // Update model name in both places now that we have embeddings
    const currentModel = this.embeddingService.getCurrentModel();
    if (currentModel) {
      topicData.modelName = currentModel;
      this.topicsIndex!.modelName = currentModel;
    }

    // Save both index and topic data
    await Promise.all([
      this.saveTopicsIndex(),
      this.saveTopicData(topicId, topicData)
    ]);

    return document;
  }

  /**
   * Search for similar chunks in a topic
   */
  public async search(
    topicId: string,
    queryEmbedding: number[],
    topK: number
  ): Promise<SearchResult[]> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }

    // Load topic data (only the topic we're searching)
    const topicData = await this.loadTopicData(topicId);

    // Get all chunks for this topic
    const topicChunks = Object.values(topicData.chunks);

    if (topicChunks.length === 0) {
      return [];
    }

    // Calculate similarities
    const results: SearchResult[] = topicChunks.map((chunk) => {
      const similarity = this.embeddingService.cosineSimilarity(
        queryEmbedding,
        chunk.embedding
      );

      return {
        chunk,
        similarity,
        documentName: chunk.metadata.documentName,
      };
    });

    // Sort by similarity (descending) and return top-k
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  /**
   * Get documents for a topic
   */
  public async getDocumentsByTopic(topicId: string): Promise<Document[]> {
    if (!this.topicsIndex) {
      await this.loadDatabase();
    }

    // Load topic data
    const topicData = await this.loadTopicData(topicId);
    return Object.values(topicData.documents);
  }

  /**
   * Get the database directory location
   */
  public getDatabaseLocation(): string {
    return this.getDatabaseDir();
  }

  /**
   * Generate a unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Clear entire database
   */
  public async clearDatabase(): Promise<void> {
    // Delete all topic files
    if (this.topicsIndex) {
      for (const topicId of Object.keys(this.topicsIndex.topics)) {
        try {
          const topicPath = this.getTopicDataPath(topicId);
          await fs.unlink(topicPath);
        } catch (error) {
          console.error(`Failed to delete topic file ${topicId}:`, error);
        }
      }
    }

    // Clear cache
    this.topicCache.clear();

    // Reset index
    this.topicsIndex = {
      topics: {},
      modelName: this.embeddingService.getCurrentModel() || 'unknown',
      lastUpdated: Date.now(),
    };
    await this.saveTopicsIndex();
  }
}

