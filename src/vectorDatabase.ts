/**
 * Vector database implementation using VSCode Secret Storage
 */

import * as vscode from 'vscode';
import { VectorDatabase, Topic, Document, TextChunk, SearchResult } from './types';
import { EmbeddingService } from './embeddingService';
import { EXTENSION, LIMITS } from './constants';

export class VectorDatabaseService {
  private static instance: VectorDatabaseService;
  private context: vscode.ExtensionContext;
  private database: VectorDatabase | null = null;
  private embeddingService: EmbeddingService;

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
   * Load database from secret storage
   */
  public async loadDatabase(): Promise<void> {
    try {
      const data = await this.context.secrets.get(EXTENSION.SECRET_NAME);
      if (data) {
        this.database = JSON.parse(data);
      } else {
        // Initialize empty database
        this.database = {
          topics: {},
          documents: {},
          chunks: {},
          modelName: this.embeddingService.getCurrentModel() || 'unknown',
          lastUpdated: Date.now(),
        };
        await this.saveDatabase();
      }
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to load database: ${error}`);
      throw error;
    }
  }

  /**
   * Save database to secret storage
   */
  public async saveDatabase(): Promise<void> {
    if (!this.database) {
      throw new Error('Database not initialized');
    }

    try {
      this.database.lastUpdated = Date.now();
      const data = JSON.stringify(this.database);

      // VSCode secrets have size limits, check if we're within limits
      if (data.length > LIMITS.MAX_DATABASE_SIZE) {
        vscode.window.showWarningMessage('Database is getting large. Consider removing old topics.');
      }

      await this.context.secrets.store(EXTENSION.SECRET_NAME, data);
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to save database: ${error}`);
      throw error;
    }
  }

  /**
   * Create a new topic
   */
  public async createTopic(name: string, description?: string): Promise<Topic> {
    if (!this.database) {
      await this.loadDatabase();
    }

    // Check if topic already exists
    const existingTopic = Object.values(this.database!.topics).find(
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

    this.database!.topics[topic.id] = topic;
    await this.saveDatabase();

    return topic;
  }

  /**
   * Get all topics
   */
  public async getTopics(): Promise<Topic[]> {
    if (!this.database) {
      await this.loadDatabase();
    }
    return Object.values(this.database!.topics);
  }

  /**
   * Get a topic by name
   */
  public async getTopicByName(name: string): Promise<Topic | null> {
    if (!this.database) {
      await this.loadDatabase();
    }

    const topic = Object.values(this.database!.topics).find(
      (t) => t.name.toLowerCase() === name.toLowerCase()
    );

    return topic || null;
  }

  /**
   * Delete a topic and all its documents and chunks
   */
  public async deleteTopic(topicId: string): Promise<void> {
    if (!this.database) {
      await this.loadDatabase();
    }

    // Delete all chunks associated with this topic
    Object.keys(this.database!.chunks).forEach((chunkId) => {
      if (this.database!.chunks[chunkId].topicId === topicId) {
        delete this.database!.chunks[chunkId];
      }
    });

    // Delete all documents associated with this topic
    Object.keys(this.database!.documents).forEach((docId) => {
      if (this.database!.documents[docId].topicId === topicId) {
        delete this.database!.documents[docId];
      }
    });

    // Delete the topic
    delete this.database!.topics[topicId];

    await this.saveDatabase();
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
    if (!this.database) {
      await this.loadDatabase();
    }

    if (!this.database!.topics[topicId]) {
      throw new Error('Topic not found');
    }

    const document: Document = {
      id: this.generateId(),
      topicId,
      name,
      filePath,
      fileType,
      addedAt: Date.now(),
      chunkCount: chunks.length,
    };

    // Add document
    this.database!.documents[document.id] = document;

    // Add chunks
    chunks.forEach((chunk) => {
      this.database!.chunks[chunk.id] = chunk;
    });

    // Update topic
    this.database!.topics[topicId].documentCount++;
    this.database!.topics[topicId].updatedAt = Date.now();

    await this.saveDatabase();

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
    if (!this.database) {
      await this.loadDatabase();
    }

    // Get all chunks for this topic
    const topicChunks = Object.values(this.database!.chunks).filter(
      (chunk) => chunk.topicId === topicId
    );

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
    if (!this.database) {
      await this.loadDatabase();
    }

    return Object.values(this.database!.documents).filter(
      (doc) => doc.topicId === topicId
    );
  }

  /**
   * Get database statistics
   */
  public async getStats(): Promise<{
    topicCount: number;
    documentCount: number;
    chunkCount: number;
    modelName: string;
    lastUpdated: number;
  }> {
    if (!this.database) {
      await this.loadDatabase();
    }

    return {
      topicCount: Object.keys(this.database!.topics).length,
      documentCount: Object.keys(this.database!.documents).length,
      chunkCount: Object.keys(this.database!.chunks).length,
      modelName: this.database!.modelName,
      lastUpdated: this.database!.lastUpdated,
    };
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
    this.database = {
      topics: {},
      documents: {},
      chunks: {},
      modelName: this.embeddingService.getCurrentModel() || 'unknown',
      lastUpdated: Date.now(),
    };
    await this.saveDatabase();
  }
}

