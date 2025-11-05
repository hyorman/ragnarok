/**
 * Vector Store Factory - Creates and manages LangChain vector stores
 * Supports both MemoryVectorStore (default) and FaissStore (optional)
 *
 * Architecture: Hybrid approach with graceful fallback
 * - Primary: MemoryVectorStore (always available)
 * - Optional: FaissStore (for better performance with large datasets)
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';
import { VectorStore } from '@langchain/core/vectorstores';
import { Embeddings } from '@langchain/core/embeddings';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { Logger } from '../utils/logger';

// Try to import FAISS (optional dependency)
// FAISS provides better performance for large datasets but requires native compilation
// If unavailable, we gracefully fall back to MemoryVectorStore (which works perfectly)
let FaissStore: any = null;
let faissAvailable = false;
let faissError: string | null = null;

try {
  // Dynamic import to handle optional dependency
  const faissModule = require('@langchain/community/vectorstores/faiss');
  FaissStore = faissModule.FaissStore;
  faissAvailable = true;
  console.log('✓ FAISS vector store available - will use for better performance with large datasets');
} catch (error) {
  // FAISS not available - this is completely fine, MemoryVectorStore works great
  faissAvailable = false;
  faissError = error instanceof Error ? error.message : String(error);
  console.log('ℹ️  FAISS not available - using MemoryVectorStore (this is perfectly fine for most use cases)');
  if (faissError.includes('Cannot find module')) {
    console.log('   Reason: faiss-node is an optional dependency that requires native compilation');
    console.log('   Impact: None - MemoryVectorStore provides excellent performance for typical workloads');
  }
}

export type VectorStoreType = 'memory' | 'faiss';

export interface VectorStoreConfig {
  type: VectorStoreType;
  topicId: string;
  storageDir: string;
}

export interface VectorStoreMetadata {
  type: VectorStoreType;
  topicId: string;
  documentCount: number;
  chunkCount: number;
  createdAt: number;
  updatedAt: number;
}

/**
 * Factory for creating and managing LangChain vector stores
 */
export class VectorStoreFactory {
  private logger: Logger;
  private embeddings: Embeddings;
  private storageDir: string;
  private storeCache: Map<string, VectorStore> = new Map();

  constructor(embeddings: Embeddings, storageDir: string) {
    this.logger = new Logger('VectorStoreFactory');
    this.embeddings = embeddings;
    this.storageDir = storageDir;

    this.logger.info('VectorStoreFactory initialized', {
      faissAvailable,
      storageDir,
    });
  }

  /**
   * Check if FAISS is available
   */
  public static isFaissAvailable(): boolean {
    return faissAvailable;
  }

  /**
   * Get information about FAISS availability
   */
  public static getFaissStatus(): { available: boolean; error?: string } {
    return {
      available: faissAvailable,
      error: faissError || undefined
    };
  }

  /**
   * Get the recommended store type based on availability and dataset size
   */
  public getRecommendedStoreType(expectedChunkCount?: number): VectorStoreType {
    // Use FAISS for larger datasets if available
    if (faissAvailable && expectedChunkCount && expectedChunkCount > 1000) {
      this.logger.debug('Recommending FAISS for large dataset', { expectedChunkCount });
      return 'faiss';
    }

    // Default to memory store
    this.logger.debug('Recommending MemoryVectorStore', {
      faissAvailable,
      expectedChunkCount
    });
    return 'memory';
  }

  /**
   * Create a new vector store
   */
  public async createStore(
    config: VectorStoreConfig,
    initialDocuments?: LangChainDocument[]
  ): Promise<VectorStore> {
    this.logger.info('Creating vector store', {
      type: config.type,
      topicId: config.topicId,
      documentCount: initialDocuments?.length || 0
    });

    let store: VectorStore;

    try {
      if (config.type === 'faiss' && faissAvailable) {
        // Create FAISS store
        store = await this.createFaissStore(config, initialDocuments);
      } else {
        // Create or fallback to Memory store
        if (config.type === 'faiss' && !faissAvailable) {
          this.logger.warn('FAISS requested but not available, falling back to MemoryVectorStore');
          this.logger.info('MemoryVectorStore provides excellent performance for most use cases');
          this.logger.info('To enable FAISS: Ensure native compilation tools are available, then reinstall');
        }
        store = await this.createMemoryStore(config, initialDocuments) as any as VectorStore;
      }

      // Cache the store
      this.storeCache.set(config.topicId, store);

      this.logger.info('Vector store created successfully', {
        type: config.type,
        topicId: config.topicId
      });

      return store;
    } catch (error) {
      this.logger.error('Failed to create vector store', {
        error: error instanceof Error ? error.message : String(error),
        config
      });
      throw error;
    }
  }

  /**
   * Load an existing vector store
   */
  public async loadStore(
    topicId: string,
    preferredType?: VectorStoreType
  ): Promise<VectorStore | null> {
    this.logger.info('Loading vector store', { topicId, preferredType });

    // Check cache first
    const cachedStore = this.storeCache.get(topicId);
    if (cachedStore) {
      this.logger.debug('Returning cached store', { topicId });
      return cachedStore;
    }

    try {
      // Try to load from disk
      const storePath = this.getStorePath(topicId);
      const metadataPath = this.getMetadataPath(topicId);

      // Check if store exists
      try {
        await fs.access(metadataPath);
      } catch {
        this.logger.debug('Store not found', { topicId });
        return null;
      }

      // Load metadata
      const metadataJson = await fs.readFile(metadataPath, 'utf-8');
      const metadata: VectorStoreMetadata = JSON.parse(metadataJson);

      let store: VectorStore;

      // Load based on type (with fallback)
      if (metadata.type === 'faiss' && faissAvailable) {
        store = await this.loadFaissStore(topicId);
      } else {
        if (metadata.type === 'faiss' && !faissAvailable) {
          this.logger.warn('FAISS store found but FAISS not available, falling back to MemoryVectorStore');
        }
        store = await this.loadMemoryStore(topicId) as any as VectorStore;
      }

      // Cache the loaded store
      this.storeCache.set(topicId, store);

      this.logger.info('Vector store loaded successfully', {
        topicId,
        type: metadata.type
      });

      return store;
    } catch (error) {
      this.logger.error('Failed to load vector store', {
        error: error instanceof Error ? error.message : String(error),
        topicId
      });
      return null;
    }
  }

  /**
   * Save a vector store to disk
   */
  public async saveStore(
    topicId: string,
    store: VectorStore,
    metadata: Partial<VectorStoreMetadata>
  ): Promise<void> {
    this.logger.info('Saving vector store', { topicId });

    try {
      const storePath = this.getStorePath(topicId);
      const metadataPath = this.getMetadataPath(topicId);

      // Ensure directory exists
      await fs.mkdir(path.dirname(storePath), { recursive: true });

      // Determine store type
      const storeType = FaissStore && store instanceof FaissStore ? 'faiss' : 'memory';

      // Save the store
      if (storeType === 'faiss' && FaissStore && store instanceof FaissStore) {
        // FAISS stores have a save method
        await (store as any).save(storePath);
      } else if (store instanceof MemoryVectorStore) {
        // MemoryVectorStore doesn't have native persistence, we'll serialize manually
        const vectors = (store as any).memoryVectors || [];
        await fs.writeFile(
          storePath,
          JSON.stringify({ vectors }, null, 2),
          'utf-8'
        );
      }

      // Save metadata
      const fullMetadata: VectorStoreMetadata = {
        type: storeType,
        topicId,
        documentCount: metadata.documentCount || 0,
        chunkCount: metadata.chunkCount || 0,
        createdAt: metadata.createdAt || Date.now(),
        updatedAt: Date.now(),
      };

      await fs.writeFile(
        metadataPath,
        JSON.stringify(fullMetadata, null, 2),
        'utf-8'
      );

      this.logger.info('Vector store saved successfully', {
        topicId,
        type: storeType
      });
    } catch (error) {
      this.logger.error('Failed to save vector store', {
        error: error instanceof Error ? error.message : String(error),
        topicId
      });
      throw error;
    }
  }

  /**
   * Delete a vector store
   */
  public async deleteStore(topicId: string): Promise<void> {
    this.logger.info('Deleting vector store', { topicId });

    try {
      // Remove from cache
      this.storeCache.delete(topicId);

      // Delete files
      const storePath = this.getStorePath(topicId);
      const metadataPath = this.getMetadataPath(topicId);

      try {
        await fs.unlink(storePath);
      } catch {
        // File might not exist
      }

      try {
        await fs.unlink(metadataPath);
      } catch {
        // File might not exist
      }

      this.logger.info('Vector store deleted successfully', { topicId });
    } catch (error) {
      this.logger.error('Failed to delete vector store', {
        error: error instanceof Error ? error.message : String(error),
        topicId
      });
      throw error;
    }
  }

  /**
   * Add documents to an existing store
   */
  public async addDocuments(
    topicId: string,
    store: VectorStore,
    documents: LangChainDocument[]
  ): Promise<void> {
    this.logger.info('Adding documents to vector store', {
      topicId,
      documentCount: documents.length
    });

    try {
      await store.addDocuments(documents);

      this.logger.info('Documents added successfully', {
        topicId,
        documentCount: documents.length
      });
    } catch (error) {
      this.logger.error('Failed to add documents', {
        error: error instanceof Error ? error.message : String(error),
        topicId,
        documentCount: documents.length
      });
      throw error;
    }
  }

  /**
   * Clear the store cache
   */
  public clearCache(topicId?: string): void {
    if (topicId) {
      this.storeCache.delete(topicId);
      this.logger.debug('Cache cleared for topic', { topicId });
    } else {
      this.storeCache.clear();
      this.logger.debug('All store cache cleared');
    }
  }

  // ==================== Private Methods ====================

  /**
   * Create a Memory vector store
   */
  private async createMemoryStore(
    config: VectorStoreConfig,
    initialDocuments?: LangChainDocument[]
  ): Promise<MemoryVectorStore> {
    this.logger.debug('Creating MemoryVectorStore', { topicId: config.topicId });

    if (initialDocuments && initialDocuments.length > 0) {
      return await MemoryVectorStore.fromDocuments(
        initialDocuments,
        this.embeddings
      );
    } else {
      return new MemoryVectorStore(this.embeddings);
    }
  }

  /**
   * Create a FAISS vector store
   */
  private async createFaissStore(
    config: VectorStoreConfig,
    initialDocuments?: LangChainDocument[]
  ): Promise<any> {
    if (!FaissStore) {
      throw new Error('FAISS is not available');
    }

    this.logger.debug('Creating FaissStore', { topicId: config.topicId });

    if (initialDocuments && initialDocuments.length > 0) {
      return await FaissStore.fromDocuments(
        initialDocuments,
        this.embeddings
      );
    } else {
      // FAISS requires at least one document to initialize
      // Create with a dummy document that will be removed
      const dummyDoc = new LangChainDocument({
        pageContent: '_dummy_initialization_document_',
        metadata: { isDummy: true },
      });

      const store = await FaissStore.fromDocuments(
        [dummyDoc],
        this.embeddings
      );

      // Note: We can't easily remove the dummy doc, but it won't affect search
      // as it has very different content from real documents

      return store;
    }
  }

  /**
   * Load a Memory vector store from disk
   */
  private async loadMemoryStore(topicId: string): Promise<MemoryVectorStore> {
    this.logger.debug('Loading MemoryVectorStore', { topicId });

    const storePath = this.getStorePath(topicId);

    try {
      const data = await fs.readFile(storePath, 'utf-8');
      const { vectors } = JSON.parse(data);

      // Reconstruct the store from serialized data
      const store = new MemoryVectorStore(this.embeddings);

      // Manually populate the store
      // Note: This is a workaround since MemoryVectorStore doesn't have native persistence
      if (vectors && vectors.length > 0) {
        const documents = vectors.map((v: any) => new LangChainDocument({
          pageContent: v.content,
          metadata: v.metadata,
        }));

        await store.addDocuments(documents);
      }

      return store;
    } catch (error) {
      this.logger.warn('Failed to load MemoryVectorStore, creating new one', {
        error: error instanceof Error ? error.message : String(error),
        topicId
      });
      return new MemoryVectorStore(this.embeddings);
    }
  }

  /**
   * Load a FAISS vector store from disk
   */
  private async loadFaissStore(topicId: string): Promise<any> {
    if (!FaissStore) {
      throw new Error('FAISS is not available');
    }

    this.logger.debug('Loading FaissStore', { topicId });

    const storePath = this.getStorePath(topicId);
    return await FaissStore.load(storePath, this.embeddings);
  }

  /**
   * Get the file path for storing vector data
   */
  private getStorePath(topicId: string): string {
    return path.join(this.storageDir, `vector-${topicId}`);
  }

  /**
   * Get the file path for storing metadata
   */
  private getMetadataPath(topicId: string): string {
    return path.join(this.storageDir, `vector-${topicId}-metadata.json`);
  }
}
