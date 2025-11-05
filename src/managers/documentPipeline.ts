/**
 * Document Pipeline - End-to-end document processing orchestrator
 * Coordinates loading, splitting, embedding, and vector storage
 *
 * Architecture: Pipeline pattern with progress tracking
 * Integrates: DocumentLoaderFactory → SemanticChunker → EmbeddingService → VectorStoreFactory
 */

import * as vscode from 'vscode';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { VectorStore } from '@langchain/core/vectorstores';
import { DocumentLoaderFactory, LoaderOptions } from '../loaders/documentLoaderFactory';
import { SemanticChunker, ChunkingOptions } from '../splitters/semanticChunker';
import { EmbeddingService } from '../embeddingService';
import { VectorStoreFactory } from '../stores/vectorStoreFactory';
import { Logger } from '../utils/logger';
import { CONFIG } from '../constants';

export interface PipelineOptions {
  /** Document loading options */
  loaderOptions?: Partial<LoaderOptions>;

  /** Chunking options */
  chunkingOptions?: ChunkingOptions;

  /** Embedding model to use */
  embeddingModel?: string;

  /** Vector store type preference */
  vectorStoreType?: 'memory' | 'faiss';

  /** Batch size for embedding generation */
  embeddingBatchSize?: number;

  /** Progress callback */
  onProgress?: (progress: PipelineProgress) => void;
}

export interface PipelineProgress {
  stage: 'loading' | 'chunking' | 'embedding' | 'storing' | 'complete';
  progress: number; // 0-100
  message: string;
  details?: any;
}

export interface PipelineResult {
  /** Successfully processed documents */
  success: boolean;

  /** Processing stages completed */
  stages: {
    loading: boolean;
    chunking: boolean;
    embedding: boolean;
    storing: boolean;
  };

  /** Result metadata */
  metadata: {
    originalDocuments: number;
    chunksCreated: number;
    chunksEmbedded: number;
    chunksStored: number;
    totalTime: number;
    stageTimings: {
      loading: number;
      chunking: number;
      embedding: number;
      storing: number;
    };
  };

  /** Generated chunks */
  chunks: LangChainDocument[];

  /** Errors if any */
  errors?: string[];
}

/**
 * Orchestrates document processing pipeline
 */
export class DocumentPipeline {
  private logger: Logger;
  private documentLoader: DocumentLoaderFactory;
  private semanticChunker: SemanticChunker;
  private embeddingService: EmbeddingService;
  private vectorStoreFactory: VectorStoreFactory | null = null;

  constructor() {
    this.logger = new Logger('DocumentPipeline');
    this.documentLoader = new DocumentLoaderFactory();
    this.semanticChunker = new SemanticChunker();
    this.embeddingService = EmbeddingService.getInstance();

    this.logger.info('DocumentPipeline initialized');
  }

  /**
   * Initialize the pipeline with vector store factory
   */
  public async initialize(storageDir: string, embeddingModel?: string): Promise<void> {
    this.logger.info('Initializing pipeline', { storageDir, embeddingModel });

    try {
      // Initialize embedding service
      await this.embeddingService.initialize(embeddingModel);

      // Create vector store factory with a custom embeddings wrapper
      const embeddingsWrapper = {
        embedDocuments: async (texts: string[]) => {
          return await this.embeddingService.embedBatch(texts);
        },
        embedQuery: async (text: string) => {
          return await this.embeddingService.embed(text);
        },
      };

      this.vectorStoreFactory = new VectorStoreFactory(embeddingsWrapper as any, storageDir);

      this.logger.info('Pipeline initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize pipeline', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Process a single document through the entire pipeline
   */
  public async processDocument(
    filePath: string,
    topicId: string,
    options: PipelineOptions = {}
  ): Promise<PipelineResult> {
    return await this.processDocuments([filePath], topicId, options);
  }

  /**
   * Process multiple documents through the entire pipeline
   */
  public async processDocuments(
    filePaths: string[],
    topicId: string,
    options: PipelineOptions = {}
  ): Promise<PipelineResult> {
    const startTime = Date.now();
    const errors: string[] = [];

    this.logger.info('Starting document pipeline', {
      fileCount: filePaths.length,
      topicId,
      options,
    });

    const result: PipelineResult = {
      success: false,
      stages: {
        loading: false,
        chunking: false,
        embedding: false,
        storing: false,
      },
      metadata: {
        originalDocuments: filePaths.length,
        chunksCreated: 0,
        chunksEmbedded: 0,
        chunksStored: 0,
        totalTime: 0,
        stageTimings: {
          loading: 0,
          chunking: 0,
          embedding: 0,
          storing: 0,
        },
      },
      chunks: [],
      errors: [],
    };

    try {
      // Ensure initialized
      if (!this.vectorStoreFactory) {
        throw new Error('Pipeline not initialized. Call initialize() first.');
      }

      // Stage 1: Load documents
      const loadStartTime = Date.now();
      this.reportProgress(options.onProgress, {
        stage: 'loading',
        progress: 0,
        message: `Loading ${filePaths.length} document(s)...`,
      });

      const loadedDocs = await this.loadDocuments(filePaths, options);
      result.stages.loading = true;
      result.metadata.stageTimings.loading = Date.now() - loadStartTime;

      this.logger.info('Documents loaded', {
        count: loadedDocs.length,
        time: result.metadata.stageTimings.loading,
      });

      // Stage 2: Chunk documents
      const chunkStartTime = Date.now();
      this.reportProgress(options.onProgress, {
        stage: 'chunking',
        progress: 25,
        message: 'Chunking documents...',
      });

      const chunkingResult = await this.semanticChunker.chunkDocuments(
        loadedDocs,
        options.chunkingOptions
      );

      result.chunks = chunkingResult.chunks;
      result.metadata.chunksCreated = chunkingResult.chunkCount;
      result.stages.chunking = true;
      result.metadata.stageTimings.chunking = Date.now() - chunkStartTime;

      this.logger.info('Documents chunked', {
        chunkCount: chunkingResult.chunkCount,
        strategy: chunkingResult.strategy,
        time: result.metadata.stageTimings.chunking,
      });

      // Stage 3: Generate embeddings
      const embedStartTime = Date.now();
      this.reportProgress(options.onProgress, {
        stage: 'embedding',
        progress: 50,
        message: `Generating embeddings for ${result.chunks.length} chunks...`,
      });

      await this.generateEmbeddings(result.chunks, options);
      result.metadata.chunksEmbedded = result.chunks.length;
      result.stages.embedding = true;
      result.metadata.stageTimings.embedding = Date.now() - embedStartTime;

      this.logger.info('Embeddings generated', {
        chunkCount: result.metadata.chunksEmbedded,
        time: result.metadata.stageTimings.embedding,
      });

      // Stage 4: Store in vector database
      const storeStartTime = Date.now();
      this.reportProgress(options.onProgress, {
        stage: 'storing',
        progress: 75,
        message: 'Storing in vector database...',
      });

      await this.storeDocuments(result.chunks, topicId, options);
      result.metadata.chunksStored = result.chunks.length;
      result.stages.storing = true;
      result.metadata.stageTimings.storing = Date.now() - storeStartTime;

      this.logger.info('Documents stored', {
        chunkCount: result.metadata.chunksStored,
        time: result.metadata.stageTimings.storing,
      });

      // Complete
      result.success = true;
      result.metadata.totalTime = Date.now() - startTime;

      this.reportProgress(options.onProgress, {
        stage: 'complete',
        progress: 100,
        message: `Successfully processed ${filePaths.length} document(s)`,
        details: result.metadata,
      });

      this.logger.info('Pipeline completed successfully', {
        totalTime: result.metadata.totalTime,
        metadata: result.metadata,
      });

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      errors.push(errorMessage);

      this.logger.error('Pipeline failed', {
        error: errorMessage,
        stage: this.getCurrentStage(result.stages),
        metadata: result.metadata,
      });

      result.success = false;
      result.errors = errors;
      result.metadata.totalTime = Date.now() - startTime;

      return result;
    }
  }

  /**
   * Add documents to an existing vector store
   */
  public async addDocumentsToStore(
    vectorStore: VectorStore,
    filePaths: string[],
    options: PipelineOptions = {}
  ): Promise<PipelineResult> {
    const startTime = Date.now();

    this.logger.info('Adding documents to existing store', {
      fileCount: filePaths.length,
    });

    const result: PipelineResult = {
      success: false,
      stages: {
        loading: false,
        chunking: false,
        embedding: false,
        storing: false,
      },
      metadata: {
        originalDocuments: filePaths.length,
        chunksCreated: 0,
        chunksEmbedded: 0,
        chunksStored: 0,
        totalTime: 0,
        stageTimings: {
          loading: 0,
          chunking: 0,
          embedding: 0,
          storing: 0,
        },
      },
      chunks: [],
    };

    try {
      // Load documents
      const loadStartTime = Date.now();
      const loadedDocs = await this.loadDocuments(filePaths, options);
      result.stages.loading = true;
      result.metadata.stageTimings.loading = Date.now() - loadStartTime;

      // Chunk documents
      const chunkStartTime = Date.now();
      const chunkingResult = await this.semanticChunker.chunkDocuments(
        loadedDocs,
        options.chunkingOptions
      );
      result.chunks = chunkingResult.chunks;
      result.metadata.chunksCreated = chunkingResult.chunkCount;
      result.stages.chunking = true;
      result.metadata.stageTimings.chunking = Date.now() - chunkStartTime;

      // Add to vector store (embeddings generated automatically by LangChain)
      const storeStartTime = Date.now();
      await vectorStore.addDocuments(result.chunks);
      result.metadata.chunksStored = result.chunks.length;
      result.stages.embedding = true;
      result.stages.storing = true;
      result.metadata.stageTimings.storing = Date.now() - storeStartTime;

      result.success = true;
      result.metadata.totalTime = Date.now() - startTime;

      this.logger.info('Documents added to store successfully', {
        chunkCount: result.chunks.length,
        totalTime: result.metadata.totalTime,
      });

      return result;
    } catch (error) {
      this.logger.error('Failed to add documents to store', {
        error: error instanceof Error ? error.message : String(error),
      });

      result.success = false;
      result.errors = [error instanceof Error ? error.message : String(error)];
      result.metadata.totalTime = Date.now() - startTime;

      return result;
    }
  }

  // ==================== Private Methods ====================

  /**
   * Load documents using DocumentLoaderFactory
   */
  private async loadDocuments(
    filePaths: string[],
    options: PipelineOptions
  ): Promise<LangChainDocument[]> {
    const loaderOptions = filePaths.map((filePath) => ({
      filePath,
      ...options.loaderOptions,
    }));

    const results = await this.documentLoader.loadDocuments(loaderOptions);

    // Flatten all documents
    const allDocuments = results.flatMap((result) => result.documents);

    return allDocuments;
  }

  /**
   * Generate embeddings for chunks (Note: This is redundant with LangChain's built-in embedding)
   * Kept for explicit control and progress tracking
   */
  private async generateEmbeddings(
    chunks: LangChainDocument[],
    options: PipelineOptions
  ): Promise<void> {
    // With LangChain, embeddings are generated automatically when adding to vector store
    // This method is a placeholder for explicit embedding generation if needed

    // We can add explicit embedding here if we want to pre-generate and cache them
    const batchSize = options.embeddingBatchSize || 50;

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, Math.min(i + batchSize, chunks.length));
      const texts = batch.map((chunk) => chunk.pageContent);

      // Pre-generate embeddings (optional)
      await this.embeddingService.embedBatch(texts);

      // Report progress
      if (options.onProgress) {
        const progress = 50 + Math.floor((i / chunks.length) * 25);
        this.reportProgress(options.onProgress, {
          stage: 'embedding',
          progress,
          message: `Embedding batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(chunks.length / batchSize)}`,
        });
      }
    }
  }

  /**
   * Store documents in vector store
   */
  private async storeDocuments(
    chunks: LangChainDocument[],
    topicId: string,
    options: PipelineOptions
  ): Promise<void> {
    if (!this.vectorStoreFactory) {
      throw new Error('VectorStoreFactory not initialized');
    }

    // Try to load existing store or create new one
    let vectorStore = await this.vectorStoreFactory.loadStore(topicId);

    if (vectorStore) {
      // Add to existing store
      this.logger.debug('Adding to existing vector store', { topicId });
      await this.vectorStoreFactory.addDocuments(topicId, vectorStore, chunks);
    } else {
      // Create new store
      this.logger.debug('Creating new vector store', { topicId });

      const storeType = options.vectorStoreType ||
        this.vectorStoreFactory.getRecommendedStoreType(chunks.length);

      vectorStore = await this.vectorStoreFactory.createStore(
        {
          type: storeType,
          topicId,
          storageDir: '', // Already set in factory
        },
        chunks
      );
    }

    // Save the store
    await this.vectorStoreFactory.saveStore(topicId, vectorStore, {
      documentCount: 1, // This should be tracked properly
      chunkCount: chunks.length,
    });
  }

  /**
   * Report progress to callback
   */
  private reportProgress(
    callback: ((progress: PipelineProgress) => void) | undefined,
    progress: PipelineProgress
  ): void {
    if (callback) {
      callback(progress);
    }
  }

  /**
   * Get current stage name
   */
  private getCurrentStage(stages: PipelineResult['stages']): string {
    if (!stages.loading) return 'loading';
    if (!stages.chunking) return 'chunking';
    if (!stages.embedding) return 'embedding';
    if (!stages.storing) return 'storing';
    return 'complete';
  }
}
