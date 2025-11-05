/**
 * Embedding service using Transformers.js for local sentence transformers
 * Refactored with async-mutex to prevent race conditions
 *
 * Note: @huggingface/transformers is dynamically imported because it's an ESM-only package
 * and VS Code extensions run in CommonJS mode. Dynamic import() allows us to load ESM packages
 * at runtime without TypeScript compilation errors.
 */

import * as vscode from 'vscode';
import { Mutex } from 'async-mutex';
import { CONFIG } from './constants';
import { Logger } from './utils/logger';

// Type definitions for the dynamically imported transformers module
type TransformersModule = any; // Dynamic import type - resolved at runtime
type FeatureExtractionPipeline = any;

export class EmbeddingService {
  private static instance: EmbeddingService;
  private pipeline: FeatureExtractionPipeline | null = null;
  private currentModel: string | null = null;
  private initMutex: Mutex = new Mutex();
  private initPromise: Promise<void> | null = null;
  private logger: Logger;
  private transformers: TransformersModule | null = null;

  private constructor() {
    this.logger = new Logger('EmbeddingService');
  }

  /**
   * Dynamically import and configure the transformers module
   * This is done lazily to avoid CommonJS/ESM compatibility issues
   */
  private async loadTransformers(): Promise<TransformersModule> {
    if (this.transformers) {
      return this.transformers;
    }

    this.transformers = await import('@huggingface/transformers');
    const { env } = this.transformers;

    // Configure transformers.js environment for cross-platform compatibility
    env.allowLocalModels = true;
    env.allowRemoteModels = true; // Allow downloading models from HuggingFace
    env.useBrowserCache = false;

    // Use WebAssembly backend for ONNX (cross-platform ML inference)
    env.backends = {
      onnx: {
        wasm: {
          proxy: false,
          numThreads: 1, // Single thread for stability
        }
      }
    };

    this.logger.info('EmbeddingService configured: WASM backend (ONNX), Sharp enabled (image processing)');
    return this.transformers;
  }

  public static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  /**
   * Initialize the embedding model with mutex to prevent race conditions
   */
  public async initialize(modelName?: string): Promise<void> {
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    const targetModel = modelName || config.get<string>(CONFIG.EMBEDDING_MODEL, 'Xenova/all-MiniLM-L6-v2');

    // Already initialized with the same model
    if (this.pipeline && this.currentModel === targetModel) {
      this.logger.debug(`Model ${targetModel} already initialized`);
      return;
    }

    // Use mutex to prevent concurrent initializations
    return this.initMutex.runExclusive(async () => {
      // Double-check after acquiring lock
      if (this.pipeline && this.currentModel === targetModel) {
        this.logger.debug(`Model ${targetModel} initialized while waiting for lock`);
        return;
      }

      // Wait for existing initialization if in progress
      if (this.initPromise) {
        this.logger.debug('Waiting for existing initialization to complete');
        await this.initPromise;
        if (this.currentModel === targetModel) {
          return;
        }
      }

      // Start new initialization
      this.logger.info(`Initializing embedding model: ${targetModel}`);
      this.initPromise = this._initializePipeline(targetModel);

      try {
        await this.initPromise;
        this.logger.info(`Successfully initialized model: ${targetModel}`);
      } catch (error) {
        this.logger.error(`Failed to initialize model: ${targetModel}`, error);
        throw error;
      } finally {
        this.initPromise = null;
      }
    });
  }

  private async _initializePipeline(modelName: string): Promise<void> {
    const maxRetries = 3;
    let lastError: Error | null = null;

    // Load transformers module
    const transformers = await this.loadTransformers();
    const { pipeline } = transformers;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: `Loading embedding model: ${modelName}${attempt > 1 ? ` (Attempt ${attempt}/${maxRetries})` : ''}`,
            cancellable: false,
          },
          async (progress) => {
            progress.report({ message: 'Downloading and initializing...' });

            // Create the feature extraction pipeline
            this.pipeline = await pipeline('feature-extraction', modelName, {
              progress_callback: (progressData: any) => {
                if (progressData.status === 'progress' && progressData.progress) {
                  const percent = Math.round(progressData.progress);
                  progress.report({
                    message: `${progressData.file || 'Model'}: ${percent}%`,
                    increment: 1
                  });
                }
              }
            });

            // Validate the pipeline by testing with dummy text
            await this.pipeline('test', { pooling: 'mean', normalize: true });

            this.currentModel = modelName;
            progress.report({ message: 'Model loaded successfully!' });
          }
        );

        // Success - exit retry loop
        return;

      } catch (error: any) {
        lastError = error;
        this.logger.warn(`Initialization attempt ${attempt} failed:`, error.message);

        // Clean up failed state
        this.pipeline = null;
        this.currentModel = null;

        // Don't retry on last attempt
        if (attempt < maxRetries) {
          // Wait before retry with exponential backoff
          const backoffMs = 1000 * Math.pow(2, attempt - 1);
          this.logger.debug(`Waiting ${backoffMs}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, backoffMs));
        }
      }
    }

    // All retries failed
    this.logger.error('All initialization attempts failed', lastError);
    const errorMsg = lastError?.message || String(lastError);
    throw new Error(`Failed to initialize embedding model "${modelName}" after ${maxRetries} attempts: ${errorMsg}`);
  }

  /**
   * Generate embeddings for a single text
   */
  public async embed(text: string): Promise<number[]> {
    if (!this.pipeline) {
      await this.initialize();
    }

    if (!this.pipeline) {
      throw new Error('Embedding pipeline not initialized');
    }

    try {
      // Generate embedding
      const output = await this.pipeline(text, {
        pooling: 'mean',
        normalize: true,
      });

      // Convert to regular array
      const embedding = Array.from((output as any).data) as number[];
      this.logger.debug(`Generated embedding with dimension: ${embedding.length}`);
      return embedding;
    } catch (error) {
      this.logger.error('Failed to generate embedding', error);
      throw new Error(`Failed to generate embedding: ${error}`);
    }
  }

  /**
   * Generate embeddings for multiple texts in batch with progress reporting
   */
  public async embedBatch(
    texts: string[],
    progressCallback?: (progress: number) => void
  ): Promise<number[][]> {
    if (!this.pipeline) {
      await this.initialize();
    }

    if (!this.pipeline) {
      throw new Error('Embedding pipeline not initialized');
    }

    this.logger.debug(`Generating embeddings for ${texts.length} texts`);

    try {
      const embeddings: number[][] = [];

      // Process in batches to avoid memory issues
      const batchSize = 10;
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);

        for (const text of batch) {
          const output = await this.pipeline(text, {
            pooling: 'mean',
            normalize: true,
          });
          embeddings.push(Array.from((output as any).data) as number[]);

          // Report progress
          if (progressCallback) {
            progressCallback(embeddings.length / texts.length);
          }
        }

        // Allow garbage collection between batches
        if (i + batchSize < texts.length) {
          await new Promise(resolve => setImmediate(resolve));
        }
      }

      this.logger.debug(`Successfully generated ${embeddings.length} embeddings`);
      return embeddings;
    } catch (error) {
      this.logger.error('Failed to generate batch embeddings', error);
      throw new Error(`Failed to generate batch embeddings: ${error}`);
    }
  }

  /**
   * Calculate cosine similarity between two embeddings
   */
  public cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return similarity;
  }

  /**
   * Get the current model name
   */
  public getCurrentModel(): string | null {
    return this.currentModel;
  }

  /**
   * Clear the model cache and reset
   */
  public async clearCache(): Promise<void> {
    const previousModel = this.currentModel;
    this.logger.info('Clearing embedding model cache', { previousModel });

    this.pipeline = null;
    this.currentModel = null;

    this.logger.info('Embedding model cache cleared successfully');
    vscode.window.showInformationMessage('Embedding model cache cleared. Model will reload on next use.');
  }
}

