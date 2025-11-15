/**
 * Embedding service using Transformers.js for local sentence transformers
 * Refactored with async-mutex to prevent race conditions
 *
 * Features:
 * - Local embedding generation using HuggingFace Transformers.js
 * - Multiple similarity metrics via LangChain (cosine, euclidean, inner product)
 * - Cross-platform WASM backend for ONNX models
 * - Batch processing with progress tracking
 *
 * Note: @huggingface/transformers is dynamically imported because it's an ESM-only package
 * and VS Code extensions run in CommonJS mode. Dynamic import() allows us to load ESM packages
 * at runtime without TypeScript compilation errors.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Mutex } from 'async-mutex';
import { cosineSimilarity as langchainCosineSimilarity, euclideanDistance, innerProduct } from '@langchain/core/utils/math';
import { CONFIG, DEFAULTS } from '../utils/constants';
import { Logger } from '../utils/logger';

// Type definitions for the dynamically imported transformers module
type TransformersModule = any; // Dynamic import type - resolved at runtime
type FeatureExtractionPipeline = any;

export class EmbeddingService {
  private static instance: EmbeddingService;
  private pipeline: FeatureExtractionPipeline | null = null;
  private currentModel: string = DEFAULTS.EMBEDDING_MODEL;
  private lastSuccessfulModel: string | null = null;
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
   * Resolve the configured local model path (if provided)
   */
  private resolveLocalModelPath(config: vscode.WorkspaceConfiguration): string | null {
    const configuredPath = (config.get<string>(CONFIG.LOCAL_MODEL_PATH, DEFAULTS.LOCAL_MODEL_PATH) ?? '').trim();
    if (!configuredPath) {
      return null;
    }

    // Support tilde expansion for convenience
    const expandedPath = configuredPath.replace(/^~(?=$|\/|\\)/, os.homedir());

    // Relative paths are resolved against the first workspace folder if available
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const normalizedPath = path.isAbsolute(expandedPath)
      ? expandedPath
      : path.resolve(workspaceFolder ?? process.cwd(), expandedPath);

    if (!fs.existsSync(normalizedPath)) {
      throw new Error(`Local embedding model path "${configuredPath}" does not exist (resolved to "${normalizedPath}")`);
    }

    return normalizedPath;
  }

  /**
   * Initialize the embedding model with mutex to prevent race conditions
   * @param modelName - Optional explicit embedding model name
   */
  public async initialize(modelName?: string): Promise<void> {
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    const configuredModel = config.get<string>(CONFIG.EMBEDDING_MODEL);
    let localModelPath: string | null = null;
    if (!modelName) {
      try {
        localModelPath = this.resolveLocalModelPath(config);
      } catch (error) {
        this.logger.error('Invalid local embedding model path configuration', error);
        throw error;
      }
    }

    // Priority:
    // 1. Explicit parameter (for programmatic control, tests)
    // 2. Explicit local model path from config (overrides remote identifier)
    // 3. User's remote config setting (respect user preference)
    // 4. Currently loaded model (avoid unnecessary reloads)
    // 5. Default model (fallback)
    const targetModel =
      modelName ??
      localModelPath ??
      configuredModel ??
      (this.pipeline ? this.currentModel : null) ??
      DEFAULTS.EMBEDDING_MODEL;

    try {
      await this.initializeModel(targetModel);
    } catch (error) {
      const isConfigDrivenAttempt =
        !modelName && (
          (localModelPath && localModelPath === targetModel) ||
          (!!configuredModel && configuredModel === targetModel)
        );

      if (isConfigDrivenAttempt) {
        const fallbackModel =
          this.lastSuccessfulModel ?? DEFAULTS.EMBEDDING_MODEL;

        if (fallbackModel && fallbackModel !== targetModel) {
          const fallbackReason = this.lastSuccessfulModel
            ? `previously downloaded model "${fallbackModel}"`
            : `default model "${fallbackModel}"`;
          const message = `RAGnarōk: Model "${targetModel}" could not be loaded. Falling back to ${fallbackReason}.`;
          this.logger.warn(message);
          vscode.window.showWarningMessage(message);

          await this.initializeModel(fallbackModel);
          return;
        }
      }

      throw error;
    }
  }

  private async initializeModel(targetModel: string): Promise<void> {
    // Already initialized with the same model
    if (this.pipeline && this.currentModel === targetModel) {
      this.logger.debug(`Model ${targetModel} already initialized`);
      return;
    }

    // Use mutex to prevent concurrent initializations
    await this.initMutex.runExclusive(async () => {
      // Double-check after acquiring lock
      if (this.pipeline && this.currentModel === targetModel) {
        this.logger.debug(`Model ${targetModel} initialized while waiting for lock`);
        return;
      }

      // Wait for existing initialization if in progress
      if (this.initPromise) {
        this.logger.debug('Waiting for existing initialization to complete');
        await this.initPromise;
        // Check if initialization succeeded and we have the right model
        if (this.pipeline && this.currentModel === targetModel) {
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

            progress.report({ message: 'Model loaded successfully!' });
          }
        );

        this.currentModel = modelName;
        this.lastSuccessfulModel = modelName;
        this.logger.info(`Embedding model initialized successfully: ${modelName}`);

        // Success - exit retry loop
        return;

      } catch (error: any) {
        lastError = error;
        this.logger.warn(`Initialization attempt ${attempt} failed:`, error.message);

        // Clean up failed state - keep currentModel unchanged during retries
        this.pipeline = null;

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
   * Truncate text to fit within model's token limit
   * Most models use ~256 word pieces, which is roughly 512-768 characters
   * We'll be conservative and truncate at 512 characters to be safe
   */
  private truncateText(text: string, maxChars: number = 512): string {
    if (text.length <= maxChars) {
      return text;
    }

    // Truncate and add ellipsis
    return text.substring(0, maxChars - 3) + '...';
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
      // Truncate text to fit model's limit (256 word pieces ~= 512 chars)
      const truncatedText = this.truncateText(text);

      // Generate embedding
      const output = await this.pipeline(truncatedText, {
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
      // Smaller batches but parallelized for much better performance
      const batchSize = 1000; // Process 1000 at a time in parallel

      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);

        // Log progress for large batches
        if (texts.length > 100) {
          const progressPercent = Math.round(((i + batch.length) / texts.length) * 100);
          this.logger.info(`Generating embeddings: ${i + batch.length}/${texts.length} (${progressPercent}%)`);
        }

        // Process batch in parallel for MUCH faster performance
        const batchPromises = batch.map(async (text) => {
          // Truncate text to fit model's limit (256 word pieces ~= 512 chars)
          const truncatedText = this.truncateText(text);

          const output = await this.pipeline!(truncatedText, {
            pooling: 'mean',
            normalize: true,
          });
          return Array.from((output as any).data) as number[];
        });

        const batchEmbeddings = await Promise.all(batchPromises);
        embeddings.push(...batchEmbeddings);

        // Report progress
        if (progressCallback) {
          progressCallback(embeddings.length / texts.length);
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
   * Calculate cosine similarity between two embeddings using LangChain's implementation
   */
  public cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    // LangChain's cosineSimilarity works on matrices, so wrap vectors in arrays
    const result = langchainCosineSimilarity([a], [b]);
    return result[0][0];
  }

  /**
   * Calculate Euclidean distance between two embeddings using LangChain's implementation
   */
  public euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    // LangChain's euclideanDistance works on matrices, so wrap vectors in arrays
    const result = euclideanDistance([a], [b]);
    return result[0][0];
  }

  /**
   * Calculate inner product (dot product) between two embeddings using LangChain's implementation
   */
  public innerProduct(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    // LangChain's innerProduct works on matrices, so wrap vectors in arrays
    const result = innerProduct([a], [b]);
    return result[0][0];
  }

  /**
   * Get the current model name
   * Returns the name of the model that is currently loaded or will be loaded on next initialization.
   * Returns the default model name if no specific model has been set.
   */
  public getCurrentModel(): string {
    return this.currentModel;
  }

  /**
   * Clear the model cache and reset
   */
  public async clearCache(): Promise<void> {
    const previousModel = this.currentModel;
    this.logger.info('Clearing embedding model cache', { previousModel });

    this.pipeline = null;
    // Reset to default model name so next initialization knows what to use
    this.currentModel = DEFAULTS.EMBEDDING_MODEL;
    this.lastSuccessfulModel = null;

    this.logger.info('Embedding model cache cleared successfully');
    vscode.window.showInformationMessage('Embedding model cache cleared. Model will reload on next use.');
  }

  /**
   * Dispose of all resources and clean up
   * Should be called when the service is no longer needed
   */
  public dispose(): void {
    this.logger.info('Disposing EmbeddingService');

    // Clear pipeline
    this.pipeline = null;
    // Reset to default model name for consistency
    this.currentModel = DEFAULTS.EMBEDDING_MODEL;
    this.lastSuccessfulModel = null;

    // Clear transformers module reference
    this.transformers = null;

    // Clear initialization promise
    this.initPromise = null;

    this.logger.info('EmbeddingService disposed');
  }
}
