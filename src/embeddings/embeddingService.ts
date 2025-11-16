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
// Alias LangChain math helpers to avoid name collisions with class methods
import { cosineSimilarity as langchainCosineSimilarity, euclideanDistance as langchainEuclideanDistance, innerProduct as langchainInnerProduct } from '@langchain/core/utils/math';
import { CONFIG, DEFAULTS } from '../utils/constants';
import { Logger } from '../utils/logger';

// Type definitions for the dynamically imported transformers module
type TransformersModule = any; // Dynamic import type - resolved at runtime
type FeatureExtractionPipeline = any;
export type AvailableModel = {
  name: string;
  source: 'curated' | 'local';
  downloaded?: boolean;
};

export class EmbeddingService {
  private static instance: EmbeddingService;
  private static readonly CURATED_MODELS = [
    'Xenova/all-MiniLM-L6-v2',
    'Xenova/all-MiniLM-L12-v2',
    'Xenova/paraphrase-MiniLM-L6-v2',
    'Xenova/multi-qa-MiniLM-L6-cos-v1',
  ];
  private static readonly DEFAULT_MODEL = EmbeddingService.CURATED_MODELS[0];
  private pipeline: FeatureExtractionPipeline | null = null;
  private currentModel: string = EmbeddingService.DEFAULT_MODEL;
  private lastSuccessfulModel: string | null = null;
  private initMutex: Mutex = new Mutex();
  private initPromise: Promise<void> | null = null;
  private logger: Logger;
  private transformers: TransformersModule | null = null;
  // Cache resolved local model path to avoid repeated resolution
  private resolvedLocalModelPath: string | null = null;

  private constructor() {
    this.logger = new Logger('EmbeddingService');
  }

  /**
   * Get the currently configured local model base path, if any.
   * Returns null when the config is unset or invalid.
   */
  public getLocalModelPath(): string | null {
    return this.resolvedLocalModelPath;
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
        wasm: { proxy: false, numThreads: 2 }, // Increased to 2 threads for better performance
      }
    };

    // If the extension has a configured local model base path, set it so
    // transformers.js can resolve short model names (e.g. 'my-model' -> `${localModelPath}/my-model`).
    try {
      const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
      const resolved = this.resolveLocalModelPath(config);
      if (resolved) {
        // transformers.env.localModelPath is used by the runtime when given short model names
        env.localModelPath = resolved;
        this.resolvedLocalModelPath = resolved;
        this.logger.info(`Transformers env.localModelPath set to ${resolved}`);
      }
    } catch (err: any) {
      // Don't fail initialization just because the configured path is invalid; log and continue
      this.logger.warn('Could not set transformers env.localModelPath from config:', err?.message ?? err);
    }

    this.logger.info('EmbeddingService configured: WASM backend (ONNX), Sharp enabled (image processing)');
    return this.transformers;
  }

  /**
   * List local models inside the configured local model path.
   * Returns an array of model identifiers (directory names). If the configured
   * path is empty or invalid, returns an empty array.
   */
  public async listLocalModels(): Promise<string[]> {
    try {
      const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
      const resolved = this.resolveLocalModelPath(config);
      // Cache resolved path for other callers
      this.resolvedLocalModelPath = resolved;

      if (!resolved) return [];

      const entries = await fs.promises.readdir(resolved, { withFileTypes: true });
      // Consider directories as individual models; allow files that look like model files too
      const models = entries
        .filter((e) => e.isDirectory())
        .map((d) => d.name)
        .sort((a, b) => a.localeCompare(b));

      return models;
    } catch (err: any) {
      this.logger.warn('Failed to list local models', err?.message ?? err);
      return [];
    }
  }

  /**
   * List remote models that have already been downloaded to the transformers cache directory.
   */
  private async listDownloadedRemoteModels(): Promise<string[]> {
    try {
      const transformers = await this.loadTransformers();
      const cacheDir = transformers?.env?.cacheDir;
      if (!cacheDir) return [];

      const normalizedCacheDir = path.isAbsolute(cacheDir)
        ? cacheDir
        : path.resolve(cacheDir);

      if (!fs.existsSync(normalizedCacheDir)) {
        this.logger.debug(`Transformers cache directory not found at ${normalizedCacheDir}`);
        return [];
      }

      const ownerEntries = await fs.promises.readdir(normalizedCacheDir, { withFileTypes: true });
      const downloadedModels: string[] = [];

      for (const owner of ownerEntries) {
        if (!owner.isDirectory()) continue;

        const ownerPath = path.join(normalizedCacheDir, owner.name);
        try {
          const modelEntries = await fs.promises.readdir(ownerPath, { withFileTypes: true });
          for (const model of modelEntries) {
            if (model.isDirectory()) {
              downloadedModels.push(`${owner.name}/${model.name}`);
            }
          }
        } catch (err: any) {
          // Skip folders we cannot read but keep processing the rest
          this.logger.debug(`Unable to inspect cached models under ${ownerPath}`, err?.message ?? err);
        }
      }

      return downloadedModels.sort((a, b) => a.localeCompare(b));
    } catch (err: any) {
      this.logger.warn('Failed to list downloaded remote models', err?.message ?? err);
      return [];
    }
  }

  /**
   * Combine configured local models with any remote models already downloaded to disk.
   */
  public async listAvailableModels(): Promise<AvailableModel[]> {
    const available: AvailableModel[] = [];
    const downloaded = new Set(await this.listDownloadedRemoteModels());

    for (const name of EmbeddingService.CURATED_MODELS) {
      available.push({
        name,
        source: 'curated',
        downloaded: downloaded.has(name),
      });
    }

    const localModels = await this.listLocalModels();
    for (const name of localModels) {
      available.push({ name, source: 'local', downloaded: true });
    }

    return available;
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

    // Priority:
    // 1. Explicit parameter (for programmatic control, tests)
    // 2. Currently loaded model (avoid unnecessary reloads)
    // 3. Default model (fallback)
    const targetModel =
      modelName ??
      (this.pipeline ? this.currentModel : null) ??
      EmbeddingService.DEFAULT_MODEL;

    try {
      await this.initializeModel(targetModel);
    } catch (error) {
      const isConfigDrivenAttempt = !modelName;

      if (isConfigDrivenAttempt) {
        const fallbackModel = this.lastSuccessfulModel ?? EmbeddingService.DEFAULT_MODEL;

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

        // If curated download failed and there are local models available, try them all sequentially
        const localModels = await this.listLocalModels();
        if (localModels.length > 0) {
          this.logger.warn(`Remote model "${targetModel}" could not be loaded. Attempting local models: ${localModels.join(', ')}`);
          // Let the user know we'll try local models
          vscode.window.showWarningMessage(`RAGnarōk: Remote model "${targetModel}" could not be loaded. Trying local models...`);

          // Try each local model until one initializes successfully
          for (const local of localModels) {
            try {
              this.logger.info(`Attempting to initialize local model: ${local}`);
              await this.initializeModel(local);
              const successMsg = `RAGnarōk: Successfully initialized local model "${local}" after failing to load "${targetModel}".`;
              this.logger.info(successMsg);
              vscode.window.showInformationMessage(successMsg);
              return;
            } catch (localErr: any) {
              this.logger.warn(`Failed to initialize local model "${local}":`, localErr?.message ?? localErr);
              // continue to next local model
            }
          }

          // If we exhausted local models, surface a consolidated error message and fall through to throw
          const message = `RAGnarōk: Failed to initialize any local models after remote model "${targetModel}" failed.`;
          this.logger.error(message);
          vscode.window.showErrorMessage(message);
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
    * Note: the LangChain helper is imported as `langchainEuclideanDistance` to
    * avoid confusion with this class method name.
   */
  public euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    // LangChain's euclideanDistance works on matrices, so wrap vectors in arrays
    const result = langchainEuclideanDistance([a], [b]);
    return result[0][0];
  }

  /**
   * Calculate inner product (dot product) between two embeddings using LangChain's implementation
    * Note: the LangChain helper is imported as `langchainInnerProduct` to
    * avoid confusion with this class method name.
   */
  public innerProduct(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    // LangChain's innerProduct works on matrices, so wrap vectors in arrays
    const result = langchainInnerProduct([a], [b]);
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
    this.currentModel = EmbeddingService.DEFAULT_MODEL;
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
    this.currentModel = EmbeddingService.DEFAULT_MODEL;
    this.lastSuccessfulModel = null;

    // Clear transformers module reference
    this.transformers = null;

    // Clear initialization promise
    this.initPromise = null;

    this.logger.info('EmbeddingService disposed');
  }
}
