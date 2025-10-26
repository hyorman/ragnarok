/**
 * Embedding service using Transformers.js for local sentence transformers
 */

import * as vscode from 'vscode';
import { pipeline, FeatureExtractionPipeline, env } from '@huggingface/transformers';
import { CONFIG } from './constants';

export class EmbeddingService {
  private static instance: EmbeddingService;
  private pipeline: FeatureExtractionPipeline | null = null;
  private currentModel: string | null = null;
  private isInitializing: boolean = false;
  private initializationPromise: Promise<void> | null = null;

  private constructor() {
    // Configure transformers.js environment at module load time
    env.allowLocalModels = true;
    env.allowRemoteModels = true; // Allow downloading models from HuggingFace
    env.useBrowserCache = false;
    // Disable ONNX WASM backend to avoid loading unnecessary dependencies
    if ("backend" in env) {
      (env as any).backends.onnx.wasm = false;
    }
  }

  public static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  /**
   * Initialize the embedding model
   */
  public async initialize(modelName?: string): Promise<void> {
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    const targetModel = modelName || config.get<string>('embeddingModel', 'Xenova/all-MiniLM-L6-v2');

    // If already initialized with the same model, return
    if (this.pipeline && this.currentModel === targetModel) {
      return;
    }

    // If initialization is in progress, wait for it
    if (this.isInitializing && this.initializationPromise) {
      return this.initializationPromise;
    }

    this.isInitializing = true;
    this.initializationPromise = this._initializePipeline(targetModel);

    try {
      await this.initializationPromise;
    } finally {
      this.isInitializing = false;
      this.initializationPromise = null;
    }
  }

  private async _initializePipeline(modelName: string): Promise<void> {
    try {
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Loading embedding model: ${modelName}`,
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

          this.currentModel = modelName;
          progress.report({ message: 'Model loaded successfully!' });
        }
      );
    } catch (error: any) {
      this.pipeline = null;
      this.currentModel = null;
      // Log full error details for debugging
      console.error('Full error details:', error);
      console.error('Error stack:', error.stack);
      const errorMsg = error.message || String(error);
      throw new Error(`Failed to initialize embedding model "${modelName}": ${errorMsg}`);
    }
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
      const embedding = Array.from(output.data) as number[];
      return embedding;
    } catch (error) {
      throw new Error(`Failed to generate embedding: ${error}`);
    }
  }

  /**
   * Generate embeddings for multiple texts in batch
   */
  public async embedBatch(texts: string[]): Promise<number[][]> {
    if (!this.pipeline) {
      await this.initialize();
    }

    if (!this.pipeline) {
      throw new Error('Embedding pipeline not initialized');
    }

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
          embeddings.push(Array.from(output.data) as number[]);
        }
      }

      return embeddings;
    } catch (error) {
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
    this.pipeline = null;
    this.currentModel = null;
    vscode.window.showInformationMessage('Embedding model cache cleared. Model will reload on next use.');
  }
}

