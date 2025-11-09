/**
 * LangChain-compatible wrapper for our existing EmbeddingService
 * This allows us to use our HuggingFace Transformers.js embeddings
 * with LangChain's vector stores and other components
 */

import { Embeddings, EmbeddingsParams } from "@langchain/core/embeddings";
import { EmbeddingService } from "./embeddingService";

export interface TransformersEmbeddingsParams extends EmbeddingsParams {
  /** Model name to use (e.g., "Xenova/all-MiniLM-L6-v2") */
  modelName?: string;
}

/**
 * LangChain Embeddings implementation using our existing EmbeddingService
 * which uses @huggingface/transformers (Transformers.js) for local embeddings
 */
export class TransformersEmbeddings extends Embeddings {
  private embeddingService: EmbeddingService;
  private modelName?: string;

  constructor(fields?: TransformersEmbeddingsParams) {
    super(fields ?? {});
    this.embeddingService = EmbeddingService.getInstance();
    this.modelName = fields?.modelName;
  }

  /**
   * Embed a list of documents (batch operation)
   */
  async embedDocuments(documents: string[]): Promise<number[][]> {
    // Initialize with configured model if specified
    if (this.modelName) {
      await this.embeddingService.initialize(this.modelName);
    } else {
      await this.embeddingService.initialize();
    }

    // Use the batch embedding method for efficiency
    return await this.embeddingService.embedBatch(documents);
  }

  /**
   * Embed a single query text
   */
  async embedQuery(query: string): Promise<number[]> {
    // Initialize with configured model if specified
    if (this.modelName) {
      await this.embeddingService.initialize(this.modelName);
    } else {
      await this.embeddingService.initialize();
    }

    return await this.embeddingService.embed(query);
  }
}
