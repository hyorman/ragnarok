/**
 * Semantic Chunker - Intelligent text splitting using LangChain splitters
 * Preserves document structure and semantic boundaries
 *
 * Architecture: Strategy pattern with automatic splitter selection
 * - Markdown: Respects heading hierarchy
 * - Code: Preserves syntax structure
 * - General: Smart boundary detection
 */

import * as vscode from 'vscode';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { MarkdownTextSplitter } from '@langchain/textsplitters';
import { Logger } from '../utils/logger';
import { CONFIG } from '../constants';

export interface ChunkingOptions {
  /** Target chunk size in characters */
  chunkSize?: number;

  /** Overlap between chunks for context continuity */
  chunkOverlap?: number;

  /** Preserve document structure (headings, sections) */
  preserveStructure?: boolean;

  /** File type hint for optimal chunking */
  fileType?: 'markdown' | 'code' | 'text' | 'html';

  /** Custom separators for splitting */
  customSeparators?: string[];

  /** Add heading metadata to chunks */
  includeHeadingMetadata?: boolean;
}

export interface ChunkingResult {
  /** Chunked documents */
  chunks: LangChainDocument[];

  /** Chunking strategy used */
  strategy: string;

  /** Total chunks created */
  chunkCount: number;

  /** Original document count */
  documentCount: number;

  /** Processing time in milliseconds */
  processingTime: number;

  /** Statistics */
  stats: {
    avgChunkSize: number;
    minChunkSize: number;
    maxChunkSize: number;
    totalCharacters: number;
  };
}

/**
 * Semantic Chunker for intelligent text splitting
 */
export class SemanticChunker {
  private logger: Logger;
  private defaultChunkSize: number;
  private defaultChunkOverlap: number;

  constructor() {
    this.logger = new Logger('SemanticChunker');

    // Load configuration
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    this.defaultChunkSize = config.get<number>(CONFIG.CHUNK_SIZE, 1000);
    this.defaultChunkOverlap = config.get<number>(CONFIG.CHUNK_OVERLAP, 200);

    this.logger.info('SemanticChunker initialized', {
      defaultChunkSize: this.defaultChunkSize,
      defaultChunkOverlap: this.defaultChunkOverlap,
    });
  }

  /**
   * Chunk documents using the appropriate strategy
   */
  public async chunkDocuments(
    documents: LangChainDocument[],
    options: ChunkingOptions = {}
  ): Promise<ChunkingResult> {
    const startTime = Date.now();

    this.logger.info('Chunking documents', {
      documentCount: documents.length,
      options,
    });

    try {
      // Determine chunking strategy
      const strategy = this.determineStrategy(documents, options);

      this.logger.debug('Using chunking strategy', { strategy });

      // Get or create splitter
      const splitter = this.createSplitter(strategy, options);

      // Split documents
      const chunks = await splitter.splitDocuments(documents);

      // Add chunk metadata
      const enrichedChunks = this.enrichChunks(chunks, options);

      // Calculate statistics
      const stats = this.calculateStats(enrichedChunks);

      const processingTime = Date.now() - startTime;

      this.logger.info('Chunking completed', {
        strategy,
        documentCount: documents.length,
        chunkCount: enrichedChunks.length,
        processingTime,
        stats,
      });

      return {
        chunks: enrichedChunks,
        strategy,
        chunkCount: enrichedChunks.length,
        documentCount: documents.length,
        processingTime,
        stats,
      };
    } catch (error) {
      this.logger.error('Failed to chunk documents', {
        error: error instanceof Error ? error.message : String(error),
        documentCount: documents.length,
      });
      throw error;
    }
  }

  /**
   * Chunk a single document
   */
  public async chunkDocument(
    document: LangChainDocument,
    options: ChunkingOptions = {}
  ): Promise<LangChainDocument[]> {
    const result = await this.chunkDocuments([document], options);
    return result.chunks;
  }

  /**
   * Get recommended chunk size based on use case
   */
  public static getRecommendedChunkSize(useCase: 'qa' | 'summarization' | 'search'): number {
    switch (useCase) {
      case 'qa':
        // Smaller chunks for precise Q&A
        return 500;
      case 'summarization':
        // Larger chunks for summarization
        return 2000;
      case 'search':
        // Medium chunks for search
        return 1000;
      default:
        return 1000;
    }
  }

  // ==================== Private Methods ====================

  /**
   * Determine the best chunking strategy based on documents and options
   */
  private determineStrategy(
    documents: LangChainDocument[],
    options: ChunkingOptions
  ): 'markdown' | 'recursive' | 'code' {
    // Explicit file type
    if (options.fileType === 'markdown') {
      return 'markdown';
    }

    if (options.fileType === 'code') {
      return 'code';
    }

    // Check document metadata
    const hasMarkdown = documents.some(
      (doc) => doc.metadata.isMarkdown || doc.metadata.fileType === 'markdown'
    );

    if (hasMarkdown && options.preserveStructure !== false) {
      return 'markdown';
    }

    // Check if content looks like code
    const hasCode = documents.some((doc) => {
      const content = doc.pageContent;
      // Simple heuristic: check for common code patterns
      const codePatterns = [
        /^(function|class|def|import|const|let|var)\s+/m,
        /^\s*(public|private|protected)\s+/m,
        /\{[\s\S]*\}/m, // Curly braces
      ];
      return codePatterns.some((pattern) => pattern.test(content));
    });

    if (hasCode) {
      return 'code';
    }

    // Default to recursive character splitter
    return 'recursive';
  }

  /**
   * Create the appropriate splitter based on strategy
   */
  private createSplitter(
    strategy: 'markdown' | 'recursive' | 'code',
    options: ChunkingOptions
  ) {
    const chunkSize = options.chunkSize ?? this.defaultChunkSize;
    const chunkOverlap = options.chunkOverlap ?? this.defaultChunkOverlap;

    switch (strategy) {
      case 'markdown':
        return this.createMarkdownSplitter(chunkSize, chunkOverlap);

      case 'code':
        return this.createCodeSplitter(chunkSize, chunkOverlap);

      case 'recursive':
      default:
        return this.createRecursiveSplitter(
          chunkSize,
          chunkOverlap,
          options.customSeparators
        );
    }
  }

  /**
   * Create a Markdown text splitter that respects heading hierarchy
   */
  private createMarkdownSplitter(
    chunkSize: number,
    chunkOverlap: number
  ): MarkdownTextSplitter {
    return new MarkdownTextSplitter({
      chunkSize,
      chunkOverlap,
      // Preserve heading hierarchy
      // MarkdownTextSplitter automatically handles heading detection
    });
  }

  /**
   * Create a code-aware splitter
   */
  private createCodeSplitter(
    chunkSize: number,
    chunkOverlap: number
  ): RecursiveCharacterTextSplitter {
    // Code-specific separators that respect syntax structure
    const codeSeparators = [
      '\n\n',           // Double newline (between functions/classes)
      '\nclass ',       // Class definitions
      '\nfunction ',    // Function definitions
      '\ndef ',         // Python function definitions
      '\n\n',           // Paragraph breaks
      '\n',             // Single newline
      '. ',             // Sentence breaks
      ' ',              // Word breaks
      '',               // Character breaks
    ];

    return new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
      separators: codeSeparators,
    });
  }

  /**
   * Create a recursive character splitter with smart boundaries
   */
  private createRecursiveSplitter(
    chunkSize: number,
    chunkOverlap: number,
    customSeparators?: string[]
  ): RecursiveCharacterTextSplitter {
    // Default smart separators that respect semantic boundaries
    const defaultSeparators = [
      '\n\n',     // Paragraph breaks (highest priority)
      '\n',       // Line breaks
      '. ',       // Sentence breaks
      '! ',       // Exclamation breaks
      '? ',       // Question breaks
      '; ',       // Semicolon breaks
      ', ',       // Comma breaks
      ' ',        // Word breaks
      '',         // Character breaks (last resort)
    ];

    const separators = customSeparators || defaultSeparators;

    return new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
      separators,
    });
  }

  /**
   * Enrich chunks with additional metadata
   */
  private enrichChunks(
    chunks: LangChainDocument[],
    options: ChunkingOptions
  ): LangChainDocument[] {
    return chunks.map((chunk, index) => {
      // Extract heading information if present
      const headingInfo = this.extractHeadingInfo(chunk.pageContent);

      const metadata: Record<string, any> = {
        ...chunk.metadata,
        chunkIndex: index,
        chunkSize: chunk.pageContent.length,
        chunkId: this.generateChunkId(chunk, index),
        chunkedAt: Date.now(),
      };

      // Add heading metadata if requested and available
      if (options.includeHeadingMetadata !== false && headingInfo) {
        metadata.headingPath = headingInfo.headingPath;
        metadata.headingLevel = headingInfo.headingLevel;
        metadata.sectionTitle = headingInfo.sectionTitle;
      }

      return new LangChainDocument({
        pageContent: chunk.pageContent,
        metadata,
      });
    });
  }

  /**
   * Extract heading information from markdown content
   */
  private extractHeadingInfo(content: string): {
    headingPath: string[];
    headingLevel: number;
    sectionTitle: string;
  } | null {
    // Extract the first heading if present
    const headingMatch = content.match(/^(#{1,6})\s+(.+)$/m);

    if (!headingMatch) {
      return null;
    }

    const level = headingMatch[1].length;
    const title = headingMatch[2].trim();

    return {
      headingPath: [title],
      headingLevel: level,
      sectionTitle: title,
    };
  }

  /**
   * Generate a unique ID for a chunk
   */
  private generateChunkId(chunk: LangChainDocument, index: number): string {
    const source = chunk.metadata.source || chunk.metadata.filePath || 'unknown';
    const fileName = chunk.metadata.fileName || 'unknown';
    // Create a simple hash-like ID
    const timestamp = Date.now();
    return `${fileName}-${index}-${timestamp}`;
  }

  /**
   * Calculate statistics about the chunks
   */
  private calculateStats(chunks: LangChainDocument[]): {
    avgChunkSize: number;
    minChunkSize: number;
    maxChunkSize: number;
    totalCharacters: number;
  } {
    if (chunks.length === 0) {
      return {
        avgChunkSize: 0,
        minChunkSize: 0,
        maxChunkSize: 0,
        totalCharacters: 0,
      };
    }

    const sizes = chunks.map((chunk) => chunk.pageContent.length);
    const totalCharacters = sizes.reduce((sum, size) => sum + size, 0);

    return {
      avgChunkSize: Math.round(totalCharacters / chunks.length),
      minChunkSize: Math.min(...sizes),
      maxChunkSize: Math.max(...sizes),
      totalCharacters,
    };
  }
}
