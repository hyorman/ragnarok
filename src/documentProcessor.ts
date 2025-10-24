/**
 * Document processing for PDF, Markdown, and HTML files
 */

import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import pdfParse from 'pdf-parse';
import TurndownService from 'turndown';
import { CONFIG } from './constants';

export interface ProcessedDocument {
  text: string;
  markdown: string;  // Always store markdown version
  metadata: {
    fileName: string;
    fileType: string;
    pageCount?: number;
  };
}

export interface SemanticChunk {
  text: string;
  headingPath: string[];  // e.g., ["Getting Started", "Installation", "Prerequisites"]
  headingLevel: number;
  sectionTitle: string;
  startPosition: number;
  endPosition: number;
}

export class DocumentProcessor {
  /**
   * Process a document based on its file type
   * Always converts to markdown for semantic structure
   */
  public static async processDocument(filePath: string): Promise<ProcessedDocument> {
    const ext = path.extname(filePath).toLowerCase();
    const fileName = path.basename(filePath);

    let markdown: string;
    let metadata: ProcessedDocument['metadata'] = {
      fileName,
      fileType: ext.substring(1),
    };

    // Get configuration
    const config = vscode.workspace.getConfiguration(CONFIG.ROOT);
    const pdfStructureDetection = config.get<string>('pdfStructureDetection', 'heuristic');

    switch (ext) {
      case '.pdf':
        const pdfText = await this.processPDF(filePath);
        // For PDFs, use heuristic detection based on user preference
        const useHeuristics = pdfStructureDetection === 'heuristic';
        markdown = this.textToMarkdown(pdfText, useHeuristics);
        break;
      case '.md':
      case '.markdown':
        markdown = await this.processMarkdown(filePath);
        metadata.fileType = 'markdown';
        break;
      case '.html':
      case '.htm':
        markdown = await this.processHTML(filePath);
        metadata.fileType = 'html';
        break;
      default:
        throw new Error(`Unsupported file type: ${ext}`);
    }

    // Extract plain text from markdown for backward compatibility
    const text = this.markdownToPlainText(markdown);

    return { text, markdown, metadata };
  }

  /**
   * Process PDF file
   */
  private static async processPDF(filePath: string): Promise<string> {
    try {
      const dataBuffer = fs.readFileSync(filePath);
      const data = await pdfParse(dataBuffer);
      return data.text;
    } catch (error) {
      throw new Error(`Failed to process PDF: ${error}`);
    }
  }

  /**
   * Process Markdown file - returns markdown as-is
   */
  private static async processMarkdown(filePath: string): Promise<string> {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      return content;
    } catch (error) {
      throw new Error(`Failed to process Markdown: ${error}`);
    }
  }

  /**
   * Process HTML file - converts to markdown
   */
  private static async processHTML(filePath: string): Promise<string> {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const turndownService = new TurndownService();
      return turndownService.turndown(content);
    } catch (error) {
      throw new Error(`Failed to process HTML: ${error}`);
    }
  }

  /**
   * Convert plain text to markdown with structure detection
   * Note: This is a best-effort approach for PDFs which lack semantic structure
   */
  private static textToMarkdown(text: string, useHeuristics: boolean = true): string {
    if (!useHeuristics) {
      // Return as-is - will use plain text chunking
      return text;
    }

    // Split into lines first for better analysis
    const lines = text.split('\n');
    const result: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (!line) {
        result.push('');
        continue;
      }

      // Multi-factor heuristic for heading detection
      const isLikelyHeading = this.isLikelyHeading(line, lines, i);

      if (isLikelyHeading.isHeading) {
        result.push(`${'#'.repeat(isLikelyHeading.level)} ${line}`);
      } else {
        result.push(line);
      }
    }

    return result.join('\n');
  }

  /**
   * Heuristic heading detection with multiple signals
   * WARNING: This is inherently unreliable for PDFs. Markdown/HTML is always better.
   */
  private static isLikelyHeading(
    line: string,
    allLines: string[],
    index: number
  ): { isHeading: boolean; level: number } {

    // Signals that suggest a heading:
    const signals = {
      shortLine: line.length < 80 && line.length > 3,
      fewWords: line.split(/\s+/).length <= 12,
      noEndPunctuation: !/[.!?,;:]$/.test(line),
      startsCapital: /^[A-Z0-9]/.test(line),
      mostlyCapitalized: false,
      allCaps: false,
      titleCase: false,
      followedByContent: false,
      noLowercaseStart: true,
      numeric: /^\d+\.?\s/.test(line), // "1. Introduction" or "1 Introduction"
    };

    // Check capitalization patterns
    const words = line.split(/\s+/);
    const capitalizedWords = words.filter(w => /^[A-Z]/.test(w)).length;
    signals.titleCase = capitalizedWords === words.length && words.length > 0;
    signals.allCaps = line === line.toUpperCase() && /[A-Z]/.test(line);
    signals.mostlyCapitalized = capitalizedWords / words.length > 0.7;

    // Check if followed by content (not another heading-like line)
    if (index < allLines.length - 1) {
      const nextLine = allLines[index + 1].trim();
      if (nextLine && nextLine.length > 80) {
        signals.followedByContent = true;
      }
    }

    // Scoring system
    let score = 0;
    let level = 2; // Default to H2

    if (signals.shortLine) { score += 1; }
    if (signals.fewWords) { score += 1; }
    if (signals.noEndPunctuation) { score += 2; }
    if (signals.startsCapital) { score += 1; }
    if (signals.followedByContent) { score += 2; }

    // Strong signals
    if (signals.allCaps) {
      score += 3;
      level = 1; // H1 for all caps
    }
    if (signals.titleCase && signals.fewWords) {
      score += 3;
      level = 2; // H2 for title case
    }
    if (signals.numeric) {
      score += 2;
      level = 2;
    }

    // Threshold: need at least 5 points
    const isHeading = score >= 5;

    return { isHeading, level };
  }

  /**
   * Convert markdown to plain text
   */
  private static markdownToPlainText(markdown: string): string {
    // Remove markdown syntax for cleaner text
    return markdown
      .replace(/#{1,6}\s/g, '') // Remove headers
      .replace(/\*\*(.+?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.+?)\*/g, '$1') // Remove italic
      .replace(/\[(.+?)\]\(.+?\)/g, '$1') // Remove links but keep text
      .replace(/`{1,3}(.+?)`{1,3}/g, '$1') // Remove code blocks
      .replace(/^\s*[-*+]\s/gm, '') // Remove list markers
      .replace(/^\s*\d+\.\s/gm, '') // Remove numbered list markers
      .replace(/\n{3,}/g, '\n\n') // Normalize line breaks
      .trim();
  }

  /**
   * Parse markdown and create semantic chunks based on heading hierarchy
   */
  public static splitIntoSemanticChunks(
    markdown: string,
    chunkSize: number,
    overlap: number
  ): SemanticChunk[] {
    const lines = markdown.split('\n');
    const chunks: SemanticChunk[] = [];

    // Parse document structure
    interface Section {
      heading: string;
      level: number;
      startLine: number;
      content: string[];
      path: string[];
    }

    const sections: Section[] = [];
    const headingStack: Array<{ level: number; text: string }> = [];
    let currentSection: Section | null = null;

    // Parse markdown into sections
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);

      if (headingMatch) {
        // Save previous section
        if (currentSection) {
          sections.push(currentSection);
        }

        const level = headingMatch[1].length;
        const heading = headingMatch[2].trim();

        // Update heading stack
        while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
          headingStack.pop();
        }
        headingStack.push({ level, text: heading });

        // Create path from stack
        const path = headingStack.map(h => h.text);

        // Start new section
        currentSection = {
          heading,
          level,
          startLine: i,
          content: [],
          path: [...path]
        };
      } else if (currentSection) {
        // Add content to current section
        currentSection.content.push(line);
      } else if (line.trim()) {
        // Content before first heading
        if (!currentSection) {
          currentSection = {
            heading: '(Introduction)',
            level: 1,
            startLine: i,
            content: [line],
            path: ['(Introduction)']
          };
        }
      }
    }

    // Save last section
    if (currentSection) {
      sections.push(currentSection);
    }

    // Now chunk each section with smart boundaries and overlap
    let globalPosition = 0;

    for (const section of sections) {
      const sectionText = section.content.join('\n').trim();

      if (!sectionText) {
        continue;
      }

      // If section is small enough, keep as single chunk
      if (sectionText.length <= chunkSize) {
        chunks.push({
          text: sectionText,
          headingPath: section.path,
          headingLevel: section.level,
          sectionTitle: section.heading,
          startPosition: globalPosition,
          endPosition: globalPosition + sectionText.length
        });
        globalPosition += sectionText.length;
      } else {
        // Split large section with overlap
        const sectionChunks = this.splitTextWithBoundaries(sectionText, chunkSize, overlap);

        for (let i = 0; i < sectionChunks.length; i++) {
          const chunkText = sectionChunks[i];
          chunks.push({
            text: chunkText,
            headingPath: section.path,
            headingLevel: section.level,
            sectionTitle: section.heading,
            startPosition: globalPosition,
            endPosition: globalPosition + chunkText.length
          });
          globalPosition += chunkText.length;
        }
      }
    }

    return chunks;
  }

  /**
   * Split text with smart boundaries and overlap (helper method)
   */
  private static splitTextWithBoundaries(
    text: string,
    chunkSize: number,
    overlap: number
  ): string[] {
    const chunks: string[] = [];
    const cleanedText = text.replace(/\s+/g, ' ').trim();

    if (cleanedText.length <= chunkSize) {
      return [cleanedText];
    }

    let start = 0;
    while (start < cleanedText.length) {
      let end = start + chunkSize;

      // Smart boundary detection
      if (end < cleanedText.length) {
        // Priority 1: Sentence boundary
        const sentenceEnd = cleanedText.substring(start, end).lastIndexOf('. ');
        if (sentenceEnd > chunkSize * 0.5) {
          end = start + sentenceEnd + 1;
        } else {
          // Priority 2: Word boundary
          const wordEnd = cleanedText.substring(start, end).lastIndexOf(' ');
          if (wordEnd > chunkSize * 0.5) {
            end = start + wordEnd;
          }
        }
      }

      const chunk = cleanedText.substring(start, end).trim();
      if (chunk.length > 0) {
        chunks.push(chunk);
      }

      // Move with overlap
      start = end - overlap;

      // Ensure progress
      if (chunks.length > 0 && start <= 0) {
        start = end;
      }
    }

    return chunks;
  }

  /**
   * Split text into chunks with overlap (legacy method - kept for backward compatibility)
   */
  public static splitIntoChunks(
    text: string,
    chunkSize: number,
    overlap: number
  ): string[] {
    const chunks: string[] = [];

    // Clean and normalize text
    const cleanedText = text.replace(/\s+/g, ' ').trim();

    if (cleanedText.length <= chunkSize) {
      return [cleanedText];
    }

    let start = 0;
    while (start < cleanedText.length) {
      let end = start + chunkSize;

      // If not at the end, try to break at sentence or word boundary
      if (end < cleanedText.length) {
        // Look for sentence ending
        const sentenceEnd = cleanedText.substring(start, end).lastIndexOf('. ');
        if (sentenceEnd > chunkSize * 0.5) {
          end = start + sentenceEnd + 1;
        } else {
          // Look for word boundary
          const wordEnd = cleanedText.substring(start, end).lastIndexOf(' ');
          if (wordEnd > chunkSize * 0.5) {
            end = start + wordEnd;
          }
        }
      }

      const chunk = cleanedText.substring(start, end).trim();
      if (chunk.length > 0) {
        chunks.push(chunk);
      }

      // Move start position with overlap
      start = end - overlap;

      // Ensure we make progress
      if (chunks.length > 0 && start <= 0) {
        start = end;
      }
    }

    return chunks;
  }

  /**
   * Validate if a file exists and is readable
   */
  public static async validateFile(filePath: string): Promise<boolean> {
    try {
      await fs.promises.access(filePath, fs.constants.R_OK);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get supported file extensions
   */
  public static getSupportedExtensions(): string[] {
    return ['.pdf', '.md', '.markdown', '.html', '.htm'];
  }
}

