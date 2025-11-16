import { Document as LangChainDocument, MappingDocumentTransformer } from "@langchain/core/documents";
import { Logger } from "../utils/logger";

export interface CleaningRule {
  /** Human-friendly name to aid logging/metadata */
  name: string;
  /** Regex used to scrub unwanted content */
  pattern: RegExp;
  /** Replacement string; defaults to removing the match */
  replacement?: string;
}

export interface DocumentCleaningOptions {
  /** Remove Markdown/HTML style Table of Contents sections */
  removeTableOfContents?: boolean;
  /** Drop pages that look like a PDF table of contents (early pages with dotted leaders) */
  dropLikelyTocPages?: boolean;
  /** Additional custom cleaning rules to apply */
  additionalRules?: CleaningRule[];
  /** Normalize whitespace (collapse blank lines, trim) */
  normalizeWhitespace?: boolean;
}

const DEFAULT_OPTIONS: Required<Omit<DocumentCleaningOptions, "additionalRules">> = {
  removeTableOfContents: true,
  dropLikelyTocPages: true,
  normalizeWhitespace: true,
};

const TABLE_OF_CONTENTS_RULE: CleaningRule = {
  name: "table-of-contents",
  // Matches headings like "Table of Contents" or "Contents" followed by bullet/numbered lines
  pattern: /^\s{0,3}(?:#{1,6}\s*)?(?:table of contents|contents)\s*:?\s*\n(?:(?:[ \t]*[-*+]\s+.+|[ \t]*\d+\.\s+.+)\n?)+/gim,
  replacement: "",
};

// Handles both contiguous and spaced dotted leaders (". . .") preceding page numbers
const DOTTED_LEADER_FRAGMENT = "(?:\\.{3,}|(?:\\s*\\.\\s*){6,})";
const DOTTED_LEADER_LINE = `\\s{0,6}[A-Za-z0-9].{0,120}${DOTTED_LEADER_FRAGMENT}\\s*\\d{1,4}\\s*$`;
const DOTTED_LEADER_TO_PAGE_REGEX = new RegExp(`${DOTTED_LEADER_FRAGMENT}\\s*\\d{1,4}`, "g");

const DOTTED_LEADER_TOC_RULE: CleaningRule = {
  name: "table-of-contents-dotted",
  // Matches blocks with dotted leaders to page numbers (common in PDFs)
  pattern: new RegExp(`^(?:${DOTTED_LEADER_LINE}\\s*){2,}`, "gm"),
  replacement: "",
};

/**
 * Lightweight document cleaner that strips boilerplate (e.g., Table of Contents)
 * before downstream chunking/embedding.
 */
export class DocumentCleaner extends MappingDocumentTransformer {
  private readonly logger: Logger;
  private readonly options: {
    removeTableOfContents: boolean;
    dropLikelyTocPages: boolean;
    normalizeWhitespace: boolean;
  };
  private readonly rules: CleaningRule[];

  constructor(options: DocumentCleaningOptions = {}) {
    super();
    this.logger = new Logger("DocumentCleaner");
    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
      normalizeWhitespace:
        options.normalizeWhitespace ?? DEFAULT_OPTIONS.normalizeWhitespace,
      removeTableOfContents:
        options.removeTableOfContents ?? DEFAULT_OPTIONS.removeTableOfContents,
      dropLikelyTocPages:
        options.dropLikelyTocPages ?? DEFAULT_OPTIONS.dropLikelyTocPages,
    };

    this.rules = [];

    if (this.options.removeTableOfContents) {
      this.rules.push(TABLE_OF_CONTENTS_RULE);
      this.rules.push(DOTTED_LEADER_TOC_RULE);
    }

    if (options.additionalRules?.length) {
      this.rules.push(...options.additionalRules);
    }
  }

  /**
   * Public helper wrapper around LangChain's transformDocuments.
   */
  public async cleanDocuments(
    documents: LangChainDocument[]
  ): Promise<LangChainDocument[]> {
    return this.transformDocuments(documents);
  }

  public async _transformDocument(
    document: LangChainDocument
  ): Promise<LangChainDocument> {
    // Drop early TOC-like PDF pages before doing text-level scrubbing
    if (
      this.options.dropLikelyTocPages &&
      this.isLikelyTableOfContentsPage(document)
    ) {
      this.logger.debug("Dropping likely table of contents page", {
        source: document.metadata.source || document.metadata.filePath,
        page: document.metadata.page,
      });

      return new LangChainDocument({
        pageContent: "",
        metadata: {
          ...document.metadata,
          cleaned: true,
          droppedAsTableOfContents: true,
        },
      });
    }

    let content = document.pageContent;
    const appliedRules: string[] = [];

    for (const rule of this.rules) {
      const nextContent = content.replace(
        rule.pattern,
        rule.replacement ?? ""
      );

      if (nextContent !== content) {
        appliedRules.push(rule.name);
        content = nextContent;
      }
    }

    if (this.options.normalizeWhitespace) {
      content = this.normalizeWhitespace(content);
    }

    if (appliedRules.length > 0) {
      this.logger.debug("Applied cleaning rules", {
        rules: appliedRules,
        originalLength: document.pageContent.length,
        cleanedLength: content.length,
        source: document.metadata.source || document.metadata.filePath,
      });
    }

    return new LangChainDocument({
      pageContent: content,
      metadata: {
        ...document.metadata,
        cleaned: true,
        cleaningRulesApplied: appliedRules,
        removedTableOfContents: appliedRules.includes("table-of-contents"),
      },
    });
  }

  private normalizeWhitespace(text: string): string {
    const PARAGRAPH_BREAK = "__RAGNAROK_PARA_BREAK__";

    let normalized = text.replace(/\r\n?/g, "\n").replace(/\u00a0/g, " ");

    // Preserve double+ newlines as explicit paragraph breaks
    normalized = normalized.replace(/\n{2,}/g, `\n${PARAGRAPH_BREAK}\n`);

    // Collapse remaining single newlines into spaces to avoid jagged chunks
    normalized = normalized.replace(/\s*\n\s*/g, " ");

    // Collapse excessive whitespace
    normalized = normalized.replace(/[ \t]{2,}/g, " ");

    // Restore paragraph breaks
    normalized = normalized.replace(
      new RegExp(`\\s*${PARAGRAPH_BREAK}\\s*`, "g"),
      "\n\n"
    );

    // Final tidy-up
    normalized = normalized.replace(/\n{3,}/g, "\n\n").trim();

    return normalized;
  }

  /**
   * Heuristic detector for PDF-style Table of Contents pages
   */
  private isLikelyTableOfContentsPage(document: LangChainDocument): boolean {
    const content = document.pageContent || "";
    if (!content) return false;

    const pageNumber =
      typeof document.metadata?.page === "number"
        ? (document.metadata.page as number)
        : undefined;

    // Prioritize early pages; if no page metadata, be conservative
    const isEarlyPage = pageNumber !== undefined ? pageNumber <= 6 : false;
    const lower = content.toLowerCase();

    const dottedLeaderMatches =
      content.match(DOTTED_LEADER_TO_PAGE_REGEX)?.length || 0;
    const containsContentsHeading =
      lower.includes("table of contents") || /^contents\b/m.test(lower);

    const alphaCount = (content.match(/[a-zA-Z]/g) || []).length;
    const digitCount = (content.match(/\d/g) || []).length;
    const totalChars = content.length || 1;
    const alphaRatio = alphaCount / totalChars;
    const digitRatio = digitCount / totalChars;

    const looksListLike =
      dottedLeaderMatches >= 3 || (digitRatio > 0.2 && alphaRatio < 0.45);

    return (isEarlyPage || containsContentsHeading) && looksListLike;
  }
}
