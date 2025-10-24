/**
 * Tests for DocumentProcessor
 * Tests chunking for various document types: text, markdown, HTML
 */

import { expect } from 'chai';
import { DocumentProcessor } from '../src/documentProcessor';
import * as path from 'path';
import * as fs from 'fs';

// Mock vscode module
const mockVscode = {
  workspace: {
    getConfiguration: () => ({
      get: (key: string, defaultValue: any) => {
        if (key === 'pdfStructureDetection') {
          return 'heuristic';
        }
        return defaultValue;
      }
    })
  }
};

(global as any).vscode = mockVscode;

describe('DocumentProcessor', function() {
  this.timeout(10000);

  const fixturesPath = path.join(__dirname, 'fixtures');

  describe('File Validation', function() {
    it('should validate existing files', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');
      const isValid = await DocumentProcessor.validateFile(mdPath);
      expect(isValid).to.be.true;
    });

    it('should reject non-existent files', async function() {
      const fakePath = path.join(fixturesPath, 'nonexistent.md');
      const isValid = await DocumentProcessor.validateFile(fakePath);
      expect(isValid).to.be.false;
    });

    it('should return supported extensions', function() {
      const extensions = DocumentProcessor.getSupportedExtensions();
      expect(extensions).to.include('.md');
      expect(extensions).to.include('.pdf');
      expect(extensions).to.include('.html');
      expect(extensions).to.include('.htm');
      expect(extensions).to.include('.markdown');
    });
  });

  describe('Markdown Processing', function() {
    it('should process markdown file', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');
      const result = await DocumentProcessor.processDocument(mdPath);

      expect(result).to.have.property('text');
      expect(result).to.have.property('markdown');
      expect(result).to.have.property('metadata');
      expect(result.metadata.fileName).to.equal('sample.md');
      expect(result.metadata.fileType).to.equal('markdown');

      console.log(`Processed markdown file: ${result.text.length} chars`);
      console.log(`Metadata:`, result.metadata);
    });

    it('should preserve markdown structure', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');
      const result = await DocumentProcessor.processDocument(mdPath);

      // Should contain headings
      expect(result.markdown).to.include('# Sample Markdown Document');
      expect(result.markdown).to.include('## Introduction');
      expect(result.markdown).to.include('### Prerequisites');
    });
  });

  describe('HTML Processing', function() {
    it('should process HTML file and convert to markdown', async function() {
      const htmlPath = path.join(fixturesPath, 'sample.html');
      const result = await DocumentProcessor.processDocument(htmlPath);

      expect(result).to.have.property('text');
      expect(result).to.have.property('markdown');
      expect(result.metadata.fileName).to.equal('sample.html');
      expect(result.metadata.fileType).to.equal('html');

      // Should have converted HTML to markdown
      expect(result.markdown).to.include('Machine Learning Basics');
      expect(result.markdown).to.match(/#{1,3}/); // Should have markdown headings

      console.log(`Processed HTML file: ${result.text.length} chars`);
      console.log(`First 200 chars of markdown:\n${result.markdown.substring(0, 200)}`);
    });
  });

  describe('Semantic Chunking', function() {
    it('should split markdown into semantic chunks', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');
      const result = await DocumentProcessor.processDocument(mdPath);

      const chunks = DocumentProcessor.splitIntoSemanticChunks(
        result.markdown,
        500,  // chunkSize
        50    // overlap
      );

      expect(chunks).to.be.an('array');
      expect(chunks.length).to.be.greaterThan(0);

      console.log(`\nCreated ${chunks.length} semantic chunks:`);

      chunks.forEach((chunk, idx) => {
        expect(chunk).to.have.property('text');
        expect(chunk).to.have.property('headingPath');
        expect(chunk).to.have.property('headingLevel');
        expect(chunk).to.have.property('sectionTitle');
        expect(chunk).to.have.property('startPosition');
        expect(chunk).to.have.property('endPosition');

        console.log(`\nChunk ${idx + 1}:`);
        console.log(`  Section: ${chunk.sectionTitle}`);
        console.log(`  Path: ${chunk.headingPath.join(' → ')}`);
        console.log(`  Level: ${chunk.headingLevel}`);
        console.log(`  Size: ${chunk.text.length} chars`);
        console.log(`  Preview: "${chunk.text.substring(0, 80)}..."`);
      });
    });

    it('should maintain heading hierarchy in chunks', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');
      const result = await DocumentProcessor.processDocument(mdPath);

      const chunks = DocumentProcessor.splitIntoSemanticChunks(
        result.markdown,
        500,
        50
      );

      // Check that chunks have proper heading paths
      const chunkWithPath = chunks.find(c => c.headingPath.length > 1);
      if (chunkWithPath) {
        expect(chunkWithPath.headingPath).to.be.an('array');
        expect(chunkWithPath.headingPath.length).to.be.greaterThan(0);
        console.log(`\nExample heading path: ${chunkWithPath.headingPath.join(' → ')}`);
      }
    });

    it('should respect chunk size limits', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');
      const result = await DocumentProcessor.processDocument(mdPath);

      const chunkSize = 300;
      const chunks = DocumentProcessor.splitIntoSemanticChunks(
        result.markdown,
        chunkSize,
        50
      );

      // Most chunks should be around the chunk size (some may be larger if sections are small)
      const oversizedChunks = chunks.filter(c => c.text.length > chunkSize * 1.5);
      const percentageOversized = (oversizedChunks.length / chunks.length) * 100;

      console.log(`\n${chunks.length} total chunks with ${chunkSize} char limit`);
      console.log(`${oversizedChunks.length} oversized chunks (${percentageOversized.toFixed(1)}%)`);

      // Most chunks should respect the size limit
      expect(percentageOversized).to.be.lessThan(30);
    });

    it('should handle documents without headings', async function() {
      const markdown = `
        This is a plain text document without any headings.
        It just contains multiple paragraphs of text that need to be chunked.
        The chunking algorithm should handle this gracefully and create chunks anyway.

        Here is another paragraph with more content to ensure we have enough text
        to create multiple chunks for testing purposes.
      `;

      const chunks = DocumentProcessor.splitIntoSemanticChunks(markdown, 100, 20);

      expect(chunks).to.be.an('array');
      expect(chunks.length).to.be.greaterThan(0);

      // Should have a default heading for content before first heading
      const firstChunk = chunks[0];
      expect(firstChunk.headingPath).to.include('(Introduction)');
    });
  });

  describe('Legacy Chunking', function() {
    it('should split text into chunks with overlap', function() {
      const text = 'This is a test. ' + 'Lorem ipsum dolor sit amet. '.repeat(20);

      const chunks = DocumentProcessor.splitIntoChunks(text, 100, 20);

      expect(chunks).to.be.an('array');
      expect(chunks.length).to.be.greaterThan(1);

      console.log(`\nCreated ${chunks.length} legacy chunks:`);
      chunks.forEach((chunk, idx) => {
        console.log(`  Chunk ${idx + 1}: ${chunk.length} chars - "${chunk.substring(0, 50)}..."`);
      });
    });

    it('should break at sentence boundaries when possible', function() {
      const text = 'First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.';

      const chunks = DocumentProcessor.splitIntoChunks(text, 40, 10);

      // Most chunks should end with a period (sentence boundary)
      const sentenceEndChunks = chunks.filter(c => c.trim().endsWith('.'));
      const percentage = (sentenceEndChunks.length / chunks.length) * 100;

      console.log(`\n${sentenceEndChunks.length}/${chunks.length} chunks (${percentage.toFixed(1)}%) end at sentence boundaries`);
      expect(percentage).to.be.greaterThan(50);
    });

    it('should handle text smaller than chunk size', function() {
      const text = 'Short text.';
      const chunks = DocumentProcessor.splitIntoChunks(text, 100, 20);

      expect(chunks).to.have.lengthOf(1);
      expect(chunks[0]).to.equal(text);
    });
  });

  describe('Chunk Overlap', function() {
    it('should create overlapping chunks', function() {
      const text = 'Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10';

      const chunks = DocumentProcessor.splitIntoChunks(text, 20, 5);

      expect(chunks.length).to.be.greaterThan(1);

      // Check for overlap between consecutive chunks
      for (let i = 0; i < chunks.length - 1; i++) {
        const currentChunk = chunks[i];
        const nextChunk = chunks[i + 1];

        // Extract last few chars of current and first few of next
        const currentEnd = currentChunk.slice(-10);
        const nextStart = nextChunk.slice(0, 10);

        console.log(`Chunk ${i + 1} end: "...${currentEnd}"`);
        console.log(`Chunk ${i + 2} start: "${nextStart}..."`);
      }
    });
  });

  describe('Plain Text Processing', function() {
    it('should process plain text file', async function() {
      const txtPath = path.join(fixturesPath, 'sample-text.txt');

      // Read as plain text and create chunks
      const content = fs.readFileSync(txtPath, 'utf-8');
      const chunks = DocumentProcessor.splitIntoChunks(content, 500, 50);

      expect(chunks).to.be.an('array');
      expect(chunks.length).to.be.greaterThan(0);

      console.log(`\nProcessed plain text: ${content.length} chars`);
      console.log(`Created ${chunks.length} chunks`);

      chunks.slice(0, 3).forEach((chunk, idx) => {
        console.log(`\nChunk ${idx + 1}: ${chunk.length} chars`);
        console.log(`Preview: "${chunk.substring(0, 100)}..."`);
      });
    });
  });

  describe('Performance', function() {
    it('should process large documents efficiently', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');

      const startTime = Date.now();
      const result = await DocumentProcessor.processDocument(mdPath);
      const chunks = DocumentProcessor.splitIntoSemanticChunks(result.markdown, 500, 50);
      const endTime = Date.now();

      const duration = endTime - startTime;
      console.log(`\nProcessed document and created ${chunks.length} chunks in ${duration}ms`);

      expect(duration).to.be.lessThan(1000); // Should be fast
    });
  });
});

