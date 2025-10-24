/**
 * Integration Tests
 * Tests the complete pipeline: Document Processing -> Chunking -> Embedding
 */

import { expect } from 'chai';
import { DocumentProcessor } from '../src/documentProcessor';
import { EmbeddingService } from '../src/embeddingService';
import * as path from 'path';
import * as fs from 'fs';

// Mock vscode module
const mockVscode = {
  workspace: {
    getConfiguration: () => ({
      get: (key: string, defaultValue: any) => {
        if (key === 'embeddingModel') {
          return 'Xenova/all-MiniLM-L6-v2';
        }
        if (key === 'pdfStructureDetection') {
          return 'heuristic';
        }
        return defaultValue;
      }
    })
  },
  window: {
    withProgress: async (options: any, task: any) => {
      const progress = {
        report: (value: any) => {
          console.log(`Progress: ${value.message || ''}`);
        }
      };
      return await task(progress);
    }
  }
};

(global as any).vscode = mockVscode;

describe('Integration: Document Processing + Chunking + Embeddings', function() {
  this.timeout(120000); // 2 minutes for model download

  let embeddingService: EmbeddingService;
  const fixturesPath = path.join(__dirname, 'fixtures');

  before(async function() {
    console.log('\n=== Initializing Embedding Model ===');
    embeddingService = EmbeddingService.getInstance();
    await embeddingService.initialize('Xenova/all-MiniLM-L6-v2');
    console.log('Model initialized successfully!\n');
  });

  describe('Complete Pipeline: Markdown Document', function() {
    it('should process markdown, create chunks, and generate embeddings', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');

      console.log('\n=== Step 1: Process Document ===');
      const doc = await DocumentProcessor.processDocument(mdPath);
      console.log(`Processed: ${doc.metadata.fileName}`);
      console.log(`Content length: ${doc.text.length} chars`);

      console.log('\n=== Step 2: Create Semantic Chunks ===');
      const chunks = DocumentProcessor.splitIntoSemanticChunks(
        doc.markdown,
        512,  // chunk size
        50    // overlap
      );
      console.log(`Created ${chunks.length} semantic chunks`);

      console.log('\n=== Step 3: Generate Embeddings ===');
      const texts = chunks.map(c => c.text);
      const embeddings = await embeddingService.embedBatch(texts);

      expect(embeddings).to.have.lengthOf(chunks.length);
      console.log(`Generated ${embeddings.length} embeddings (${embeddings[0].length}D each)`);

      console.log('\n=== Step 4: Verify Semantic Similarity ===');
      // Chunks from the same section should be more similar to each other
      if (chunks.length >= 2) {
        const similarity = embeddingService.cosineSimilarity(
          embeddings[0],
          embeddings[1]
        );
        console.log(`Similarity between adjacent chunks: ${similarity.toFixed(4)}`);
        expect(similarity).to.be.greaterThan(0.3); // Should have some similarity
      }

      console.log('\n=== Step 5: Display Chunk Details ===');
      chunks.slice(0, 3).forEach((chunk, idx) => {
        console.log(`\n--- Chunk ${idx + 1} ---`);
        console.log(`Section: ${chunk.sectionTitle}`);
        console.log(`Path: ${chunk.headingPath.join(' → ')}`);
        console.log(`Size: ${chunk.text.length} chars`);
        console.log(`Embedding: [${embeddings[idx].slice(0, 3).map(v => v.toFixed(4)).join(', ')}, ...]`);
        console.log(`Text preview: "${chunk.text.substring(0, 100)}..."`);
      });
    });
  });

  describe('Complete Pipeline: HTML Document', function() {
    it('should process HTML, create chunks, and generate embeddings', async function() {
      const htmlPath = path.join(fixturesPath, 'sample.html');

      console.log('\n=== Processing HTML Document ===');
      const doc = await DocumentProcessor.processDocument(htmlPath);
      console.log(`Processed: ${doc.metadata.fileName}`);

      const chunks = DocumentProcessor.splitIntoSemanticChunks(doc.markdown, 512, 50);
      console.log(`Created ${chunks.length} chunks`);

      const texts = chunks.map(c => c.text);
      const embeddings = await embeddingService.embedBatch(texts);

      expect(embeddings).to.have.lengthOf(chunks.length);
      console.log(`Generated ${embeddings.length} embeddings`);
    });
  });

  describe('Complete Pipeline: Plain Text', function() {
    it('should process text, create chunks, and generate embeddings', async function() {
      const txtPath = path.join(fixturesPath, 'sample-text.txt');

      console.log('\n=== Processing Plain Text ===');
      const content = fs.readFileSync(txtPath, 'utf-8');
      console.log(`Content length: ${content.length} chars`);

      // Use legacy chunking for plain text
      const chunks = DocumentProcessor.splitIntoChunks(content, 512, 50);
      console.log(`Created ${chunks.length} chunks`);

      const embeddings = await embeddingService.embedBatch(chunks);

      expect(embeddings).to.have.lengthOf(chunks.length);
      console.log(`Generated ${embeddings.length} embeddings`);
    });
  });

  describe('Semantic Search Simulation', function() {
    it('should find relevant chunks based on query', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');

      console.log('\n=== Simulating Semantic Search ===');

      // Process document
      const doc = await DocumentProcessor.processDocument(mdPath);
      const chunks = DocumentProcessor.splitIntoSemanticChunks(doc.markdown, 512, 50);

      // Generate embeddings for all chunks
      const texts = chunks.map(c => c.text);
      const chunkEmbeddings = await embeddingService.embedBatch(texts);

      // Query
      const query = 'How do I install the dependencies?';
      console.log(`Query: "${query}"`);

      const queryEmbedding = await embeddingService.embed(query);

      // Calculate similarities
      const results = chunkEmbeddings.map((embedding, idx) => ({
        chunk: chunks[idx],
        similarity: embeddingService.cosineSimilarity(queryEmbedding, embedding),
        index: idx
      }));

      // Sort by similarity
      results.sort((a, b) => b.similarity - a.similarity);

      // Get top 3 results
      const topResults = results.slice(0, 3);

      console.log('\n=== Top 3 Results ===');
      topResults.forEach((result, idx) => {
        console.log(`\n${idx + 1}. Similarity: ${result.similarity.toFixed(4)}`);
        console.log(`   Section: ${result.chunk.sectionTitle}`);
        console.log(`   Path: ${result.chunk.headingPath.join(' → ')}`);
        console.log(`   Text: "${result.chunk.text.substring(0, 150)}..."`);
      });

      // The top result should have good similarity
      expect(topResults[0].similarity).to.be.greaterThan(0.4);

      // The top result should be from "Installation" section
      const topSection = topResults[0].chunk.headingPath.join(' ');
      console.log(`\nTop result is from: ${topSection}`);
      expect(topSection.toLowerCase()).to.satisfy((s: string) =>
        s.includes('install') || s.includes('getting started') || s.includes('prerequisite')
      );
    });
  });

  describe('Cross-Document Search', function() {
    it('should find similar content across different documents', async function() {
      console.log('\n=== Cross-Document Search ===');

      // Process multiple documents
      const mdPath = path.join(fixturesPath, 'sample.md');
      const htmlPath = path.join(fixturesPath, 'sample.html');

      const mdDoc = await DocumentProcessor.processDocument(mdPath);
      const htmlDoc = await DocumentProcessor.processDocument(htmlPath);

      const mdChunks = DocumentProcessor.splitIntoSemanticChunks(mdDoc.markdown, 512, 50);
      const htmlChunks = DocumentProcessor.splitIntoSemanticChunks(htmlDoc.markdown, 512, 50);

      // Combine all chunks with document metadata
      const allChunks = [
        ...mdChunks.map(c => ({ ...c, source: 'sample.md' })),
        ...htmlChunks.map(c => ({ ...c, source: 'sample.html' }))
      ];

      console.log(`Total chunks across documents: ${allChunks.length}`);
      console.log(`  - From sample.md: ${mdChunks.length}`);
      console.log(`  - From sample.html: ${htmlChunks.length}`);

      // Generate embeddings for all chunks
      const allTexts = allChunks.map(c => c.text);
      const allEmbeddings = await embeddingService.embedBatch(allTexts);

      // Query about machine learning
      const query = 'What is machine learning?';
      console.log(`\nQuery: "${query}"`);

      const queryEmbedding = await embeddingService.embed(query);

      // Find top results across all documents
      const results = allEmbeddings.map((embedding, idx) => ({
        chunk: allChunks[idx],
        similarity: embeddingService.cosineSimilarity(queryEmbedding, embedding)
      }));

      results.sort((a, b) => b.similarity - a.similarity);

      const topResults = results.slice(0, 3);

      console.log('\n=== Top 3 Cross-Document Results ===');
      topResults.forEach((result, idx) => {
        console.log(`\n${idx + 1}. Similarity: ${result.similarity.toFixed(4)}`);
        console.log(`   Source: ${result.chunk.source}`);
        console.log(`   Section: ${result.chunk.sectionTitle}`);
        console.log(`   Text: "${result.chunk.text.substring(0, 120)}..."`);
      });

      // Should find relevant content
      expect(topResults[0].similarity).to.be.greaterThan(0.4);
    });
  });

  describe('Performance Benchmarks', function() {
    it('should process and embed documents efficiently', async function() {
      const mdPath = path.join(fixturesPath, 'sample.md');

      console.log('\n=== Performance Benchmark ===');

      const startTotal = Date.now();

      // Document processing
      const startProcess = Date.now();
      const doc = await DocumentProcessor.processDocument(mdPath);
      const processTime = Date.now() - startProcess;

      // Chunking
      const startChunk = Date.now();
      const chunks = DocumentProcessor.splitIntoSemanticChunks(doc.markdown, 512, 50);
      const chunkTime = Date.now() - startChunk;

      // Embedding
      const startEmbed = Date.now();
      const texts = chunks.map(c => c.text);
      const embeddings = await embeddingService.embedBatch(texts);
      const embedTime = Date.now() - startEmbed;

      const totalTime = Date.now() - startTotal;

      console.log('\nTiming Results:');
      console.log(`  Document Processing: ${processTime}ms`);
      console.log(`  Chunking: ${chunkTime}ms`);
      console.log(`  Embedding (${chunks.length} chunks): ${embedTime}ms`);
      console.log(`  Average per chunk: ${(embedTime / chunks.length).toFixed(2)}ms`);
      console.log(`  Total Pipeline: ${totalTime}ms`);

      // Performance assertions
      expect(processTime).to.be.lessThan(500);
      expect(chunkTime).to.be.lessThan(100);
      expect(embedTime / chunks.length).to.be.lessThan(200); // Less than 200ms per chunk
    });
  });
});

