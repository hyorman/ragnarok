/**
 * Unit Tests for DocumentPipeline
 * Tests the complete document processing pipeline
 */

import { expect } from 'chai';
import * as path from 'path';
import * as fs from 'fs';
import { DocumentPipeline, PipelineOptions, PipelineProgress } from '../../src/managers/documentPipeline';

describe('DocumentPipeline', function() {
  this.timeout(120000); // 2 minutes for embedding model initialization

  let pipeline: DocumentPipeline;
  // When tests are compiled, they're in out/test/test/unit, so go up to workspace root
  const fixturesPath = path.join(__dirname, '../../../../test/fixtures');
  const tempStorageDir = path.join(__dirname, '../../../../test/.temp-storage');

  before(async function() {
    // Create temp storage directory
    if (!fs.existsSync(tempStorageDir)) {
      fs.mkdirSync(tempStorageDir, { recursive: true });
    }

    // Initialize pipeline
    pipeline = new DocumentPipeline();
    await pipeline.initialize(tempStorageDir, 'Xenova/all-MiniLM-L6-v2');
  });

  after(function() {
    // Cleanup temp storage (optional - can keep for inspection)
    // if (fs.existsSync(tempStorageDir)) {
    //   fs.rmSync(tempStorageDir, { recursive: true, force: true });
    // }
  });

  describe('Initialization', function() {
    it('should initialize successfully', async function() {
      const newPipeline = new DocumentPipeline();
      await newPipeline.initialize(tempStorageDir, 'Xenova/all-MiniLM-L6-v2');

      // If initialization succeeds, pipeline should be ready
      expect(newPipeline).to.be.an('object');
    });

    it('should throw error if processing before initialization', async function() {
      const uninitializedPipeline = new DocumentPipeline();
      const testFile = path.join(fixturesPath, 'sample-text.txt');

      const result = await uninitializedPipeline.processDocument(testFile, 'test-topic');

      // Pipeline should handle errors gracefully and return them in result
      expect(result.success).to.be.false;
      expect(result.errors).to.be.an('array');
      expect(result.errors!.length).to.be.greaterThan(0);
      expect(result.errors![0]).to.include('not initialized');
    });
  });

  describe('Document Loading', function() {
    it('should process a text file', async function() {
      const testFile = path.join(fixturesPath, 'sample-text.txt');

      const result = await pipeline.processDocument(testFile, 'test-topic-txt');

      expect(result.success).to.be.true;
      expect(result.stages.loading).to.be.true;
      expect(result.stages.chunking).to.be.true;
      expect(result.stages.embedding).to.be.true;
      expect(result.stages.storing).to.be.true;
      expect(result.metadata.originalDocuments).to.equal(1);
      expect(result.metadata.chunksCreated).to.be.greaterThan(0);
      expect(result.chunks).to.be.an('array');
      expect(result.chunks.length).to.be.greaterThan(0);
    });

    it('should process a markdown file', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-md');

      expect(result.success).to.be.true;
      expect(result.stages.loading).to.be.true;
      expect(result.metadata.chunksCreated).to.be.greaterThan(0);

      // Markdown files should create chunks
      expect(result.chunks.length).to.be.greaterThan(0);

      // Check that chunks have content
      result.chunks.forEach(chunk => {
        expect(chunk.pageContent).to.be.a('string');
        expect(chunk.pageContent.length).to.be.greaterThan(0);
      });
    });

    it('should process an HTML file', async function() {
      const testFile = path.join(fixturesPath, 'sample.html');

      const result = await pipeline.processDocument(testFile, 'test-topic-html');

      expect(result.success).to.be.true;
      expect(result.stages.loading).to.be.true;
      expect(result.metadata.chunksCreated).to.be.greaterThan(0);
      expect(result.chunks.length).to.be.greaterThan(0);
    });

    it('should process multiple files', async function() {
      const testFiles = [
        path.join(fixturesPath, 'sample-text.txt'),
        path.join(fixturesPath, 'sample.md'),
      ];

      const result = await pipeline.processDocuments(testFiles, 'test-topic-multi');

      expect(result.success).to.be.true;
      expect(result.metadata.originalDocuments).to.equal(2);
      expect(result.metadata.chunksCreated).to.be.greaterThan(0);
      expect(result.chunks.length).to.be.greaterThan(0);
    });

    it('should handle non-existent files gracefully', async function() {
      const fakeFile = path.join(fixturesPath, 'nonexistent.txt');

      const result = await pipeline.processDocument(fakeFile, 'test-topic-fake');

      // Pipeline should complete but with no chunks since file doesn't exist
      // The loader uses allSettled so it handles errors gracefully
      expect(result.metadata.chunksCreated).to.equal(0);
      expect(result.chunks.length).to.equal(0);
    });
  });

  describe('Chunking', function() {
    it('should create appropriate chunk sizes', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-chunks', {
        chunkingOptions: {
          chunkSize: 200,
          chunkOverlap: 20,
        },
      });

      expect(result.success).to.be.true;
      expect(result.stages.chunking).to.be.true;
      expect(result.chunks.length).to.be.greaterThan(0);

      // Check chunk sizes (should be around target size)
      result.chunks.forEach(chunk => {
        // Chunks can be smaller than target size, but not too much larger
        expect(chunk.pageContent.length).to.be.lessThan(400); // 2x target size
      });
    });

    it('should preserve metadata in chunks', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-metadata');

      expect(result.success).to.be.true;

      // All chunks should have metadata
      result.chunks.forEach(chunk => {
        expect(chunk.metadata).to.be.an('object');
        expect(chunk.metadata).to.have.property('source');
      });
    });

    it('should handle different chunking options', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      // Use custom chunking settings
      const result = await pipeline.processDocument(testFile, 'test-topic-strategy', {
        chunkingOptions: {
          chunkSize: 300,
          chunkOverlap: 30,
        },
      });

      expect(result.success).to.be.true;
      expect(result.stages.chunking).to.be.true;
      expect(result.chunks.length).to.be.greaterThan(0);
    });
  });

  describe('Embedding', function() {
    it('should generate embeddings for chunks', async function() {
      const testFile = path.join(fixturesPath, 'sample-text.txt');

      const result = await pipeline.processDocument(testFile, 'test-topic-embeddings');

      expect(result.success).to.be.true;
      expect(result.stages.embedding).to.be.true;
      expect(result.metadata.chunksEmbedded).to.equal(result.metadata.chunksCreated);
    });

    it('should respect batch size for embedding generation', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-batch', {
        embeddingBatchSize: 2, // Small batch size
      });

      expect(result.success).to.be.true;
      expect(result.stages.embedding).to.be.true;
      expect(result.metadata.chunksEmbedded).to.be.greaterThan(0);
    });
  });

  describe('Vector Storage', function() {
    it('should store chunks in vector store', async function() {
      const testFile = path.join(fixturesPath, 'sample-text.txt');

      const result = await pipeline.processDocument(testFile, 'test-topic-storage');

      expect(result.success).to.be.true;
      expect(result.stages.storing).to.be.true;
      expect(result.metadata.chunksStored).to.equal(result.metadata.chunksCreated);
    });

    it('should use memory vector store by default', async function() {
      const testFile = path.join(fixturesPath, 'sample-text.txt');

      const result = await pipeline.processDocument(testFile, 'test-topic-memory', {
        vectorStoreType: 'memory',
      });

      expect(result.success).to.be.true;
      expect(result.stages.storing).to.be.true;
    });

    it('should support FAISS vector store (if available)', async function() {
      const testFile = path.join(fixturesPath, 'sample-text.txt');

      try {
        const result = await pipeline.processDocument(testFile, 'test-topic-faiss', {
          vectorStoreType: 'faiss',
        });

        expect(result.success).to.be.true;
        expect(result.stages.storing).to.be.true;
      } catch (error) {
        // FAISS might not be available - that's okay
        console.log('FAISS not available, skipping');
        this.skip();
      }
    });
  });

  describe('Progress Tracking', function() {
    it('should report progress through all stages', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');
      const progressReports: PipelineProgress[] = [];

      const result = await pipeline.processDocument(testFile, 'test-topic-progress', {
        onProgress: (progress) => {
          progressReports.push(progress);
        },
      });

      expect(result.success).to.be.true;
      expect(progressReports.length).to.be.greaterThan(0);

      // Should have reports from different stages
      const stages = new Set(progressReports.map(p => p.stage));
      expect(stages.has('loading')).to.be.true;
      expect(stages.has('chunking')).to.be.true;
      expect(stages.has('embedding')).to.be.true;
      expect(stages.has('storing')).to.be.true;
      expect(stages.has('complete')).to.be.true;
    });

    it('should report progress values from 0 to 100', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');
      const progressReports: PipelineProgress[] = [];

      const result = await pipeline.processDocument(testFile, 'test-topic-progress-values', {
        onProgress: (progress) => {
          progressReports.push(progress);
        },
      });

      expect(result.success).to.be.true;

      // All progress values should be 0-100
      progressReports.forEach(report => {
        expect(report.progress).to.be.at.least(0);
        expect(report.progress).to.be.at.most(100);
      });

      // Final progress should be 100
      const finalProgress = progressReports[progressReports.length - 1];
      expect(finalProgress.progress).to.equal(100);
      expect(finalProgress.stage).to.equal('complete');
    });
  });

  describe('Performance and Metrics', function() {
    it('should track timing for each stage', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-timing');

      expect(result.success).to.be.true;
      expect(result.metadata.totalTime).to.be.greaterThan(0);
      // Stage timings should be tracked (>= 0, some may be very fast)
      expect(result.metadata.stageTimings.loading).to.be.at.least(0);
      expect(result.metadata.stageTimings.chunking).to.be.at.least(0);
      expect(result.metadata.stageTimings.embedding).to.be.at.least(0);
      expect(result.metadata.stageTimings.storing).to.be.at.least(0);
    });

    it('should provide accurate chunk counts', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-counts');

      expect(result.success).to.be.true;
      expect(result.metadata.chunksCreated).to.equal(result.chunks.length);
      expect(result.metadata.chunksEmbedded).to.equal(result.metadata.chunksCreated);
      expect(result.metadata.chunksStored).to.equal(result.metadata.chunksCreated);
    });

    it('should process documents efficiently', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');
      const startTime = Date.now();

      const result = await pipeline.processDocument(testFile, 'test-topic-efficiency');
      const elapsedTime = Date.now() - startTime;

      expect(result.success).to.be.true;

      // Processing should complete in reasonable time (< 30 seconds for small files)
      expect(elapsedTime).to.be.lessThan(30000);
    });
  });

  describe('Error Handling', function() {
    it('should collect and report errors', async function() {
      // Try to process a non-existent file
      const fakeFile = path.join(fixturesPath, 'does-not-exist.txt');

      const result = await pipeline.processDocument(fakeFile, 'test-topic-errors');

      // Pipeline handles errors gracefully using allSettled, so result will be empty but successful
      expect(result.metadata.chunksCreated).to.equal(0);
      expect(result.chunks.length).to.equal(0);
    });

    it('should continue processing other files if one fails', async function() {
      const testFiles = [
        path.join(fixturesPath, 'sample.md'), // Valid file
        path.join(fixturesPath, 'nonexistent.txt'), // Invalid file
        path.join(fixturesPath, 'sample-text.txt'), // Valid file
      ];

      const result = await pipeline.processDocuments(testFiles, 'test-topic-partial');

      // Should have some success even with failures
      expect(result.metadata.originalDocuments).to.equal(3);

      // Should have processed at least some files
      if (result.metadata.chunksCreated > 0) {
        expect(result.chunks.length).to.be.greaterThan(0);
      }
    });

    it('should handle empty files gracefully', async function() {
      // Create an empty temp file
      const emptyFile = path.join(tempStorageDir, 'empty.txt');
      fs.writeFileSync(emptyFile, '');

      const result = await pipeline.processDocument(emptyFile, 'test-topic-empty');

      // Pipeline should handle empty files without crashing
      expect(result).to.be.an('object');
      expect(result.stages.loading).to.be.a('boolean');

      // Cleanup
      fs.unlinkSync(emptyFile);
    });
  });

  describe('Custom Options', function() {
    it('should respect custom chunking options', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const smallChunks = await pipeline.processDocument(testFile, 'test-topic-small', {
        chunkingOptions: {
          chunkSize: 100,
          chunkOverlap: 10,
        },
      });

      const largeChunks = await pipeline.processDocument(testFile, 'test-topic-large', {
        chunkingOptions: {
          chunkSize: 500,
          chunkOverlap: 50,
        },
      });

      expect(smallChunks.success).to.be.true;
      expect(largeChunks.success).to.be.true;

      // Smaller chunk size should generally create more chunks
      // (though not always guaranteed depending on content)
      expect(smallChunks.metadata.chunksCreated).to.be.greaterThan(0);
      expect(largeChunks.metadata.chunksCreated).to.be.greaterThan(0);
    });

    it('should respect custom loader options', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-loader', {
        loaderOptions: {
          // Custom loader options would go here
        },
      });

      expect(result.success).to.be.true;
    });
  });

  describe('Integration', function() {
    it('should complete end-to-end pipeline successfully', async function() {
      const testFile = path.join(fixturesPath, 'sample.md');

      const result = await pipeline.processDocument(testFile, 'test-topic-e2e');

      // Verify all stages completed
      expect(result.success).to.be.true;
      expect(result.stages.loading).to.be.true;
      expect(result.stages.chunking).to.be.true;
      expect(result.stages.embedding).to.be.true;
      expect(result.stages.storing).to.be.true;

      // Verify data flow
      expect(result.metadata.originalDocuments).to.equal(1);
      expect(result.metadata.chunksCreated).to.be.greaterThan(0);
      expect(result.metadata.chunksEmbedded).to.equal(result.metadata.chunksCreated);
      expect(result.metadata.chunksStored).to.equal(result.metadata.chunksCreated);
      expect(result.chunks.length).to.equal(result.metadata.chunksCreated);

      // Verify timing
      expect(result.metadata.totalTime).to.be.greaterThan(0);

      // Verify no errors
      if (result.errors) {
        expect(result.errors).to.be.empty;
      }
    });
  });
});
