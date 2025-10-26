/**
 * Chunk Inspection Script
 * Processes sample documents and saves chunks to JSON for inspection
 * Run with: node scripts/inspect-chunks.js
 */

const path = require('path');
const fs = require('fs');
const Module = require('module');

// Mock vscode module by intercepting require
const originalRequire = Module.prototype.require;
Module.prototype.require = function(id) {
  if (id === 'vscode') {
    return {
      workspace: {
        getConfiguration: () => ({
          get: (key, defaultValue) => {
            if (key === 'embeddingModel') return 'Xenova/all-MiniLM-L6-v2';
            if (key === 'pdfStructureDetection') return 'heuristic';
            return defaultValue;
          }
        })
      },
      window: {
        withProgress: async (options, task) => {
          const progress = {
            report: (value) => console.log(`Progress: ${value.message || ''}`)
          };
          return await task(progress);
        }
      },
      ProgressLocation: {
        Notification: 15,
        Window: 10,
        SourceControl: 1
      }
    };
  }
  return originalRequire.apply(this, arguments);
};

// Now load our compiled modules
const { DocumentProcessor } = require('../out/src/documentProcessor');
const { EmbeddingService } = require('../out/src/embeddingService');

async function inspectDocument(filePath, chunkSize = 512, overlap = 50) {
  console.log(`\n📄 Processing: ${path.basename(filePath)}`);
  console.log('━'.repeat(60));

  const embeddingService = EmbeddingService.getInstance();

  // Process document
  console.log('  Step 1: Reading and processing document...');
  const doc = await DocumentProcessor.processDocument(filePath);

  // Create semantic chunks
  console.log('  Step 2: Creating semantic chunks...');
  const chunks = DocumentProcessor.splitIntoSemanticChunks(doc.markdown, chunkSize, overlap);
  console.log(`  ✓ Created ${chunks.length} chunks`);

  // Generate embeddings
  console.log('  Step 3: Generating embeddings...');
  const texts = chunks.map(c => c.text);
  const embeddings = await embeddingService.embedBatch(texts);
  console.log(`  ✓ Generated ${embeddings.length} embeddings`);

  // Format output
  const output = {
    documentName: doc.metadata.fileName,
    filePath: filePath,
    fileType: doc.metadata.fileType,
    processedAt: new Date().toISOString(),
    metadata: {
      totalCharacters: doc.text.length,
      totalChunks: chunks.length,
      chunkSize: chunkSize,
      overlap: overlap,
      embeddingModel: embeddingService.getCurrentModel() || 'unknown',
      embeddingDimension: embeddings[0]?.length || 0,
    },
    chunks: chunks.map((chunk, idx) => ({
      index: idx,
      sectionTitle: chunk.sectionTitle,
      headingPath: chunk.headingPath,
      headingLevel: chunk.headingLevel,
      text: chunk.text,
      textLength: chunk.text.length,
      position: {
        start: chunk.startPosition,
        end: chunk.endPosition,
      }
    })),
  };

  console.log('  ✓ Document processing complete!');
  return output;
}

async function main() {
  console.log('\n🔍 RAG Chunk Inspector');
  console.log('═'.repeat(60));

  const fixturesPath = path.join(__dirname, '../test/fixtures');
  const outputDir = path.join(__dirname, '../test/chunk-output');

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`\n📁 Created output directory: ${outputDir}`);
  }

  // Initialize embedding service
  console.log('\n=== Initializing Embedding Model ===');
  const EmbeddingService = require('../out/src/embeddingService').EmbeddingService;
  const embeddingService = EmbeddingService.getInstance();
  await embeddingService.initialize('Xenova/all-MiniLM-L6-v2');
  console.log('Model initialized successfully!\n');

  // Documents to process
  const documents = [
    { name: 'sample.md', path: path.join(fixturesPath, 'sample.md') },
    { name: 'sample.html', path: path.join(fixturesPath, 'sample.html') },
    { name: 'sample-text.txt', path: path.join(fixturesPath, 'sample-text.txt') },
  ];

  const allResults = [];

  // Process each document
  for (const doc of documents) {
    if (fs.existsSync(doc.path)) {
      try {
        const result = await inspectDocument(doc.path);
        allResults.push(result);

        // Save individual document output
        const outputPath = path.join(outputDir, `${doc.name}.json`);
        fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
        console.log(`  💾 Saved to: ${outputPath}\n`);
      } catch (error) {
        console.error(`  ❌ Error processing ${doc.name}:`, error.message);
      }
    } else {
      console.log(`  ⚠️  File not found: ${doc.path}\n`);
    }
  }

  // Save combined output
  const combinedPath = path.join(outputDir, 'all-chunks.json');
  fs.writeFileSync(combinedPath, JSON.stringify(allResults, null, 2));
  console.log(`\n💾 Combined output saved to: ${combinedPath}`);

  // Print summary
  console.log('\n📊 Summary');
  console.log('═'.repeat(60));
  allResults.forEach((result) => {
    console.log(`\n${result.documentName}:`);
    console.log(`  • Total chunks: ${result.metadata.totalChunks}`);
    console.log(`  • Total characters: ${result.metadata.totalCharacters}`);
    console.log(`  • Embedding dimension: ${result.metadata.embeddingDimension}`);
    console.log(`  • Sections:`);

    const sections = new Set(result.chunks.map(c => c.sectionTitle));
    sections.forEach(section => {
      const count = result.chunks.filter(c => c.sectionTitle === section).length;
      console.log(`    - ${section}: ${count} chunk(s)`);
    });
  });

  // Print sample chunks
  console.log('\n\n📝 Sample Chunks (First chunk from each document)');
  console.log('═'.repeat(60));
  allResults.forEach((result) => {
    if (result.chunks.length > 0) {
      const chunk = result.chunks[0];
      console.log(`\n[${result.documentName}] Chunk #${chunk.index}`);
      console.log(`Section: ${chunk.sectionTitle}`);
      console.log(`Path: ${chunk.headingPath.join(' → ')}`);
      console.log(`Text (${chunk.textLength} chars): "${chunk.text.substring(0, 150)}..."`);
    }
  });

  console.log('\n\n✅ Chunk inspection complete!');
  console.log(`\n📂 All files saved in: ${outputDir}`);
  console.log('\nYou can now inspect the JSON files to see:');
  console.log('  • Chunk text and structure');
  console.log('  • Section hierarchy');
  console.log('  • Full embedding vectors');
  console.log('  • Position information');
}

// Run the script
main().catch(err => {
  console.error('\n❌ Error:', err);
  process.exit(1);
});

