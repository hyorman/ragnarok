/**
 * Unit Tests for RAGAgent
 * Tests orchestration, query planning, retrieval, and iterative refinement
 */

import { expect } from 'chai';
import { RAGAgent } from '../../src/agents/ragAgent';
import { RetrievalStrategy } from '../../src/utils/types';
import { VectorStore } from '@langchain/core/vectorstores';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { Embeddings } from '@langchain/core/embeddings';

// Mock VectorStore for testing
class MockVectorStore extends VectorStore {
  _vectorstoreType(): string {
    return 'mock';
  }

  private documents: LangChainDocument[] = [];

  constructor() {
    super(new MockEmbeddings(), {});
  }

  async addDocuments(docs: LangChainDocument[]): Promise<void> {
    this.documents.push(...docs);
  }

  async addVectors(): Promise<void> {
    // Not used
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number
  ): Promise<[LangChainDocument, number][]> {
    // Return mock results with normalized scores
    return this.documents.slice(0, k).map((doc, index) => {
      const score = Math.max(0, 1 - index * 0.1); // Decreasing scores
      return [doc, score];
    });
  }
}

// Mock Embeddings
class MockEmbeddings extends Embeddings {
  constructor() {
    super({});
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map(() => [0.1, 0.2, 0.3]);
  }

  async embedQuery(text: string): Promise<number[]> {
    return [0.1, 0.2, 0.3];
  }
}

describe('RAGAgent', function () {
  this.timeout(30000); // 30 seconds for LLM tests

  let agent: RAGAgent;
  let mockVectorStore: MockVectorStore;

  beforeEach(async function () {
    agent = new RAGAgent();
    mockVectorStore = new MockVectorStore();

    // Add mock documents
    const mockDocs = [
      new LangChainDocument({
        pageContent: 'Python is a programming language',
        metadata: { chunkId: 'chunk1', source: 'test1.txt' },
      }),
      new LangChainDocument({
        pageContent: 'JavaScript is used for web development',
        metadata: { chunkId: 'chunk2', source: 'test2.txt' },
      }),
      new LangChainDocument({
        pageContent: 'TypeScript is a superset of JavaScript',
        metadata: { chunkId: 'chunk3', source: 'test3.txt' },
      }),
      new LangChainDocument({
        pageContent: 'Machine learning uses Python',
        metadata: { chunkId: 'chunk4', source: 'test4.txt' },
      }),
      new LangChainDocument({
        pageContent: 'React is a JavaScript library',
        metadata: { chunkId: 'chunk5', source: 'test5.txt' },
      }),
    ];

    await mockVectorStore.addDocuments(mockDocs);
    await agent.initialize(mockVectorStore);
  });

  describe('Initialization', function () {
    it('should initialize successfully', function () {
      const newAgent = new RAGAgent();
      expect(newAgent).to.be.an('object');
    });

    it('should initialize with vector store', async function () {
      const newAgent = new RAGAgent();
      await newAgent.initialize(mockVectorStore);

      // Verify the agent is properly initialized and can execute queries
      expect(newAgent).to.be.an('object');
      expect(newAgent).to.respondTo('simpleQuery');
      expect(newAgent).to.respondTo('query');
    });

    it('should use configuration from query options', async function () {
      // Configuration is passed as options, not stored in agent
      const results = await agent.simpleQuery('test query', 3, RetrievalStrategy.VECTOR);

      expect(results).to.be.an('array');
      expect(results.length).to.be.at.most(3);
    });
  });

  describe('Simple Query (Fast Path)', function () {
    it('should execute simple query without planning', async function () {
      const results = await agent.simpleQuery('Python programming', 3);

      expect(results).to.be.an('array');
      expect(results.length).to.be.at.most(3);

      results.forEach((result) => {
        expect(result).to.have.property('document');
        expect(result).to.have.property('score');
        expect(result).to.have.property('source');
      });
    });

    it('should respect topK parameter', async function () {
      const results = await agent.simpleQuery('programming', 2);

      expect(results.length).to.be.at.most(2);
    });

    it('should return results with scores', async function () {
      const results = await agent.simpleQuery('JavaScript', 5);

      results.forEach((result) => {
        expect(result.score).to.be.a('number');
        expect(result.score).to.be.within(0, 1);
      });
    });

    it('should throw error if not initialized', async function () {
      const uninitializedAgent = new RAGAgent();

      try {
        await uninitializedAgent.simpleQuery('test');
        expect.fail('Should have thrown error');
      } catch (error) {
        expect(error).to.be.an('error');
      }
    });
  });

  describe('Full Query with Planning', function () {
    it('should execute full RAG query', async function () {
      const result = await agent.query('What is Python?', {
        useLLM: false,
        topK: 3,
      });

      expect(result).to.have.property('query');
      expect(result).to.have.property('plan');
      expect(result).to.have.property('results');
      expect(result).to.have.property('iterations');
      expect(result).to.have.property('avgConfidence');
      expect(result).to.have.property('confidenceMet');
      expect(result).to.have.property('executionTime');
      expect(result).to.have.property('metadata');
    });

    it('should create query plan', async function () {
      const result = await agent.query('machine learning', {
        useLLM: false,
      });

      expect(result.plan).to.be.an('object');
      expect(result.plan).to.have.property('originalQuery');
      expect(result.plan).to.have.property('complexity');
      expect(result.plan).to.have.property('subQueries');
      expect(result.plan).to.have.property('strategy');
    });

    it('should return results array', async function () {
      const result = await agent.query('programming languages', {
        useLLM: false,
        topK: 5,
      });

      expect(result.results).to.be.an('array');
      expect(result.results.length).to.be.at.most(5);
    });

    it('should calculate average confidence', async function () {
      const result = await agent.query('JavaScript frameworks', {
        useLLM: false,
      });

      expect(result.avgConfidence).to.be.a('number');
      expect(result.avgConfidence).to.be.within(0, 1);
    });

    it('should track execution time', async function () {
      const result = await agent.query('Python', { useLLM: false });

      expect(result.executionTime).to.be.a('number');
      expect(result.executionTime).to.be.at.least(0); // Can be 0 if very fast
    });

    it('should include metadata', async function () {
      const result = await agent.query('test query', { useLLM: false });

      expect(result.metadata).to.have.property('totalResults');
      expect(result.metadata).to.have.property('uniqueDocuments');
      expect(result.metadata).to.have.property('strategy');
      expect(result.metadata).to.have.property('subQueriesExecuted');
    });
  });

  describe('Query Options', function () {
    it('should accept topic name', async function () {
      const result = await agent.query('programming', {
        useLLM: false,
        topicName: 'Programming Languages',
      });

      expect(result).to.be.an('object');
      expect(result.results).to.be.an('array');
    });

    it('should accept workspace context', async function () {
      const result = await agent.query('refactoring', {
        useLLM: false,
        workspaceContext: 'Current file: main.ts',
      });

      expect(result).to.be.an('object');
    });

    it('should accept retrieval strategy', async function () {
      const vectorResult = await agent.query('test', {
        useLLM: false,
        retrievalStrategy: RetrievalStrategy.VECTOR,
      });

      expect(vectorResult.metadata.strategy).to.equal(RetrievalStrategy.VECTOR);
    });

    it('should respect topK parameter', async function () {
      const result = await agent.query('programming', {
        useLLM: false,
        topK: 2,
      });

      expect(result.results.length).to.be.at.most(2);
    });

    it('should accept confidence threshold', async function () {
      const result = await agent.query('test', {
        useLLM: false,
        confidenceThreshold: 0.5,
      });

      expect(result.confidenceMet).to.be.a('boolean');
    });
  });

  describe('Iterative Refinement', function () {
    it('should perform iterative refinement when enabled', async function () {
      const result = await agent.query('compare Python and JavaScript', {
        useLLM: false,
        enableIterativeRefinement: true,
        maxIterations: 2,
      });

      expect(result.iterations).to.be.a('number');
      expect(result.iterations).to.be.at.least(1);
    });

    it('should disable iterative refinement for simple queries', async function () {
      const result = await agent.query('Python', {
        useLLM: false,
        enableIterativeRefinement: true,
      });

      // Simple queries should be single-shot
      expect(result.iterations).to.equal(1);
    });

    it('should respect max iterations', async function () {
      const result = await agent.query('complex query with multiple concepts', {
        useLLM: false,
        enableIterativeRefinement: true,
        maxIterations: 1,
        confidenceThreshold: 0.99, // High threshold won't be met
      });

      expect(result.iterations).to.be.at.most(1);
    });

    it('should stop when confidence threshold met', async function () {
      const result = await agent.query('Python programming', {
        useLLM: false,
        enableIterativeRefinement: true,
        confidenceThreshold: 0.1, // Low threshold
        maxIterations: 5,
      });

      // Should stop early due to low threshold
      expect(result.confidenceMet).to.be.true;
    });
  });

  describe('Parallel Execution', function () {
    it('should execute parallel sub-queries', async function () {
      const result = await agent.query('Python versus JavaScript', {
        useLLM: false,
      });

      // Comparison query should use parallel strategy
      expect(result.plan.strategy).to.equal('parallel');
      expect(result.metadata.subQueriesExecuted).to.be.greaterThan(0);
    });

    it('should handle multiple parallel queries', async function () {
      const result = await agent.query(
        'Python and JavaScript and TypeScript',
        {
          useLLM: false,
        }
      );

      expect(result.results).to.be.an('array');
      expect(result.results.length).to.be.greaterThan(0);
    });
  });

  describe('Sequential Execution', function () {
    it('should execute sequential sub-queries', async function () {
      // Long queries might use sequential
      const longQuery = 'What are the steps to learn Python programming from beginner to advanced level';
      const result = await agent.query(longQuery, {
        useLLM: false,
      });

      expect(result.results).to.be.an('array');
    });
  });

  describe('Result Deduplication', function () {
    it('should deduplicate results with same chunkId', async function () {
      const result = await agent.query('Python', {
        useLLM: false,
        topK: 10,
      });

      // Check metadata shows deduplication happened
      expect(result.metadata.uniqueDocuments).to.be.at.most(
        result.metadata.totalResults
      );
    });

    it('should preserve unique documents', async function () {
      const result = await agent.query('programming', {
        useLLM: false,
      });

      const chunkIds = result.results.map((r) => r.document.metadata.chunkId);
      const uniqueIds = new Set(chunkIds);

      // All results should have unique chunk IDs
      expect(uniqueIds.size).to.equal(chunkIds.length);
    });
  });

  describe('Result Ranking', function () {
    it('should rank results by score', async function () {
      const result = await agent.query('Python', {
        useLLM: false,
        topK: 5,
      });

      // Check scores are in descending order
      for (let i = 1; i < result.results.length; i++) {
        expect(result.results[i - 1].score).to.be.at.least(
          result.results[i].score
        );
      }
    });

    it('should return highest scoring results', async function () {
      const result = await agent.query('JavaScript', {
        useLLM: false,
        topK: 3,
      });

      expect(result.results.length).to.be.at.most(3);

      // First result should have highest score
      if (result.results.length > 1) {
        expect(result.results[0].score).to.be.at.least(
          result.results[result.results.length - 1].score
        );
      }
    });
  });

  describe('Error Handling', function () {
    it('should throw error if not initialized', async function () {
      const uninitializedAgent = new RAGAgent();

      try {
        await uninitializedAgent.query('test');
        expect.fail('Should have thrown error');
      } catch (error) {
        expect(error).to.be.an('error');
        expect((error as Error).message).to.include('not initialized');
      }
    });

    it('should handle empty query gracefully', async function () {
      const result = await agent.query('', { useLLM: false });

      expect(result).to.be.an('object');
      expect(result.results).to.be.an('array');
    });

    it('should handle invalid options gracefully', async function () {
      const result = await agent.query('test', {
        useLLM: false,
        topK: -1, // Invalid topK
      });

      expect(result).to.be.an('object');
    });
  });

  describe('Configuration Management', function () {
    it('should accept configuration via query options', async function () {
      // Configuration is now passed as options to query methods
      const result = await agent.query('test query', {
        maxIterations: 2,
        confidenceThreshold: 0.8,
        retrievalStrategy: RetrievalStrategy.HYBRID,
        useLLM: false,
      });

      expect(result).to.have.property('iterations');
      expect(result).to.have.property('avgConfidence');
    });

    it('should allow updating vector store', function () {
      const newVectorStore = new MockVectorStore();
      agent.setVectorStore(newVectorStore);

      // Should not throw
      expect(agent).to.be.an('object');
    });
  });

  describe('Query Plan Integration', function () {
    it('should use query planner for complex queries', async function () {
      const result = await agent.query('compare React and Vue frameworks', {
        useLLM: false,
      });

      expect(result.plan.complexity).to.equal('complex');
      expect(result.plan.subQueries.length).to.be.greaterThan(0);
    });

    it('should use simple plan for simple queries', async function () {
      const result = await agent.query('Python', { useLLM: false });

      expect(result.plan.complexity).to.equal('simple');
      expect(result.plan.subQueries).to.have.lengthOf(1);
    });
  });

  describe('Result Structure', function () {
    it('should include document in results', async function () {
      const result = await agent.query('test', { useLLM: false });

      result.results.forEach((r) => {
        expect(r.document).to.be.an('object');
        expect(r.document).to.have.property('pageContent');
        expect(r.document).to.have.property('metadata');
      });
    });

    it('should include score in results', async function () {
      const result = await agent.query('test', { useLLM: false });

      result.results.forEach((r) => {
        expect(r.score).to.be.a('number');
        expect(r.score).to.be.within(0, 1);
      });
    });

    it('should include source in results', async function () {
      const result = await agent.query('test', { useLLM: false });

      result.results.forEach((r) => {
        expect(r.source).to.be.oneOf(['vector', 'hybrid', 'keyword']);
      });
    });
  });

  describe('Performance', function () {
    it('should complete query in reasonable time', async function () {
      const startTime = Date.now();

      await agent.query('Python programming', { useLLM: false });

      const elapsed = Date.now() - startTime;
      expect(elapsed).to.be.lessThan(5000); // 5 seconds
    });

    it('should handle multiple queries', async function () {
      const queries = ['Python', 'JavaScript', 'TypeScript'];

      const results = await Promise.all(
        queries.map((q) => agent.query(q, { useLLM: false, topK: 2 }))
      );

      expect(results).to.have.lengthOf(3);
      results.forEach((result) => {
        expect(result.results).to.be.an('array');
      });
    });
  });
});
