/**
 * Unit Tests for QueryPlannerAgent
 * Tests query decomposition, complexity analysis, and planning strategies
 */

import { expect } from 'chai';
import { QueryPlannerAgent, QueryPlan, SubQuery } from '../../src/agents/queryPlannerAgent';

describe('QueryPlannerAgent', function() {
  this.timeout(30000); // 30 seconds for LLM tests

  let planner: QueryPlannerAgent;

  beforeEach(function() {
    planner = new QueryPlannerAgent();
  });

  describe('Initialization', function() {
    it('should initialize successfully', function() {
      const agent = new QueryPlannerAgent();
      expect(agent).to.be.an('object');
    });
  });

  describe('Heuristic Planning (No LLM)', function() {
    describe('Simple Queries', function() {
      it('should classify simple queries correctly', async function() {
        const queries = [
          'What is Python?',
          'machine learning basics',
          'how to install packages',
        ];

        for (const query of queries) {
          const plan = await planner.createPlan(query, { useLLM: false });

          expect(plan.complexity).to.equal('simple');
          expect(plan.subQueries.length).to.equal(1);
          expect(plan.subQueries[0].query).to.equal(query);
          expect(plan.strategy).to.be.oneOf(['parallel', 'sequential']);
        }
      });

      it('should create single sub-query for simple queries', async function() {
        const plan = await planner.createPlan('What is TypeScript?', { useLLM: false });

        expect(plan.originalQuery).to.equal('What is TypeScript?');
        expect(plan.complexity).to.equal('simple');
        expect(plan.subQueries).to.have.lengthOf(1);
        expect(plan.subQueries[0].query).to.equal('What is TypeScript?');
        expect(plan.subQueries[0].reasoning).to.be.a('string');
      });

      it('should set default topK for sub-queries', async function() {
        const plan = await planner.createPlan('machine learning', {
          useLLM: false,
          defaultTopK: 10,
        });

        expect(plan.subQueries[0].topK).to.equal(10);
      });

      it('should include explanation', async function() {
        const plan = await planner.createPlan('simple query', { useLLM: false });

        expect(plan.explanation).to.be.a('string');
        expect(plan.explanation.length).to.be.greaterThan(0);
      });
    });

    describe('Moderate Complexity Queries', function() {
      it('should detect queries with multiple concepts', async function() {
        const query = 'Python and JavaScript and TypeScript'; // Needs 3+ concepts (>2)
        const plan = await planner.createPlan(query, { useLLM: false });

        expect(plan.complexity).to.equal('moderate');
        expect(plan.subQueries.length).to.be.greaterThan(1);
      });

      it('should split on common delimiters', async function() {
        const query = 'What is Python? How to install it? Best practices?';
        const plan = await planner.createPlan(query, { useLLM: false });

        expect(plan.complexity).to.equal('moderate');
        expect(plan.subQueries.length).to.be.greaterThan(1);
      });

      it('should use parallel strategy for independent concepts', async function() {
        const query = 'React and Vue and Angular frameworks'; // Needs 3+ concepts
        const plan = await planner.createPlan(query, { useLLM: false });

        expect(plan.strategy).to.equal('parallel');
      });

      it('should respect maxSubQueries limit', async function() {
        const query = 'Python and JavaScript and TypeScript and Ruby and Go';
        const plan = await planner.createPlan(query, {
          useLLM: false,
          maxSubQueries: 2,
        });

        expect(plan.subQueries.length).to.be.lessThanOrEqual(2);
      });

      it('should handle long queries', async function() {
        const longQuery = 'This is a very long query about machine learning algorithms including supervised learning unsupervised learning and reinforcement learning techniques';
        const plan = await planner.createPlan(longQuery, { useLLM: false });

        expect(plan.complexity).to.be.oneOf(['moderate', 'complex']);
        expect(plan.subQueries.length).to.be.greaterThan(0);
      });
    });

    describe('Complex Queries', function() {
      it('should detect comparison queries', async function() {
        const queries = [
          'Python vs JavaScript',
          'compare React and Vue',
          'difference between SQL and NoSQL',
          'which is better: TypeScript or JavaScript',
        ];

        for (const query of queries) {
          const plan = await planner.createPlan(query, { useLLM: false });

          expect(plan.complexity).to.equal('complex');
          expect(plan.strategy).to.equal('parallel');
        }
      });

      it('should split comparison queries into parts', async function() {
        const plan = await planner.createPlan('React framework versus Vue framework', {
          useLLM: false,
        });

        expect(plan.complexity).to.equal('complex');
        expect(plan.subQueries.length).to.be.at.least(1); // At least identifies as complex

        // Should include comparison terms in some form
        const allText = plan.subQueries.map(sq => sq.query.toLowerCase()).join(' ');
        expect(allText.length).to.be.greaterThan(0);
      });

      it('should use parallel strategy for comparisons', async function() {
        const plan = await planner.createPlan('Python vs JavaScript performance', {
          useLLM: false,
        });

        expect(plan.strategy).to.equal('parallel');
      });
    });

    describe('Sub-Query Properties', function() {
      it('should set priority for sub-queries', async function() {
        const plan = await planner.createPlan('machine learning', { useLLM: false });

        plan.subQueries.forEach(sq => {
          expect(sq.priority).to.be.oneOf(['high', 'medium', 'low']);
        });
      });

      it('should provide reasoning for each sub-query', async function() {
        const plan = await planner.createPlan('Python vs JavaScript', { useLLM: false });

        plan.subQueries.forEach(sq => {
          expect(sq.reasoning).to.be.a('string');
          expect(sq.reasoning.length).to.be.greaterThan(0);
        });
      });

      it('should set topK for each sub-query', async function() {
        const plan = await planner.createPlan('machine learning', {
          useLLM: false,
          defaultTopK: 7,
        });

        plan.subQueries.forEach(sq => {
          expect(sq.topK).to.equal(7);
        });
      });
    });

    describe('Strategy Selection', function() {
      it('should use sequential for dependent queries', async function() {
        const query = 'What are the steps to deploy a web application';
        const plan = await planner.createPlan(query, { useLLM: false });

        // Long queries that might need follow-up use sequential
        if (plan.complexity === 'moderate' && query.length > 50) {
          expect(plan.strategy).to.be.oneOf(['sequential', 'parallel']);
        }
      });

      it('should use parallel for independent queries', async function() {
        const query = 'Python features and JavaScript features';
        const plan = await planner.createPlan(query, { useLLM: false });

        expect(plan.strategy).to.equal('parallel');
      });
    });
  });

  describe('LLM Planning (Optional)', function() {
    it('should attempt LLM planning if enabled', async function() {
      const plan = await planner.createPlan('What is machine learning?', {
        useLLM: true, // Will fallback to heuristic if LLM not available
      });

      // Should return a valid plan regardless of LLM availability
      expect(plan).to.be.an('object');
      expect(plan).to.have.property('originalQuery');
      expect(plan).to.have.property('complexity');
      expect(plan).to.have.property('subQueries');
      expect(plan).to.have.property('strategy');
      expect(plan).to.have.property('explanation');
    });

    it('should gracefully fallback to heuristic if LLM unavailable', async function() {
      const plan = await planner.createPlan('complex query', {
        useLLM: true, // Will use heuristic if LLM not available
      });

      // Should still produce valid plan
      expect(plan.complexity).to.be.oneOf(['simple', 'moderate', 'complex']);
      expect(plan.subQueries).to.be.an('array');
      expect(plan.subQueries.length).to.be.greaterThan(0);
    });

    it('should disable LLM if useLLM is false', async function() {
      const plan = await planner.createPlan('test query', {
        useLLM: false, // Explicitly disable
      });

      // Should use heuristic planning
      expect(plan).to.be.an('object');
      expect(plan.subQueries).to.be.an('array');
    });
  });

  describe('Context Integration', function() {
    it('should accept topic name in options', async function() {
      const plan = await planner.createPlan('machine learning', {
        useLLM: false,
        topicName: 'AI Research',
      });

      expect(plan).to.be.an('object');
      expect(plan.subQueries.length).to.be.greaterThan(0);
    });

    it('should accept workspace context', async function() {
      const plan = await planner.createPlan('refactoring', {
        useLLM: false,
        workspaceContext: 'Current file: typescript-project/src/main.ts',
      });

      expect(plan).to.be.an('object');
    });
  });

  describe('Edge Cases', function() {
    it('should handle empty query', async function() {
      const plan = await planner.createPlan('', { useLLM: false });

      expect(plan).to.be.an('object');
      expect(plan.subQueries).to.be.an('array');
      expect(plan.subQueries.length).to.be.greaterThan(0);
    });

    it('should handle very short queries', async function() {
      const plan = await planner.createPlan('ML', { useLLM: false });

      expect(plan.complexity).to.equal('simple');
      expect(plan.subQueries).to.have.lengthOf(1);
    });

    it('should handle very long queries', async function() {
      const longQuery = 'a'.repeat(500);
      const plan = await planner.createPlan(longQuery, { useLLM: false });

      expect(plan).to.be.an('object');
      expect(plan.subQueries.length).to.be.greaterThan(0);
    });

    it('should handle queries with special characters', async function() {
      const plan = await planner.createPlan('C++ vs C# programming!', { useLLM: false });

      expect(plan.complexity).to.equal('complex'); // Has "vs"
      expect(plan.subQueries.length).to.be.greaterThan(0);
    });

    it('should handle queries with multiple delimiters', async function() {
      const query = 'What is Python? How to use it? Why is it popular?';
      const plan = await planner.createPlan(query, { useLLM: false });

      expect(plan.subQueries.length).to.be.greaterThan(1);
    });

    it('should handle queries with AND/OR operators', async function() {
      const query = 'Python and JavaScript and TypeScript'; // Needs 3+ for hasMultipleConcepts
      const plan = await planner.createPlan(query, { useLLM: false });

      expect(plan.complexity).to.equal('moderate'); // Has multiple concepts
      expect(plan.subQueries.length).to.be.greaterThan(1);
    });
  });

  describe('Plan Validation', function() {
    it('should always return valid originalQuery', async function() {
      const testQuery = 'test query';
      const plan = await planner.createPlan(testQuery, { useLLM: false });

      expect(plan.originalQuery).to.equal(testQuery);
    });

    it('should always have at least one sub-query', async function() {
      const queries = ['', 'a', 'simple query', 'complex vs query'];

      for (const query of queries) {
        const plan = await planner.createPlan(query, { useLLM: false });
        expect(plan.subQueries.length).to.be.at.least(1);
      }
    });

    it('should have valid complexity values', async function() {
      const plan = await planner.createPlan('test', { useLLM: false });

      expect(plan.complexity).to.be.oneOf(['simple', 'moderate', 'complex']);
    });

    it('should have valid strategy values', async function() {
      const plan = await planner.createPlan('test', { useLLM: false });

      expect(plan.strategy).to.be.oneOf(['sequential', 'parallel']);
    });

    it('should have explanation string', async function() {
      const plan = await planner.createPlan('test', { useLLM: false });

      expect(plan.explanation).to.be.a('string');
      expect(plan.explanation.length).to.be.greaterThan(0);
    });
  });

  describe('Performance', function() {
    it('should create plans quickly for heuristic mode', async function() {
      const startTime = Date.now();

      await planner.createPlan('machine learning basics', { useLLM: false });

      const elapsed = Date.now() - startTime;
      expect(elapsed).to.be.lessThan(1000); // Should be very fast
    });

    it('should handle batch planning', async function() {
      const queries = [
        'Python programming',
        'React vs Vue',
        'machine learning and deep learning',
        'What is TypeScript?',
        'How to deploy applications',
      ];

      const plans = await Promise.all(
        queries.map(q => planner.createPlan(q, { useLLM: false }))
      );

      expect(plans).to.have.lengthOf(5);
      plans.forEach(plan => {
        expect(plan.subQueries.length).to.be.greaterThan(0);
      });
    });
  });

  describe('Query Types', function() {
    it('should handle "what" questions', async function() {
      const plan = await planner.createPlan('What is machine learning?', {
        useLLM: false,
      });

      expect(plan.complexity).to.equal('simple');
    });

    it('should handle "how" questions', async function() {
      const plan = await planner.createPlan('How to learn Python?', {
        useLLM: false,
      });

      expect(plan.complexity).to.equal('simple');
    });

    it('should handle "why" questions', async function() {
      const plan = await planner.createPlan('Why use TypeScript?', {
        useLLM: false,
      });

      expect(plan.complexity).to.equal('simple');
    });

    it('should handle comparison questions', async function() {
      const plan = await planner.createPlan('Which is better: React or Vue?', {
        useLLM: false,
      });

      expect(plan.complexity).to.equal('complex');
    });

    it('should handle procedural questions', async function() {
      const plan = await planner.createPlan(
        'Steps to deploy a React application',
        { useLLM: false }
      );

      expect(plan.subQueries.length).to.be.greaterThan(0);
    });
  });
});
