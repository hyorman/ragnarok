# Iterative Refinement Implementation

## Overview
Implemented the complete iterative refinement feature for the RAGnarōk extension. This feature enables multi-pass retrieval with intelligent query refinement based on result quality analysis.

## Changes Made

### 1. **Main Iteration Loop Enhancement** (`ragAgent.ts` lines 468-495)
- **Before**: Stub implementation with immediate `break` after first iteration
- **After**: Full implementation with query refinement logic
- Calls `refineQueryPlan()` to analyze gaps and generate follow-up queries
- Continues iteration until confidence threshold met or max iterations reached
- Includes error handling for refinement failures

### 2. **New Method: `refineQueryPlan()`**
Main orchestrator for query refinement:
- Determines whether to use LLM or heuristic refinement
- Routes to appropriate refinement strategy
- Returns refined QueryPlan or null if no refinement needed

### 3. **New Method: `refinePlanWithLLM()`**
LLM-powered gap analysis and query generation:
- Uses VS Code Language Model API (Copilot)
- Analyzes current results for information gaps
- Generates targeted follow-up queries based on gaps
- Provides explanatory gap analysis
- Returns structured QueryPlan with refined sub-queries

**Prompt includes:**
- Original query context
- Previously executed sub-queries
- Results summary (count, confidence, uniqueness)
- Top result excerpts for context
- Structured JSON response format

### 4. **New Method: `refinePlanHeuristically()`**
Rule-based refinement for non-LLM mode:

**Three refinement strategies:**

1. **Low Confidence Strategy**:
   - Triggers when confidence < threshold
   - Adds focused queries with key terms from original query
   - Example: "Python" → "Python programming language"

2. **Sparse Results Strategy**:
   - Triggers when results < topK
   - Creates alternative phrasings
   - Example: "How to use X" → "What way to use X"

3. **Too Specific Strategy**:
   - Triggers when unique documents < 3
   - Removes modifiers to broaden search
   - Example: "best Python framework" → "Python framework"

**Heuristics:**
- Limits to max 3 refined queries per iteration
- Prioritizes queries by strategy
- Only refines if: confidence is low OR few documents OR sparse results

### 5. **New Method: `summarizeResults()`**
Helper for creating result summaries:
- Extracts top 5 results by score
- Creates compact excerpts (150 chars)
- Formats for LLM consumption
- Includes relevance scores

## Configuration Settings

The feature uses existing configuration:
- `ragnarok.agenticIterativeRefinement` - Enable/disable (default: true)
- `ragnarok.agenticMaxIterations` - Max loops (default: 3)
- `ragnarok.agenticConfidenceThreshold` - Stop threshold (default: 0.7)
- `ragnarok.agenticUseLLM` - Use Copilot for refinement (default: false)
- `ragnarok.agenticLLMModel` - LLM model choice (default: gpt-4o)

## Workflow

```
┌─────────────────────────────────────────────────┐
│ 1. Execute Initial Query Plan                  │
│    - Run sub-queries from QueryPlannerAgent    │
│    - Collect initial results                   │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│ 2. Calculate Confidence                        │
│    - Average relevance scores                  │
│    - Check vs. threshold (default: 0.7)        │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │ Threshold Met?  │
        └────┬───────┬────┘
             │       │
            YES      NO
             │       │
             │  ┌────▼─────────────────────────────┐
             │  │ 3. Analyze Gaps                  │
             │  │    LLM Mode: Ask Copilot         │
             │  │    Heuristic: Use rules          │
             │  └────┬─────────────────────────────┘
             │       │
             │  ┌────▼─────────────────────────────┐
             │  │ 4. Generate Refined Queries      │
             │  │    - Target information gaps     │
             │  │    - Alternative phrasings       │
             │  │    - Broader/narrower terms      │
             │  └────┬─────────────────────────────┘
             │       │
             │  ┌────▼─────────────────────────────┐
             │  │ 5. Execute Refined Queries       │
             │  │    - Run new sub-queries         │
             │  │    - Append to results           │
             │  └────┬─────────────────────────────┘
             │       │
             │       │ Max iterations?
             │       └────NO────┐
             │                  │
             └──────────────────┴──► Continue Loop
                                │
                               YES
                                │
┌───────────────────────────────▼─────┐
│ 6. Return Combined Results          │
│    - Deduplicate                    │
│    - Rank by score                  │
│    - Limit to topK                  │
└─────────────────────────────────────┘
```

## Example Scenarios

### Scenario 1: LLM-Based Refinement
```typescript
// User query with moderate complexity
Query: "How do React hooks work?"

// Iteration 1: Initial results (confidence: 0.55)
Results: Basic hook examples, but missing useState details

// LLM Analysis:
Gap: "Missing detailed coverage of useState and useEffect patterns"
Refined Queries:
  1. "React useState hook detailed examples"
  2. "React useEffect hook lifecycle"

// Iteration 2: Execute refined queries
Results: Comprehensive hook documentation
Final Confidence: 0.82 → STOP ✓
```

### Scenario 2: Heuristic Refinement
```typescript
// User query with low initial confidence
Query: "Python web frameworks"

// Iteration 1: Results (confidence: 0.45, only 2 unique docs)
Strategy 1: Low confidence → Add specificity
  - "framework Python web frameworks"
  - "web Python web frameworks"

Strategy 3: Too specific → Broaden search
  - "Python web" (removed modifier "frameworks")

// Iteration 2: Execute 3 refined queries
Results: More diverse framework documentation
Final Confidence: 0.73 → STOP ✓
```

## Benefits

1. **Higher Quality Results**: Multiple passes improve result relevance
2. **Gap Filling**: Identifies and fills information gaps automatically
3. **Adaptive Search**: Adjusts strategy based on result quality
4. **Intelligent (LLM Mode)**: Context-aware refinement with Copilot
5. **Fast (Heuristic Mode)**: Rule-based refinement without API calls
6. **Configurable**: Users control iterations and thresholds

## Testing

Existing tests in `test/unit/ragAgent.test.ts` cover:
- ✅ Iterative refinement execution
- ✅ Simple query bypass (single-shot)
- ✅ Max iterations enforcement
- ✅ Confidence threshold stopping

Run tests:
```bash
npm run pretest
npm test
```

## Performance Considerations

- **Iterations**: Default 3 max keeps latency reasonable
- **LLM Mode**: 2-3x slower but much smarter
- **Heuristic Mode**: Fast, no external API calls
- **Early Stop**: Confidence threshold prevents unnecessary iterations
- **Simple Queries**: Automatically skip iteration for efficiency

## Future Enhancements

1. **Learning**: Cache successful refinement patterns
2. **Context Window**: Use previous conversation context
3. **Relevance Feedback**: Let users indicate missing information
4. **Dynamic Thresholds**: Adjust based on query complexity
5. **Parallel Refinement**: Run multiple refinement strategies simultaneously
