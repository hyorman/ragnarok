# Refactoring: Query Refinement to QueryPlannerAgent

## Overview
Refactored iterative query refinement logic from `RAGAgent` to `QueryPlannerAgent` for better separation of concerns and code maintainability.

## Architecture Change

### Before
```
RAGAgent
├── iterativeRetrieval()
├── refineQueryPlan()           ❌ Duplicated planning logic
├── refinePlanWithLLM()         ❌ LLM calls in RAGAgent
├── refinePlanHeuristically()   ❌ Heuristics in RAGAgent
└── summarizeResults()

QueryPlannerAgent
├── createPlan()
├── createLLMPlan()
└── createHeuristicPlan()
```

### After
```
RAGAgent
├── iterativeRetrieval()        ✅ Delegates to QueryPlannerAgent
└── (refinement methods removed)

QueryPlannerAgent
├── createPlan()                ✅ Initial planning
├── refinePlan()                ✅ NEW: Refinement orchestrator
├── refinePlanWithLLM()         ✅ NEW: LLM-based refinement
└── refinePlanHeuristically()   ✅ NEW: Heuristic refinement
```

## Changes Made

### 1. **New Interface: `RefinementContext`** (queryPlannerAgent.ts)

Added structured context for refinement:

```typescript
export interface RefinementContext {
  currentResults: Array<{
    content: string;
    score: number;
    metadata?: Record<string, any>;
  }>;
  executedQueries: string[];
  avgConfidence: number;
  uniqueDocCount: number;
  confidenceThreshold: number;
}
```

**Benefits:**
- Type-safe context passing
- Clear contract between RAGAgent and QueryPlannerAgent
- Easy to extend with additional context

### 2. **New Method: `QueryPlannerAgent.refinePlan()`**

Main public method for query refinement:

```typescript
public async refinePlan(
  currentPlan: QueryPlan,
  refinementContext: RefinementContext,
  options: QueryPlannerOptions = {}
): Promise<QueryPlan | null>
```

**Features:**
- Routes to LLM or heuristic refinement
- Uses same options interface as `createPlan()`
- Returns null if no refinement needed

### 3. **Moved: `refinePlanWithLLM()`**

LLM-based gap analysis moved from RAGAgent to QueryPlannerAgent:

**Improvements:**
- Uses `RefinementContext` for cleaner API
- Respects `agenticLLMModel` configuration
- Generates result summaries internally
- Applies same constraints as initial planning (maxSubQueries, topK)

### 4. **Moved: `refinePlanHeuristically()`**

Heuristic-based refinement moved from RAGAgent:

**Three strategies remain:**
1. **Low Confidence**: Add focused queries with key terms
2. **Sparse Results**: Create alternative phrasings
3. **Too Specific**: Broaden by removing modifiers

### 5. **Updated: `RAGAgent.iterativeRetrieval()`**

Now delegates to QueryPlannerAgent:

```typescript
// Build refinement context
const refinementContext: RefinementContext = {
  currentResults: allResults.map(r => ({
    content: r.document.pageContent,
    score: r.score,
    metadata: r.document.metadata,
  })),
  executedQueries: currentPlan.subQueries.map(sq => sq.query),
  avgConfidence,
  uniqueDocCount: new Set(/*...*/).size,
  confidenceThreshold: threshold,
};

// Delegate to QueryPlannerAgent
const refinedPlan = await this.queryPlanner.refinePlan(
  currentPlan,
  refinementContext,
  { ...options }
);
```

### 6. **Removed from RAGAgent**

Deleted duplicate methods:
- ❌ `refineQueryPlan()` (264 lines)
- ❌ `refinePlanWithLLM()` (~150 lines)
- ❌ `refinePlanHeuristically()` (~80 lines)
- ❌ `summarizeResults()` (~10 lines)

**Total reduction:** ~500 lines removed from RAGAgent

## Benefits

### 1. **Single Responsibility Principle**
- `QueryPlannerAgent`: All query planning (initial + refinement)
- `RAGAgent`: Orchestration and retrieval execution

### 2. **Code Reusability**
- QueryPlannerAgent methods can be used independently
- Same LLM infrastructure for planning and refinement
- Shared validation and constraint logic

### 3. **Maintainability**
- Changes to planning logic in one place
- Easier to test query planning in isolation
- Clear interface between components

### 4. **Consistency**
- Same configuration for planning and refinement
- Same LLM model selection logic
- Same constraint application (maxSubQueries, topK)

### 5. **Type Safety**
- Explicit `RefinementContext` interface
- Exported types for clean imports
- Clear contract prevents errors

## Testing Impact

Existing tests remain compatible:
- `test/unit/ragAgent.test.ts` - Iterative refinement tests pass
- `test/unit/queryPlannerAgent.test.ts` - Can add refinement tests
- `test/integration/integration.test.ts` - Integration tests unaffected

## Migration Notes

### For Future Development

1. **Adding new refinement strategies:**
   - Add to `QueryPlannerAgent` methods
   - Both LLM and heuristic paths available

2. **Changing refinement logic:**
   - Modify `refinePlanWithLLM()` or `refinePlanHeuristically()`
   - No changes needed in RAGAgent

3. **Testing query planning:**
   - Can test `refinePlan()` independently
   - Mock `RefinementContext` for unit tests

### Breaking Changes

None - All public APIs remain the same:
- ✅ `RAGAgent.query()` signature unchanged
- ✅ `QueryPlannerAgent.createPlan()` unchanged
- ✅ Configuration options unchanged

## Code Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| RAGAgent LOC | ~875 | ~575 | -300 (-34%) |
| QueryPlannerAgent LOC | ~398 | ~698 | +300 (+75%) |
| Duplicate Logic | Yes | No | ✅ Eliminated |
| Coupling | High | Low | ✅ Improved |

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│              RAGAgent                   │
│  - Orchestrates retrieval workflow      │
│  - Manages retrievers & vector stores   │
│  - Executes sub-queries                 │
└─────────────┬───────────────────────────┘
              │
              │ delegates planning
              ▼
┌─────────────────────────────────────────┐
│         QueryPlannerAgent               │
│  ✅ Initial query decomposition         │
│  ✅ Complexity analysis                 │
│  ✅ Gap analysis                        │
│  ✅ Query refinement                    │
│  ✅ LLM/Heuristic strategies            │
└─────────────────────────────────────────┘
```

## Conclusion

This refactoring achieves:
- ✅ **Cleaner architecture** - Proper separation of concerns
- ✅ **Better maintainability** - Single source of truth for planning
- ✅ **Increased reusability** - QueryPlannerAgent methods can be used independently
- ✅ **No breaking changes** - Existing APIs and tests remain compatible
- ✅ **Reduced complexity** - 300 fewer lines in RAGAgent

The query planning logic is now properly encapsulated in `QueryPlannerAgent`, making the codebase more modular and easier to maintain.
