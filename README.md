<div align="center">
  <img src="./assets/icon.png" alt="RAGnarok icon" title="RAGnarok" width="120" height="120" />
  <h1>RAGnarōk - Local RAG Tool for VSCode</h1>
</div>

A powerful VSCode extension that implements Retrieval-Augmented Generation (RAG) using local sentence transformers. This extension allows you to organize documents by topics, create embeddings locally, and enable Copilot or other LLM agents to query your custom knowledge base.

## Features

- 🧠 **Local Embeddings**: Uses sentence transformers (transformers.js) running entirely locally in VSCode
- 📚 **Document Support**: Process PDF, Markdown, and HTML files
- 🏷️ **Topic Organization**: Organize your documents by topics/components
- 🔍 **Semantic Chunking**: Hierarchical chunking based on document structure (headings) with smart boundaries and overlap
- 🗂️ **Context Preservation**: Each chunk includes its heading path (e.g., "Memory Allocation → Malloc → Performance")
- 🤖 **Copilot Integration**: Register as an LLM tool that Copilot can query
- 🤝 **Agentic RAG** ✨ NEW: Intelligent multi-step retrieval with query planning, iterative refinement, and result evaluation
- 🤖 **LLM-Powered Mode** 🔥 NEW: Optional Copilot-powered query planning and evaluation for true AI reasoning
- 📝 **Workspace Context** 🆕 NEW: Automatically includes selected code and workspace context (like #codebase in Cursor)
- 🎯 **Hybrid Search**: Combines semantic vector search with keyword matching for better precision
- 💾 **Efficient Storage**: Per-topic file storage - only loads what you need
- ⚙️ **Configurable**: Choose from multiple embedding models and search strategies

## Installation

### From Source

1. Clone this repository
2. Run `npm install`
3. Run `npm run compile`
4. Press F5 to run the extension in development mode

### From VSIX

1. Download the `.vsix` file
2. Run `code --install-extension ragnarok-0.0.1.vsix`

## Quick Start

### 1. Create a Topic

```
Ctrl+Shift+P > RAG: Create New Topic
```

Enter a topic name (e.g., "React Documentation", "Company Policies") and optional description.

### 2. Add Documents

```
Ctrl+Shift+P > RAG: Add Document to Topic
```

Select a topic, then choose a PDF, Markdown, or HTML file. The extension will:
- Extract text from the document
- Split it into chunks
- Generate embeddings using the local model (downloaded on first use)
- Store everything in the vector database

### 3. Query via Copilot

Once documents are added, you can ask Copilot questions about your topics:

```
"Using the RAG query tool, search the 'React Documentation' topic for information about hooks"
```

Copilot will use the `ragQuery` tool to find relevant content with full heading context (e.g., "React Hooks → useState → Basic Usage") and provide accurate, contextual answers.

## Configuration

Open VSCode settings (`Cmd/Ctrl + ,`) and search for "RAGnarōk". Settings are organized by complexity.

📖 **[Full Configuration Guide](./CONFIGURATION_GUIDE.md)** - Detailed explanation of all settings, hierarchy, and troubleshooting.

### 🔧 Basic Settings (Always Active)

| Setting | Default | Description |
|---------|---------|-------------|
| `ragnarok.embeddingModel` | `Xenova/all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `ragnarok.topK` | `5` | Number of top results to return from queries |
| `ragnarok.chunkSize` | `512` | Maximum size of text chunks (characters) |
| `ragnarok.chunkOverlap` | `50` | Overlap between chunks (characters) |
| `ragnarok.pdfStructureDetection` | `heuristic` | PDF heading detection: "heuristic" or "none" |

### 🤝 Agentic RAG Settings

**Primary Control:**

| Setting | Default | When to Enable |
|---------|---------|----------------|
| `ragnarok.useAgenticMode` | `false` | ✅ Enable for complex queries, comparisons, multi-part questions<br>❌ Keep disabled for simple lookups (faster) |

**Agentic Behavior** _(only applies when Agentic Mode enabled)_:

| Setting | Default | Description |
|---------|---------|-------------|
| `agenticMaxIterations` | `3` | Max retrieval steps (1-10). Higher = more thorough |
| `agenticConfidenceThreshold` | `0.7` | Confidence to stop (0-1). Lower = more iterations |
| `agenticQueryDecomposition` | `true` | Break complex queries into sub-queries |
| `agenticIterativeRefinement` | `true` | Generate follow-up queries for gaps |
| `agenticRetrievalStrategy` | `hybrid` | **hybrid** (recommended) or **vector** only |

### 🤖 LLM-Powered Mode (Advanced)

**Primary Control:**

| Setting | Default | When to Enable | Requirements |
|---------|---------|----------------|--------------|
| `ragnarok.agenticUseLLM` | `false` | ✅ Enable for maximum intelligence<br>⚠️ 2-3x slower, uses Copilot quota | • GitHub Copilot subscription<br>• GitHub Copilot extension<br>• Agentic Mode enabled |

**LLM Behavior** _(only applies when LLM Mode enabled)_:

| Setting | Default | Description |
|---------|---------|-------------|
| `agenticLLMModel` | `gpt-4o` | Model to use: **gpt-4o** (best), **gpt-4o-mini** (balanced), **gpt-3.5-turbo** (fastest) |
| `agenticIncludeWorkspaceContext` | `true` | Include selected code, active file, imports (like #codebase) |

### 📊 Configuration Decision Tree

```
Simple queries (e.g., "What is useState?")
└─> Keep defaults (all disabled) ⚡ Fastest

Complex queries (e.g., "Compare X and Y")
└─> Enable useAgenticMode ✨ Multi-step retrieval
    ├─> Keep agenticUseLLM = false → Rule-based (fast)
    └─> Enable agenticUseLLM = true → AI-powered (intelligent)
        └─> Keep agenticIncludeWorkspaceContext = true
            → Workspace-aware queries 🎯
```

### 🎯 Recommended Configurations

**Beginner / Speed-Focused:**
```json
{
  "ragnarok.useAgenticMode": false
  // Simple mode: Fast single-shot retrieval
}
```

**Balanced / Recommended:**
```json
{
  "ragnarok.useAgenticMode": true,
  "ragnarok.agenticUseLLM": false,
  "ragnarok.agenticRetrievalStrategy": "hybrid"
  // Multi-step with heuristics: Good balance
}
```

**Maximum Intelligence:**
```json
{
  "ragnarok.useAgenticMode": true,
  "ragnarok.agenticUseLLM": true,
  "ragnarok.agenticLLMModel": "gpt-4o",
  "ragnarok.agenticIncludeWorkspaceContext": true
  // AI-powered + workspace context: Smartest, slowest
}
```

**Economical / Fast LLM:**
```json
{
  "ragnarok.useAgenticMode": true,
  "ragnarok.agenticUseLLM": true,
  "ragnarok.agenticLLMModel": "gpt-3.5-turbo"
  // LLM-powered but faster and cheaper
}
```

### Available Embedding Models

**For document embeddings (local, always used):**

- `Xenova/all-MiniLM-L6-v2` (Default) - Fast and efficient
- `Xenova/all-MiniLM-L12-v2` - Better quality, slower
- `Xenova/paraphrase-MiniLM-L6-v2` - Good for paraphrasing
- `Xenova/multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A

Models are downloaded automatically on first use and cached locally.

### LLM Models (for Agentic RAG)

**For query planning and evaluation (Copilot, only when LLM mode enabled):**

| Model | Speed | Quality | Cost | Use Case |
|-------|-------|---------|------|----------|
| **gpt-4o** | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Complex reasoning, best quality |
| **gpt-4o-mini** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Balanced, recommended |
| **gpt-3.5-turbo** | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | Fast, economical |

**Choose:**
- `gpt-4o` for maximum intelligence (research, complex queries)
- `gpt-4o-mini` for balanced performance (recommended default)
- `gpt-3.5-turbo` for speed and cost savings (simple queries)

## Commands

| Command | Description |
|---------|-------------|
| `RAG: Create New Topic` | Create a new topic for organizing documents |
| `RAG: Delete Topic` | Delete a topic and all its documents |
| `RAG: List All Topics` | Show all available topics |
| `RAG: Add Document to Topic` | Add a PDF, Markdown, or HTML document |
| `RAG: Refresh Topics` | Refresh the topics tree view |
| `RAG: Clear Model Cache` | Clear the embedding model cache |
| `RAG: Clear Database` | Clear the entire vector database |

## LLM Tool API

The extension registers a language model tool called `ragQuery` that can be used by Copilot or other LLM agents.

### Tool Schema

```typescript
{
  name: "ragQuery",
  parameters: {
    topic: string,                    // Topic name to search within
    query: string,                    // Search query/question
    topK?: number,                    // Number of results (optional)
    useAgenticMode?: boolean,         // Enable agentic mode (optional)
    agenticConfig?: {                 // Agentic configuration (optional)
      maxIterations?: number,         // Max retrieval steps (1-10)
      confidenceThreshold?: number,   // Confidence threshold (0-1)
      enableQueryDecomposition?: boolean,
      enableIterativeRefinement?: boolean,
      retrievalStrategy?: 'vector' | 'hybrid'
    }
  }
}
```

### Example Tool Usage

When you ask Copilot a question like:
> "What does the React documentation say about useEffect?"

Copilot can internally call:
```javascript
ragQuery({
  topic: "React Documentation",
  query: "useEffect hook usage and examples"
})
```

The tool returns:
```javascript
{
  query: "useEffect hook usage and examples",
  topicName: "React Documentation",
  results: [
    {
      text: "useEffect is a React Hook that lets you...",
      documentName: "hooks-reference.md",
      similarity: 0.89,
      metadata: {
        chunkIndex: 3,
        position: "chars 1536-2048"
      }
    },
    // ... more results
  ]
}
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     VSCode Extension                     │
├──────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Commands  │  │  Tree View   │  │    RAG Tool     │    │
│  └─────┬─────┘  └──────┬───────┘  └────────┬────────┘    │
│        │               │                   │             │
│        │               │        ┌──────────┴─────────┐   │
│        │               │        │ Agent Orchestrator │   │
│        │               │        │  ┌───────────────┐ │   │
│        │               │        │  │ QueryPlanner  │ │   │
│        │               │        │  │ ResultEval    │ │   │
│        │               │        │  │ Strategies    │ │   │
│        │               │        │  └───────────────┘ │   │
│        │               │        └───────────┬────────┘   │
│  ┌─────┴───────────────┴────────────────────┴─────────┐  │
│  │            Vector Database Service                 │  │
│  │         (Per-Topic JSON File Storage)              │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ topics.json (index)                          │  │  │
│  │  │ topic-abc123.json (embeddings + metadata)    │  │  │
│  │  │ topic-def456.json (embeddings + metadata)    │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └─────────────────────────┬──────────────────────────┘  │
│                            │                             │
│  ┌─────────────────────────┴───────────────────────────┐ │
│  │               Embedding Service                     │ │
│  │               (Transformers.js)                     │ │
│  │         - Local sentence transformers               │ │
│  │         - Cosine similarity computation             │ │
│  └─────────────────────────────────────────────────────┘ │
│                            │                             │
│  ┌─────────────────────────┴───────────────────────────┐ │
│  │            Document Processor                       │ │
│  │         (PDF / Markdown / HTML)                     │ │
│  │    - Semantic chunking with heading context         │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## How It Works

1. **Document Processing**: When you add a document:
   - All documents are converted to Markdown format
   - Heading hierarchy is parsed (e.g., # → ## → ###)
   - Content is split into semantic chunks based on sections
   - Large sections are split with smart boundaries and overlap

2. **Embedding Generation**: Each chunk is converted to a vector embedding using a sentence transformer model running locally via transformers.js.

3. **Storage**: Embeddings and metadata (including heading paths) are stored in per-topic JSON files in VSCode's extension storage directory. Each topic gets its own file for efficient loading and storage.

4. **Query Processing**: When queried (via the LLM tool or directly):
   - The query is converted to an embedding
   - Cosine similarity is calculated against all chunks in the topic
   - Top-K most similar chunks are returned with heading context

5. **LLM Integration**: Copilot or other agents receive relevant context with hierarchical structure (e.g., "Memory Management → Malloc → Usage") and can provide informed, contextual answers.

## Performance Considerations

- **Model Download**: First use requires downloading the model (~100MB for default model). Subsequent uses are instant.
- **Embedding Speed**: ~10-50 chunks/second depending on hardware (CPU-based)
- **Storage**: Each topic stored in its own JSON file for efficient access
- **Memory**: Models use ~500MB RAM when active. Only loaded topics consume additional memory.
- **Scalability**: Per-topic files mean better performance - only loads the data you're querying

## Troubleshooting

### Model not loading
- Check your internet connection (required for first download)
- Try clearing the cache: `RAG: Clear Model Cache`
- Restart VSCode

### Documents not being added
- Ensure the file is a valid PDF, Markdown, or HTML file
- Check that the file is readable
- Try with a smaller document first

### PDF headings not detected correctly
- **Best solution**: Convert PDF to Markdown first
  ```bash
  pandoc input.pdf -o output.md
  ```
- **Alternative**: Disable heuristics and use plain text chunking
  ```json
  { "ragnarok.pdfStructureDetection": "none" }
  ```

### Copilot not using the RAG tool
- Ensure you have Copilot enabled
- Explicitly mention the tool in your prompt
- Verify topics have documents: `RAG: List All Topics`

## Development

### Building

```bash
npm install
npm run compile
```

### Testing

```bash
npm run lint
npm run test
```

### Packaging

```bash
npm run package
vsce package
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Acknowledgments

- [Transformers.js](https://github.com/huggingface/transformers.js) for local ML inference
- [Hugging Face](https://huggingface.co/) for sentence transformer models
- VSCode Language Model API for LLM tool integration

## Roadmap

### ✅ Completed (v0.0.4)
- [x] Agentic RAG with query planning
- [x] Multi-step retrieval and iterative refinement
- [x] Hybrid search (semantic + keyword)
- [x] Result quality evaluation
- [x] Query complexity analysis

### 🚧 In Progress / Planned
- [ ] Support for more document formats (DOCX, TXT)
- [ ] Batch document upload
- [ ] Export/import topics
- [ ] MMR (Maximal Marginal Relevance) exposed in UI
- [ ] Cross-encoder re-ranking
- [ ] Advanced search filters
- [ ] Similarity threshold configuration
- [ ] Custom chunking strategies
- [ ] Metadata filtering
- [ ] Document versioning
- [ ] LLM-powered query expansion
- [ ] Learning from user feedback
