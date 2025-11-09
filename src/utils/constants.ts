/**
 * Central constants for the RAGnarōk extension
 * All identifiers, command names, and configuration keys are defined here
 */

/**
 * Extension identifiers
 */
export const EXTENSION = {
  ID: "ragnarok",
  DISPLAY_NAME: "RAGnarōk",
  DATABASE_DIR: "database",
  TOPICS_INDEX_FILENAME: "topics.json",
} as const;

/**
 * Configuration keys
 */
export const CONFIG = {
  ROOT: "ragnarok",
  // Basic configuration
  EMBEDDING_MODEL: "embeddingModel",
  TOP_K: "topK",
  CHUNK_SIZE: "chunkSize",
  CHUNK_OVERLAP: "chunkOverlap",
  LOG_LEVEL: "logLevel",
  RETRIEVAL_STRATEGY: "retrievalStrategy",
  // Agentic RAG configuration
  USE_AGENTIC_MODE: "useAgenticMode",
  AGENTIC_MAX_ITERATIONS: "agenticMaxIterations",
  AGENTIC_CONFIDENCE_THRESHOLD: "agenticConfidenceThreshold",
  AGENTIC_ITERATIVE_REFINEMENT: "agenticIterativeRefinement",
  AGENTIC_USE_LLM: "agenticUseLLM",
  AGENTIC_LLM_MODEL: "agenticLLMModel",
  AGENTIC_INCLUDE_WORKSPACE: "agenticIncludeWorkspaceContext",
} as const;

/**
 * Command identifiers
 */
export const COMMANDS = {
  CREATE_TOPIC: "ragnarok.createTopic",
  DELETE_TOPIC: "ragnarok.deleteTopic",
  LIST_TOPICS: "ragnarok.listTopics",
  ADD_DOCUMENT: "ragnarok.addDocument",
  ADD_GITHUB_REPO: "ragnarok.addGithubRepo",
  REFRESH_TOPICS: "ragnarok.refreshTopics",
  CLEAR_MODEL_CACHE: "ragnarok.clearModelCache",
  CLEAR_DATABASE: "ragnarok.clearDatabase",
  // GitHub token management
  ADD_GITHUB_TOKEN: "ragnarok.addGithubToken",
  LIST_GITHUB_TOKENS: "ragnarok.listGithubTokens",
  REMOVE_GITHUB_TOKEN: "ragnarok.removeGithubToken",
} as const;

/**
 * View identifiers
 */
export const VIEWS = {
  RAG_TOPICS: "ragTopics",
} as const;

/**
 * Global state keys
 */
export const STATE = {
  HAS_SHOWN_WELCOME: "ragnarok.hasShownWelcome",
} as const;

/**
 * Tool identifiers
 */
export const TOOLS = {
  RAG_QUERY: "ragQuery",
} as const;
