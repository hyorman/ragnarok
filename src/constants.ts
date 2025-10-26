/**
 * Central constants for the RAGnarōk extension
 * All identifiers, command names, and configuration keys are defined here
 */

/**
 * Extension identifiers
 */
export const EXTENSION = {
  ID: 'ragnarok',
  DISPLAY_NAME: 'RAGnarōk',
  DATABASE_DIR: 'database',
  TOPICS_INDEX_FILENAME: 'topics.json',
} as const;

/**
 * Configuration keys
 */
export const CONFIG = {
  ROOT: 'ragnarok',
  EMBEDDING_MODEL: 'ragnarok.embeddingModel',
  TOP_K: 'ragnarok.topK',
  CHUNK_SIZE: 'ragnarok.chunkSize',
  CHUNK_OVERLAP: 'ragnarok.chunkOverlap',
  PDF_STRUCTURE_DETECTION: 'ragnarok.pdfStructureDetection',
} as const;

/**
 * Command identifiers
 */
export const COMMANDS = {
  CREATE_TOPIC: 'ragnarok.createTopic',
  DELETE_TOPIC: 'ragnarok.deleteTopic',
  LIST_TOPICS: 'ragnarok.listTopics',
  ADD_DOCUMENT: 'ragnarok.addDocument',
  REFRESH_TOPICS: 'ragnarok.refreshTopics',
  CLEAR_MODEL_CACHE: 'ragnarok.clearModelCache',
  CLEAR_DATABASE: 'ragnarok.clearDatabase',
} as const;

/**
 * View identifiers
 */
export const VIEWS = {
  RAG_TOPICS: 'ragTopics',
} as const;

/**
 * Global state keys
 */
export const STATE = {
  HAS_SHOWN_WELCOME: 'ragnarok.hasShownWelcome',
} as const;

/**
 * Tool identifiers
 */
export const TOOLS = {
  RAG_QUERY: 'ragQuery',
} as const;


