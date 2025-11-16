#!/usr/bin/env node

/**
 * Inspect LanceDB chunks created by the RAGnarōk VS Code extension.
 *
 * This script connects to the extension's embedded LanceDB instance and
 * prints chunk metadata plus a short text preview for quick inspection.
 *
 * Usage:
 *   node scripts/inspect-chunks.js [--dbDir <path>] [--topic <table>] [--limit <n>] [--preview <chars>] [--json <out>]
 *
 * Notes:
 * - By default the script looks for the LanceDB folder under the standard
 *   VS Code globalStorage paths (Code, Code - Insiders, or VSCodium).
 * - You can override the path with --dbDir or the RAGNAROK_DB_DIR env var.
 */

const fs = require("fs");
const path = require("path");
const os = require("os");
const { connect } = require("@lancedb/lancedb");

const EXTENSION_ID = "hyorman.ragnarok";
const DEFAULT_TEXT_COLUMNS = ["text", "pageContent", "content"];
const DEFAULT_DB_DIR = path.join(
  os.homedir(),
  "Library",
  "Application Support",
  "Code",
  "User",
  "globalStorage",
  EXTENSION_ID,
  "database"
);

run().catch((error) => {
  console.error(`\n✖ ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});

async function run() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  const quiet = args.quiet ?? Boolean(args.jsonOut);

  const dbPath = resolveDbPath(args.dbDir || process.env.RAGNAROK_DB_DIR);
  const baseDir = getDatabaseBaseDir(dbPath);
  log(quiet, `Connecting to LanceDB at: ${dbPath}\n`);

  const db = await connect(dbPath);
  const tableNames = await db.tableNames();
  const topicsIndex = await loadTopicsIndex(baseDir);

  if (tableNames.length === 0) {
    log(quiet, "No tables found. Add documents in the extension first.");
    return;
  }

  const filteredTables = filterTables(tableNames, args.topic, topicsIndex);

  if (filteredTables.length === 0) {
    const availableTopics = Object.entries(topicsIndex || {})
      .map(([id, topic]) => `${topic?.name || "unknown"} (${id})`)
      .join(", ");
    log(quiet, `No table matched filter "${args.topic}". Available tables: ${tableNames.join(", ")}`);
    if (availableTopics) {
      log(quiet, `Available topic names: ${availableTopics}`);
    }
    return;
  }

  if (!args.jsonOut) {
    for (const tableName of filteredTables) {
      await inspectTable(db, tableName, args.limit, args.preview, quiet);
    }
  }

  if (args.jsonOut) {
    const exportPath = resolveOutputPath(args.jsonOut);
    const payload = [];

    for (const tableName of filteredTables) {
      const rows = await exportTableRows(db, tableName);
      payload.push({ table: tableName, rows });
    }

    await fs.promises.writeFile(exportPath, JSON.stringify(payload, null, 2), "utf-8");
    log(quiet, `Saved ${payload.reduce((sum, t) => sum + t.rows.length, 0)} chunks to ${exportPath}`);
  }
}

function parseArgs(argv) {
  const args = {
    help: false,
    dbDir: DEFAULT_DB_DIR,
    topic: undefined,
    limit: undefined,
    preview: 180,
    jsonOut: undefined,
    quiet: undefined,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i];
    switch (value) {
      case "--help":
      case "-h":
        args.help = true;
        break;
      case "--dbDir":
      case "--db":
        args.dbDir = argv[i + 1];
        i += 1;
        break;
      case "--topic":
        args.topic = argv[i + 1];
        i += 1;
        break;
      case "--limit":
        args.limit = parseNumber(argv[i + 1], args.limit);
        i += 1;
        break;
      case "--preview":
        args.preview = parseNumber(argv[i + 1], args.preview);
        i += 1;
        break;
      case "--json":
      case "--jsonOut":
        args.jsonOut = argv[i + 1];
        i += 1;
        break;
      case "--quiet":
        args.quiet = true;
        break;
      default:
        console.warn(`Ignoring unknown argument: ${value}`);
        break;
    }
  }

  return args;
}

function printHelp() {
  console.log(`Inspect LanceDB chunks created by the RAGnarōk extension.

Options:
  --dbDir <path>   Override database directory (default: auto-detect globalStorage)
  --topic <list>   Filter by topic/table name(s). Comma-separated, exact name or id preferred (e.g. topic-123,topic-456 or "Project X")
  --limit <n>      Rows to show per table (default: all)
  --preview <n>    Max characters for text preview (default: 180)
  --json <path>    Export all matching chunks to a JSON file
  --quiet          Suppress console output (enabled automatically when --json is used)
  -h, --help       Show this help

Env:
  RAGNAROK_DB_DIR  Same as --dbDir

Examples:
  node scripts/inspect-chunks.js
  node scripts/inspect-chunks.js --topic topic-123 --limit 10
  node scripts/inspect-chunks.js --dbDir "~/Library/Application Support/Code/User/globalStorage/${EXTENSION_ID}/database/lancedb"
  node scripts/inspect-chunks.js --json ./chunks.json
`);
}

function resolveDbPath(inputPath) {
  const candidates = [];

  if (inputPath) {
    const expanded = expandHome(inputPath);
    candidates.push(ensureLanceSubdir(expanded));
  }

  candidates.push(...buildDefaultPaths().map(ensureLanceSubdir));

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }

  const attempted = candidates.filter(Boolean).join("\n- ");
  throw new Error(
    `Could not find LanceDB folder. Provide --dbDir or set RAGNAROK_DB_DIR.\nTried:\n- ${attempted}`
  );
}

function buildDefaultPaths() {
  const home = os.homedir();
  const appNames = ["Code", "Code - Insiders", "VSCodium"];

  switch (process.platform) {
    case "darwin":
      return appNames.map((app) =>
        path.join(home, "Library", "Application Support", app, "User", "globalStorage", EXTENSION_ID)
      );
    case "win32":
      return appNames.map((app) =>
        path.join(home, "AppData", "Roaming", app, "User", "globalStorage", EXTENSION_ID)
      );
    default:
      return appNames.map((app) =>
        path.join(home, ".config", app, "User", "globalStorage", EXTENSION_ID)
      );
  }
}

function ensureLanceSubdir(baseDir) {
  if (!baseDir) return undefined;
  if (path.basename(baseDir) === "lancedb") {
    return baseDir;
  }
  const candidate = path.join(baseDir, "lancedb");
  return candidate;
}

function expandHome(p) {
  if (p.startsWith("~/")) {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
}

function pickTextColumn(columns) {
  for (const candidate of DEFAULT_TEXT_COLUMNS) {
    if (columns.includes(candidate)) {
      return candidate;
    }
  }
  // Fallback: pick the first non-vector column
  return columns.find((c) => c !== "vector" && c !== "score" && c !== "_rowid") || "text";
}

function parseNumber(raw, fallback) {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatValue(value) {
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

async function exportTableRows(db, tableName) {
  const table = await db.openTable(tableName);
  const schema = await table.schema();
  const columnNames = schema.fields?.map((f) => f.name) || [];
  const textColumn = pickTextColumn(columnNames);
  const excluded = new Set(["vector", "score", textColumn, "_rowid"]);
  const metadataColumns = columnNames.filter((col) => !excluded.has(col));
  const selectedColumns = [textColumn, ...metadataColumns];

  const rows = await table.query().select(selectedColumns).toArray();

  return rows.map((row) => {
    const text = typeof row[textColumn] === "string" ? row[textColumn] : "";
    const metadata = {};
    metadataColumns.forEach((col) => {
      if (row[col] !== undefined) {
        metadata[col] = row[col];
      }
    });
    return {
      table: tableName,
      text,
      textLength: text.length,
      metadata,
    };
  });
}

function resolveOutputPath(outPath) {
  const expanded = expandHome(outPath);
  return path.isAbsolute(expanded)
    ? expanded
    : path.join(process.cwd(), expanded);
}

function filterTables(tableNames, topicArg, topicsIndex) {
  if (!topicArg) return tableNames;

  const raw = topicArg.split(",").map((t) => t.trim()).filter(Boolean);
  if (raw.length === 0) return tableNames;

  const topicEntries = Object.entries(topicsIndex || {}).map(([id, topic]) => ({
    id,
    name: topic?.name || "",
  }));

  const matches = new Set();

  for (const token of raw) {
    const lowerToken = token.toLowerCase();

    // Exact table id match
    const exactId = tableNames.find((name) => name.toLowerCase() === lowerToken);
    if (exactId) {
      matches.add(exactId);
      continue;
    }

    // Exact topic name match -> map to id
    const exactTopic = topicEntries.find((t) => t.name.toLowerCase() === lowerToken);
    if (exactTopic && tableNames.includes(exactTopic.id)) {
      matches.add(exactTopic.id);
      continue;
    }
  }

  if (matches.size > 0) {
    return Array.from(matches);
  }

  // Fallback: substring match against table ids and topic names
  for (const token of raw) {
    const lowerToken = token.toLowerCase();

    tableNames.forEach((name) => {
      if (name.toLowerCase().includes(lowerToken)) {
        matches.add(name);
      }
    });

    topicEntries.forEach((topic) => {
      if (topic.name.toLowerCase().includes(lowerToken) && tableNames.includes(topic.id)) {
        matches.add(topic.id);
      }
    });
  }

  return Array.from(matches);
}

async function loadTopicsIndex(baseDir) {
  try {
    const topicsPath = path.join(baseDir, "topics.json");
    const data = await fs.promises.readFile(topicsPath, "utf-8");
    const parsed = JSON.parse(data);
    return parsed?.topics || {};
  } catch {
    return {};
  }
}

function getDatabaseBaseDir(dbPath) {
  if (path.basename(dbPath) === "lancedb") {
    return path.dirname(dbPath);
  }
  return dbPath;
}

async function inspectTable(db, tableName, limit, preview, quiet) {
  const table = await db.openTable(tableName);
  const rowCount = await table.countRows().catch(() => 0);
  const schema = await table.schema();
  const columnNames = schema.fields?.map((f) => f.name) || [];

  const textColumn = pickTextColumn(columnNames);
  const excluded = new Set(["vector", "score", textColumn, "_rowid"]);
  const metadataColumns = columnNames.filter((col) => !excluded.has(col));

  log(quiet, `=== ${tableName} (${rowCount} chunks) ===`);
  log(quiet, `Text column: ${textColumn}`);
  log(quiet, `Metadata columns: ${metadataColumns.length ? metadataColumns.join(", ") : "none"}`);

  const selectedColumns = [textColumn, ...metadataColumns];
  let query = table.query().select(selectedColumns);
  if (typeof limit === "number" && Number.isFinite(limit)) {
    query = query.limit(limit);
  }
  const rows = await query.toArray();

  if (rows.length === 0) {
    log(quiet, "No rows to display.\n");
    return;
  }

  rows.forEach((row, idx) => {
    const text = typeof row[textColumn] === "string" ? row[textColumn] : "";
    const previewText = text.length > preview ? `${text.slice(0, preview)}...` : text;

    log(quiet, `\n[${idx + 1}/${rows.length}]`);
    metadataColumns.forEach((col) => {
      if (row[col] !== undefined) {
        log(quiet, `- ${col}: ${formatValue(row[col])}`);
      }
    });
    log(quiet, `- textLength: ${text.length}`);
    log(quiet, `- textPreview: ${previewText}`);
  });

  log(quiet, "\n");
}

function log(quiet, message) {
  if (!quiet) {
    console.log(message);
  }
}
