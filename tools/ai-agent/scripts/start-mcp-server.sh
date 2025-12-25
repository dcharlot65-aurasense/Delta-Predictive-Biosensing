#!/bin/bash
# Start the DPB MCP Server
#
# Usage:
#   ./scripts/start-mcp-server.sh              # stdio mode (for Claude Desktop)
#   ./scripts/start-mcp-server.sh --http       # HTTP mode
#   ./scripts/start-mcp-server.sh --http --port 8080
#
# For Claude Desktop, add to claude_desktop_config.json:
# {
#   "mcpServers": {
#     "dpb": {
#       "command": "/path/to/tools/ai-agent/scripts/start-mcp-server.sh"
#     }
#   }
# }

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$AI_AGENT_DIR/../.." && pwd)"

# Check if binary exists
BINARY="$AI_AGENT_DIR/target/release/dpb-mcp-server"
if [[ ! -f "$BINARY" ]]; then
    echo "Building MCP server..." >&2
    cd "$AI_AGENT_DIR"
    cargo build --release --bin dpb-mcp-server 2>/dev/null
fi

# Check for schema file
SCHEMA_FILE="$AI_AGENT_DIR/output/api-schema.json"
SCHEMA_ARG=""
if [[ -f "$SCHEMA_FILE" ]]; then
    SCHEMA_ARG="--schema $SCHEMA_FILE"
fi

# Run the server
exec "$BINARY" $SCHEMA_ARG "$@"
