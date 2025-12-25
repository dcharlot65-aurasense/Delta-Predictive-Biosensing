# DPB AI Agent Infrastructure

This package provides infrastructure for AI agents to interact with the Delta-Predictive Biosensing (DPB) Framework.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Rust Crates  →  rustdoc JSON  →  Simplified API Schema (JSON) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  AI Agent  ←→  MCP Server  ←→  dpb-core / dpb-snn / etc        │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. API Schema Generator (`dpb-schema-gen`)

Generates machine-readable API schemas from the DPB framework.

```bash
# Generate rustdoc JSON (requires nightly)
cargo +nightly rustdoc -p dpb-core -- -Z unstable-options --output-format json

# Process into simplified schema
cargo run --bin dpb-schema-gen -- --all --output api-schema.json

# Generate OpenAPI 3.0 spec
cargo run --bin dpb-schema-gen -- --all --openapi --output openapi.json

# Include code templates
cargo run --bin dpb-schema-gen -- --all --templates --output api-schema.json
```

**Output Format:**
```json
{
  "schema_version": "1.0.0",
  "framework_version": "0.1.0",
  "crates": {
    "dpb-core": {
      "name": "dpb-core",
      "version": "0.1.0",
      "docs": "Core signal processing...",
      "types": [...],
      "functions": [...],
      "modules": [...]
    }
  }
}
```

### 2. MCP Server (`dpb-mcp-server`)

Model Context Protocol server that exposes DPB APIs as tools for AI agents.

```bash
# Start in stdio mode (for Claude Desktop)
cargo run --bin dpb-mcp-server

# Start with HTTP transport
cargo run --bin dpb-mcp-server -- --http --port 3000

# With custom schema
cargo run --bin dpb-mcp-server -- --schema api-schema.json
```

**Available Tools:**

| Tool | Description |
|------|-------------|
| `dpb_list_crates` | List all DPB crates and descriptions |
| `dpb_search_types` | Search for types by name pattern |
| `dpb_search_functions` | Search for functions by name |
| `dpb_get_type_info` | Get detailed type information |
| `dpb_get_module_tree` | Get module hierarchy for a crate |
| `dpb_encoder_*` | Encoder-specific documentation |
| `dpb_neuron_*` | Neuron model documentation |
| `dpb_snn_*` | SNN architecture documentation |
| `dpb_synth_*` | Generator documentation |

### 3. Code Templates

Structured templates for common usage patterns.

```json
{
  "encoding": [
    {
      "id": "basic_level_crossing",
      "name": "Basic Level Crossing Encoder",
      "code": "let encoder = LevelCrossingEncoder::new({{THRESHOLD}});...",
      "placeholders": [{"name": "THRESHOLD", "type_hint": "f64", ...}],
      "example": "..."
    }
  ],
  "training": [...],
  "inference": [...],
  "pipeline": [...]
}
```

## Claude Desktop Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dpb": {
      "command": "/path/to/dpb-mcp-server",
      "args": []
    }
  }
}
```

Or using the shell script:

```json
{
  "mcpServers": {
    "dpb": {
      "command": "/path/to/tools/ai-agent/scripts/start-mcp-server.sh"
    }
  }
}
```

## Quick Start

```bash
# 1. Generate the API schema
./scripts/generate-schema.sh

# 2. Start the MCP server
./scripts/start-mcp-server.sh

# 3. (Optional) Start HTTP server for development
./scripts/start-mcp-server.sh --http --port 3000
```

## API Schema Structure

### Types

```json
{
  "name": "LifNeuron",
  "path": "dpb_neurons::lif::LifNeuron",
  "kind": "struct",
  "docs": "Leaky Integrate-and-Fire neuron model",
  "generics": [],
  "fields": [
    {"name": "tau_m", "ty": "f64", "docs": "Membrane time constant"}
  ],
  "methods": [
    {"name": "new", "signature": "fn new(config: LifConfig) -> Self", ...}
  ],
  "traits": ["MembraneDynamics", "Clone", "Debug"]
}
```

### Functions

```json
{
  "name": "encode",
  "path": "dpb_core::traits::EventEncoder::encode",
  "docs": "Encode a time series signal to spike events",
  "signature": "fn encode(&self, signal: &TimeSeries) -> SpikeTrain",
  "params": [
    {"name": "signal", "ty": "&TimeSeries", "is_ref": true}
  ],
  "return_type": "SpikeTrain"
}
```

## Development

```bash
# Build all binaries
cargo build --release

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```

## Template Categories

| Category | Description |
|----------|-------------|
| `encoding` | Signal-to-spike encoding patterns |
| `training` | SNN training loops and distillation |
| `inference` | Model inference and clinical scoring |
| `signal_processing` | Filtering, preprocessing |
| `visualization` | Raster plots, dashboards |
| `pipeline` | Real-time streaming pipelines |

## License

MIT OR Apache-2.0
