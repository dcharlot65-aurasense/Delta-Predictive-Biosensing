//! Model Context Protocol (MCP) Server
//!
//! Exposes DPB framework APIs as MCP tools that AI agents can call directly.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  AI Agent  ←→  MCP Server (JSON-RPC)  ←→  DPB Framework        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each function becomes a tool with typed parameters and descriptions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;

use crate::schema::{ApiSchema, TypeDef};

/// MCP Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Enable CORS
    pub cors_enabled: bool,
    /// Allowed origins for CORS
    pub allowed_origins: Vec<String>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            name: "dpb-mcp-server".to_string(),
            version: "0.1.0".to_string(),
            host: "127.0.0.1".to_string(),
            port: 3000,
            cors_enabled: true,
            allowed_origins: vec!["*".to_string()],
        }
    }
}

/// MCP Server instance
pub struct McpServer {
    config: McpConfig,
    tools: Vec<McpTool>,
    schema: Option<Arc<ApiSchema>>,
}

impl McpServer {
    /// Create a new MCP server with the given configuration
    pub fn new(config: McpConfig) -> Self {
        Self {
            config,
            tools: Vec::new(),
            schema: None,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(McpConfig::default())
    }

    /// Load API schema and generate tools from it
    pub fn with_schema(mut self, schema: ApiSchema) -> Self {
        self.tools = generate_tools_from_schema(&schema);
        self.schema = Some(Arc::new(schema));
        self
    }

    /// Get all registered tools
    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    /// Get server info for MCP protocol
    pub fn server_info(&self) -> ServerInfo {
        ServerInfo {
            name: self.config.name.clone(),
            version: self.config.version.clone(),
        }
    }

    /// Get capabilities for MCP protocol
    pub fn capabilities(&self) -> ServerCapabilities {
        ServerCapabilities {
            tools: ToolsCapability {
                list_changed: Some(false),
            },
            resources: None,
            prompts: None,
        }
    }

    /// List all tools (MCP tools/list response)
    pub fn list_tools(&self) -> ListToolsResult {
        ListToolsResult {
            tools: self.tools.clone(),
        }
    }

    /// Call a tool by name
    pub async fn call_tool(&self, name: &str, _arguments: serde_json::Value) -> Result<CallToolResult> {
        let tool = self.tools.iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", name))?;

        // For now, return tool documentation as the result
        // In a full implementation, this would actually execute the tool
        let content = vec![ToolResultContent::Text {
            text: format!(
                "Tool: {}\nDescription: {}\n\nThis is a documentation tool. To execute the actual function, use the Rust library directly.\n\nSignature: {}",
                tool.name,
                tool.description,
                tool.input_schema.properties.get("signature")
                    .and_then(|s| s.get("const"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("N/A")
            ),
        }];

        Ok(CallToolResult {
            content,
            is_error: Some(false),
        })
    }

    /// Generate MCP manifest for Claude Desktop configuration
    pub fn generate_manifest(&self) -> McpManifest {
        McpManifest {
            name: self.config.name.clone(),
            version: self.config.version.clone(),
            description: "Delta-Predictive Biosensing Framework MCP Server".to_string(),
            tools: self.tools.iter().map(|t| ManifestTool {
                name: t.name.clone(),
                description: t.description.clone(),
            }).collect(),
        }
    }
}

/// Generate MCP tools from API schema
fn generate_tools_from_schema(schema: &ApiSchema) -> Vec<McpTool> {
    let mut tools = Vec::new();

    // Add schema introspection tools
    tools.push(McpTool {
        name: "dpb_list_crates".to_string(),
        description: "List all available DPB crates and their descriptions".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
        },
    });

    tools.push(McpTool {
        name: "dpb_search_types".to_string(),
        description: "Search for types (structs, enums, traits) by name pattern".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("pattern".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Search pattern (case-insensitive)"
                })),
                ("crate_filter".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Optional: filter to specific crate (e.g., 'dpb-core')"
                })),
            ].into_iter().collect(),
            required: vec!["pattern".to_string()],
        },
    });

    tools.push(McpTool {
        name: "dpb_search_functions".to_string(),
        description: "Search for functions by name pattern".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("pattern".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Search pattern (case-insensitive)"
                })),
                ("crate_filter".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Optional: filter to specific crate"
                })),
            ].into_iter().collect(),
            required: vec!["pattern".to_string()],
        },
    });

    tools.push(McpTool {
        name: "dpb_get_type_info".to_string(),
        description: "Get detailed information about a specific type".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("type_path".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Full type path (e.g., 'dpb_core::signal::ecg::RpeakDetector')"
                })),
            ].into_iter().collect(),
            required: vec!["type_path".to_string()],
        },
    });

    tools.push(McpTool {
        name: "dpb_get_function_info".to_string(),
        description: "Get detailed information about a specific function".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("function_path".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Full function path"
                })),
            ].into_iter().collect(),
            required: vec!["function_path".to_string()],
        },
    });

    tools.push(McpTool {
        name: "dpb_get_module_tree".to_string(),
        description: "Get the module hierarchy for a crate".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("crate_name".to_string(), serde_json::json!({
                    "type": "string",
                    "description": "Crate name (e.g., 'dpb-core')"
                })),
            ].into_iter().collect(),
            required: vec!["crate_name".to_string()],
        },
    });

    // Add tools for each major type category
    for (crate_name, crate_schema) in &schema.crates {
        // Add encoder tools
        if crate_name == "dpb-encoders" {
            for ty in &crate_schema.types {
                if ty.name.ends_with("Encoder") {
                    tools.push(create_encoder_tool(crate_name, ty));
                }
            }
        }

        // Add neuron model tools
        if crate_name == "dpb-neurons" {
            for ty in &crate_schema.types {
                if ty.name.ends_with("Neuron") {
                    tools.push(create_neuron_tool(crate_name, ty));
                }
            }
        }

        // Add SNN architecture tools
        if crate_name == "dpb-snn" {
            for ty in &crate_schema.types {
                if ty.name.ends_with("SNN") || ty.name.ends_with("Network") {
                    tools.push(create_snn_tool(crate_name, ty));
                }
            }
        }

        // Add generator tools
        if crate_name == "dpb-synth" {
            for ty in &crate_schema.types {
                if ty.name.ends_with("Generator") {
                    tools.push(create_generator_tool(crate_name, ty));
                }
            }
        }
    }

    tools
}

fn create_encoder_tool(_crate_name: &str, ty: &TypeDef) -> McpTool {
    let description = ty.docs.clone().unwrap_or_else(|| {
        format!("Encoder for converting signals to spike events: {}", ty.name)
    });

    McpTool {
        name: format!("dpb_encoder_{}", to_snake_case(&ty.name)),
        description: truncate_description(&description),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("action".to_string(), serde_json::json!({
                    "type": "string",
                    "enum": ["info", "example", "config"],
                    "description": "Action to perform: 'info' for documentation, 'example' for usage example, 'config' for configuration options"
                })),
            ].into_iter().collect(),
            required: vec!["action".to_string()],
        },
    }
}

fn create_neuron_tool(_crate_name: &str, ty: &TypeDef) -> McpTool {
    let description = ty.docs.clone().unwrap_or_else(|| {
        format!("Neuron model: {}", ty.name)
    });

    McpTool {
        name: format!("dpb_neuron_{}", to_snake_case(&ty.name)),
        description: truncate_description(&description),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("action".to_string(), serde_json::json!({
                    "type": "string",
                    "enum": ["info", "example", "parameters"],
                    "description": "Action: 'info' for model description, 'example' for usage, 'parameters' for tunable parameters"
                })),
            ].into_iter().collect(),
            required: vec!["action".to_string()],
        },
    }
}

fn create_snn_tool(_crate_name: &str, ty: &TypeDef) -> McpTool {
    let description = ty.docs.clone().unwrap_or_else(|| {
        format!("SNN architecture: {}", ty.name)
    });

    McpTool {
        name: format!("dpb_snn_{}", to_snake_case(&ty.name)),
        description: truncate_description(&description),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("action".to_string(), serde_json::json!({
                    "type": "string",
                    "enum": ["info", "example", "layers"],
                    "description": "Action: 'info' for architecture details, 'example' for usage, 'layers' for layer configuration"
                })),
            ].into_iter().collect(),
            required: vec!["action".to_string()],
        },
    }
}

fn create_generator_tool(_crate_name: &str, ty: &TypeDef) -> McpTool {
    let description = ty.docs.clone().unwrap_or_else(|| {
        format!("Synthetic signal generator: {}", ty.name)
    });

    McpTool {
        name: format!("dpb_synth_{}", to_snake_case(&ty.name)),
        description: truncate_description(&description),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: vec![
                ("action".to_string(), serde_json::json!({
                    "type": "string",
                    "enum": ["info", "example", "params"],
                    "description": "Action: 'info' for generator details, 'example' for usage, 'params' for configurable parameters"
                })),
            ].into_iter().collect(),
            required: vec!["action".to_string()],
        },
    }
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

fn truncate_description(s: &str) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    if first_line.len() > 200 {
        format!("{}...", &first_line[..197])
    } else {
        first_line.to_string()
    }
}

// ============================================================================
// MCP Protocol Types
// ============================================================================

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Tool name (must be unique)
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    #[serde(rename = "inputSchema")]
    pub input_schema: ToolInputSchema,
}

/// Tool input schema (JSON Schema format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// Schema type (always "object" for MCP tools)
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Property definitions
    pub properties: HashMap<String, serde_json::Value>,
    /// Required properties
    pub required: Vec<String>,
}

/// Tool parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Parameter description
    pub description: String,
    /// Whether required
    pub required: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Enum values (if applicable)
    pub enum_values: Option<Vec<String>>,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
    pub resources: Option<serde_json::Value>,
    pub prompts: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

/// Result of tools/list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    pub tools: Vec<McpTool>,
}

/// Result of tools/call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolResult {
    pub content: Vec<ToolResultContent>,
    #[serde(rename = "isError")]
    pub is_error: Option<bool>,
}

/// Content types for tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolResultContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { uri: String, mime_type: Option<String>, text: Option<String> },
}

/// MCP manifest for Claude Desktop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub tools: Vec<ManifestTool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestTool {
    pub name: String,
    pub description: String,
}

// ============================================================================
// JSON-RPC Types
// ============================================================================

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, code: i32, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
                data: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_server_creation() {
        let server = McpServer::default_config();
        assert_eq!(server.config.name, "dpb-mcp-server");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("EcgRPeakEncoder"), "ecg_r_peak_encoder");
        assert_eq!(to_snake_case("LifNeuron"), "lif_neuron");
    }
}
