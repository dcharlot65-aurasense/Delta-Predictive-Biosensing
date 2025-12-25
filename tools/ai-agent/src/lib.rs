//! AI Agent Infrastructure for Delta-Predictive Biosensing Framework
//!
//! This crate provides tools for AI agents to interact with the DPB framework:
//!
//! - **API Schema Generation**: Extract machine-readable API schemas from rustdoc JSON
//! - **MCP Server**: Model Context Protocol server for AI tool invocation
//! - **Code Templates**: Structured usage patterns for code generation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Rust Crates  →  rustdoc JSON  →  Simplified API Schema (JSON) │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  AI Agent  ←→  MCP Server  ←→  dpb-core / dpb-snn / etc        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod schema;
pub mod mcp;
pub mod templates;

pub use schema::{ApiSchema, CrateSchema, TypeDef, FunctionDef};
pub use mcp::{McpServer, McpTool, ToolParameter};
pub use templates::{CodeTemplate, TemplateRegistry};
