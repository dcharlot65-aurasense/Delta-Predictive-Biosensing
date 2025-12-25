//! API Schema Generator
//!
//! Generates machine-readable API schemas from rustdoc JSON output.
//!
//! # Usage
//!
//! ```bash
//! # Generate rustdoc JSON for all crates
//! cargo +nightly rustdoc -p dpb-core -- -Z unstable-options --output-format json
//!
//! # Process into simplified schema
//! cargo run --bin dpb-schema-gen -- --input target/doc/dpb_core.json --output api-schema.json
//!
//! # Generate for all crates
//! cargo run --bin dpb-schema-gen -- --all --output api-schema.json
//! ```

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use dpb_ai_agent::schema::{ApiSchema, CrateSchema, extract_crate_schema};
use dpb_ai_agent::templates::TemplateRegistry;

/// API Schema Generator for DPB Framework
#[derive(Parser, Debug)]
#[command(name = "dpb-schema-gen")]
#[command(about = "Generate machine-readable API schemas from rustdoc JSON")]
#[command(version)]
struct Args {
    /// Input rustdoc JSON file (use with single crate)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Process all crates in the workspace
    #[arg(long)]
    all: bool,

    /// Output file for the schema
    #[arg(short, long, default_value = "api-schema.json")]
    output: PathBuf,

    /// Also generate code templates
    #[arg(long)]
    templates: bool,

    /// Templates output file
    #[arg(long, default_value = "code-templates.json")]
    templates_output: PathBuf,

    /// Generate OpenAPI spec instead of custom schema
    #[arg(long)]
    openapi: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Workspace root directory
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .init();
    }

    println!("DPB API Schema Generator");
    println!("========================\n");

    let mut schema = ApiSchema::new("0.1.0");

    if args.all {
        // Process all crates in workspace
        println!("Processing all workspace crates...\n");
        schema = generate_all_crates(&args.workspace, args.verbose)?;
    } else if let Some(input) = &args.input {
        // Process single crate
        println!("Processing: {:?}\n", input);
        let crate_schema = extract_crate_schema(input)?;
        schema.crates.insert(crate_schema.name.clone(), crate_schema);
    } else {
        // Generate from source files directly (fallback)
        println!("No input specified. Generating schema from source analysis...\n");
        schema = generate_from_source(&args.workspace)?;
    }

    // Print summary
    println!("\nSchema Summary:");
    println!("--------------");
    println!("Crates: {}", schema.crates.len());
    for (name, crate_schema) in &schema.crates {
        println!("  {} v{}: {} types, {} functions",
            name,
            crate_schema.version,
            crate_schema.types.len(),
            crate_schema.functions.len()
        );
    }

    // Save schema
    if args.openapi {
        let openapi = generate_openapi(&schema);
        std::fs::write(&args.output, serde_json::to_string_pretty(&openapi)?)?;
        println!("\nOpenAPI spec saved to: {:?}", args.output);
    } else {
        schema.save(&args.output)?;
        println!("\nSchema saved to: {:?}", args.output);
    }

    // Generate templates if requested
    if args.templates {
        let registry = TemplateRegistry::default_templates();
        registry.save(&args.templates_output)?;
        println!("Templates saved to: {:?}", args.templates_output);
    }

    Ok(())
}

/// Generate schema for all crates in workspace
fn generate_all_crates(workspace: &PathBuf, verbose: bool) -> Result<ApiSchema> {
    let mut schema = ApiSchema::new("0.1.0");

    // List of DPB crates to process
    let crates = [
        "dpb-core",
        "dpb-neurons",
        "dpb-encoders",
        "dpb-snn",
        "dpb-synth",
        "dpb-clinical",
        "dpb-norms",
        "dpb-viz",
        "dpb-lsl",
        "dpb-export",
        "dpb-federated",
        "dpb-cognitive",
        "dpb-bench",
        "dpb-python",
        "dpb-ffi",
        "dpb-wasm",
        "dpb-mobile",
    ];

    for crate_name in &crates {
        let json_name = crate_name.replace('-', "_");
        let json_path = workspace.join("target/doc").join(format!("{}.json", json_name));

        if json_path.exists() {
            if verbose {
                println!("Processing: {}", crate_name);
            }
            match extract_crate_schema(&json_path) {
                Ok(crate_schema) => {
                    schema.crates.insert(crate_name.to_string(), crate_schema);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to process {}: {}", crate_name, e);
                }
            }
        } else {
            if verbose {
                println!("Skipping {} (no JSON found at {:?})", crate_name, json_path);
            }
        }
    }

    // If no JSON files found, generate from source
    if schema.crates.is_empty() {
        println!("No rustdoc JSON found. Generating from source analysis...");
        schema = generate_from_source(workspace)?;
    }

    Ok(schema)
}

/// Generate schema by analyzing source files directly
fn generate_from_source(workspace: &PathBuf) -> Result<ApiSchema> {
    use std::fs;

    let mut schema = ApiSchema::new("0.1.0");

    // Crate configurations
    let crate_configs = [
        ("dpb-core", "Core signal processing primitives, types, and traits"),
        ("dpb-neurons", "Spiking neuron models (LIF, Izhikevich, AdEx, etc.)"),
        ("dpb-encoders", "Event-based encoders for biosignals (77+ encoders)"),
        ("dpb-snn", "Spiking Neural Network architectures and training"),
        ("dpb-synth", "Synthetic biosignal generation and augmentation"),
        ("dpb-clinical", "Clinical analysis, HIPAA compliance, normative data"),
        ("dpb-norms", "Age/sex-stratified normative databases"),
        ("dpb-viz", "Visualization toolkit (raster plots, heatmaps, dashboards)"),
        ("dpb-lsl", "Lab Streaming Layer integration"),
        ("dpb-export", "Model export (ONNX, TensorFlow, neuromorphic)"),
        ("dpb-federated", "Privacy-preserving federated learning"),
        ("dpb-cognitive", "Cognitive assessment tasks"),
        ("dpb-bench", "Benchmarking and profiling"),
        ("dpb-python", "Python bindings (PyO3)"),
        ("dpb-ffi", "C FFI bindings"),
        ("dpb-wasm", "WebAssembly bindings"),
        ("dpb-mobile", "iOS/Android runtime"),
    ];

    for (crate_name, description) in &crate_configs {
        let crate_path = workspace.join("crates").join(crate_name);
        let lib_path = crate_path.join("src/lib.rs");

        if lib_path.exists() {
            let mut crate_schema = CrateSchema::new(crate_name, "0.1.0");
            crate_schema.docs = Some(description.to_string());

            // Parse lib.rs for public exports
            if let Ok(content) = fs::read_to_string(&lib_path) {
                parse_rust_source(&content, &mut crate_schema);
            }

            schema.crates.insert(crate_name.to_string(), crate_schema);
        }
    }

    Ok(schema)
}

/// Simple Rust source parser for extracting public API
fn parse_rust_source(content: &str, schema: &mut CrateSchema) {
    use dpb_ai_agent::schema::{TypeDef, TypeKind, FunctionDef, ModuleDef, Visibility};

    // Extract module-level docs
    let doc_pattern = regex::Regex::new(r"//!\s*(.*)").unwrap();
    let docs: Vec<String> = doc_pattern
        .captures_iter(content)
        .map(|c| c[1].to_string())
        .collect();
    if !docs.is_empty() {
        schema.docs = Some(docs.join("\n"));
    }

    // Extract pub use statements (re-exports)
    let pub_use_pattern = regex::Regex::new(
        r"pub use (\w+(?:::\w+)*)::\{([^}]+)\};"
    ).unwrap();

    for cap in pub_use_pattern.captures_iter(content) {
        let base_path = &cap[1];
        let items = &cap[2];
        for item in items.split(',').map(|s| s.trim()) {
            if !item.is_empty() {
                // Record as a type for now
                schema.types.push(TypeDef {
                    name: item.to_string(),
                    path: format!("{}::{}", base_path, item),
                    kind: TypeKind::TypeAlias, // Placeholder
                    docs: None,
                    generics: Vec::new(),
                    fields: Vec::new(),
                    methods: Vec::new(),
                    traits: Vec::new(),
                    visibility: Visibility::Public,
                });
            }
        }
    }

    // Extract pub struct
    let struct_pattern = regex::Regex::new(
        r"(?m)^pub struct (\w+)"
    ).unwrap();

    for cap in struct_pattern.captures_iter(content) {
        let name = &cap[1];
        schema.types.push(TypeDef {
            name: name.to_string(),
            path: format!("{}::{}", schema.name.replace('-', "_"), name),
            kind: TypeKind::Struct,
            docs: None,
            generics: Vec::new(),
            fields: Vec::new(),
            methods: Vec::new(),
            traits: Vec::new(),
            visibility: Visibility::Public,
        });
    }

    // Extract pub enum
    let enum_pattern = regex::Regex::new(
        r"(?m)^pub enum (\w+)"
    ).unwrap();

    for cap in enum_pattern.captures_iter(content) {
        let name = &cap[1];
        schema.types.push(TypeDef {
            name: name.to_string(),
            path: format!("{}::{}", schema.name.replace('-', "_"), name),
            kind: TypeKind::Enum,
            docs: None,
            generics: Vec::new(),
            fields: Vec::new(),
            methods: Vec::new(),
            traits: Vec::new(),
            visibility: Visibility::Public,
        });
    }

    // Extract pub trait
    let trait_pattern = regex::Regex::new(
        r"(?m)^pub trait (\w+)"
    ).unwrap();

    for cap in trait_pattern.captures_iter(content) {
        let name = &cap[1];
        schema.types.push(TypeDef {
            name: name.to_string(),
            path: format!("{}::{}", schema.name.replace('-', "_"), name),
            kind: TypeKind::Trait,
            docs: None,
            generics: Vec::new(),
            fields: Vec::new(),
            methods: Vec::new(),
            traits: Vec::new(),
            visibility: Visibility::Public,
        });
    }

    // Extract pub fn
    let fn_pattern = regex::Regex::new(
        r"(?m)^pub fn (\w+)"
    ).unwrap();

    for cap in fn_pattern.captures_iter(content) {
        let name = &cap[1];
        schema.functions.push(FunctionDef {
            name: name.to_string(),
            path: format!("{}::{}", schema.name.replace('-', "_"), name),
            docs: None,
            signature: format!("pub fn {}(...)", name),
            params: Vec::new(),
            return_type: None,
            generics: Vec::new(),
            is_async: false,
            is_unsafe: false,
            is_const: false,
            visibility: Visibility::Public,
        });
    }

    // Extract pub mod
    let mod_pattern = regex::Regex::new(
        r"(?m)^pub mod (\w+)"
    ).unwrap();

    for cap in mod_pattern.captures_iter(content) {
        let name = &cap[1];
        schema.modules.push(ModuleDef {
            path: name.to_string(),
            docs: None,
            visibility: Visibility::Public,
        });
    }
}

/// Generate OpenAPI 3.0 specification
fn generate_openapi(schema: &ApiSchema) -> serde_json::Value {
    let mut paths = serde_json::Map::new();
    let mut components = serde_json::Map::new();
    let mut schemas_map = serde_json::Map::new();

    // Generate paths for each function
    for (crate_name, crate_schema) in &schema.crates {
        for func in &crate_schema.functions {
            let path = format!("/api/{}/{}", crate_name.replace('-', "_"), func.name);

            let mut params = Vec::new();
            for param in &func.params {
                params.push(serde_json::json!({
                    "name": param.name,
                    "in": "query",
                    "description": format!("Parameter of type {}", param.ty),
                    "required": true,
                    "schema": {
                        "type": rust_type_to_openapi(&param.ty)
                    }
                }));
            }

            paths.insert(path.clone(), serde_json::json!({
                "post": {
                    "summary": func.docs.clone().unwrap_or_else(|| func.name.clone()),
                    "operationId": format!("{}_{}", crate_name.replace('-', "_"), func.name),
                    "tags": [crate_name],
                    "parameters": params,
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object"
                                    }
                                }
                            }
                        }
                    }
                }
            }));
        }

        // Generate schemas for each type
        for ty in &crate_schema.types {
            let type_schema = match ty.kind {
                dpb_ai_agent::schema::TypeKind::Struct => {
                    let mut properties = serde_json::Map::new();
                    for field in &ty.fields {
                        if let Some(name) = &field.name {
                            properties.insert(name.clone(), serde_json::json!({
                                "type": rust_type_to_openapi(&field.ty),
                                "description": field.docs.clone().unwrap_or_default()
                            }));
                        }
                    }
                    serde_json::json!({
                        "type": "object",
                        "description": ty.docs.clone().unwrap_or_default(),
                        "properties": properties
                    })
                }
                dpb_ai_agent::schema::TypeKind::Enum => {
                    let variants: Vec<String> = ty.fields.iter()
                        .filter_map(|f| f.name.clone())
                        .collect();
                    serde_json::json!({
                        "type": "string",
                        "description": ty.docs.clone().unwrap_or_default(),
                        "enum": variants
                    })
                }
                _ => serde_json::json!({
                    "type": "object",
                    "description": ty.docs.clone().unwrap_or_default()
                })
            };

            schemas_map.insert(ty.name.clone(), type_schema);
        }
    }

    components.insert("schemas".to_string(), serde_json::Value::Object(schemas_map));

    serde_json::json!({
        "openapi": "3.0.0",
        "info": {
            "title": "DPB Framework API",
            "description": "Delta-Predictive Biosensing Framework - API Reference",
            "version": schema.framework_version,
            "license": {
                "name": "MIT OR Apache-2.0"
            }
        },
        "servers": [
            {
                "url": "https://api.dpb.local",
                "description": "Local DPB API server"
            }
        ],
        "paths": paths,
        "components": components
    })
}

fn rust_type_to_openapi(rust_type: &str) -> &'static str {
    match rust_type {
        "f32" | "f64" => "number",
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" => "integer",
        "bool" => "boolean",
        "String" | "&str" | "str" => "string",
        _ => "object",
    }
}
