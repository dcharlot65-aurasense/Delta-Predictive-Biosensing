//! MCP (Model Context Protocol) Server
//!
//! Exposes DPB Framework APIs as tools that AI agents can call directly.
//!
//! # Usage
//!
//! ```bash
//! # Start the MCP server (stdio mode for Claude Desktop)
//! cargo run --bin dpb-mcp-server
//!
//! # Start with HTTP transport
//! cargo run --bin dpb-mcp-server -- --http --port 3000
//!
//! # Load custom schema
//! cargo run --bin dpb-mcp-server -- --schema api-schema.json
//! ```
//!
//! # Claude Desktop Configuration
//!
//! Add to `claude_desktop_config.json`:
//! ```json
//! {
//!   "mcpServers": {
//!     "dpb": {
//!       "command": "path/to/dpb-mcp-server",
//!       "args": []
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use clap::Parser;
use serde::Deserialize;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use dpb_ai_agent::schema::ApiSchema;
use dpb_ai_agent::mcp::{
    McpServer, McpConfig, JsonRpcRequest, JsonRpcResponse, CallToolResult, ToolResultContent,
};
use dpb_ai_agent::templates::TemplateRegistry;

/// MCP Server for DPB Framework
#[derive(Parser, Debug)]
#[command(name = "dpb-mcp-server")]
#[command(about = "Model Context Protocol server for DPB Framework")]
#[command(version)]
struct Args {
    /// API schema file to load
    #[arg(short, long)]
    schema: Option<PathBuf>,

    /// Enable HTTP transport instead of stdio
    #[arg(long)]
    http: bool,

    /// HTTP port (when using --http)
    #[arg(long, default_value = "3000")]
    port: u16,

    /// HTTP host (when using --http)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("debug")
            .init();
    }

    // Load or generate schema
    let schema = if let Some(schema_path) = &args.schema {
        ApiSchema::load(schema_path)?
    } else {
        generate_default_schema()
    };

    // Create MCP server
    let config = McpConfig {
        name: "dpb-mcp-server".to_string(),
        version: "0.1.0".to_string(),
        host: args.host.clone(),
        port: args.port,
        cors_enabled: true,
        allowed_origins: vec!["*".to_string()],
    };

    let server = Arc::new(McpServer::new(config).with_schema(schema));

    if args.http {
        // HTTP mode
        run_http_server(server, &args.host, args.port).await
    } else {
        // Stdio mode (for Claude Desktop)
        run_stdio_server(server).await
    }
}

/// Run MCP server over stdio (for Claude Desktop integration)
async fn run_stdio_server(server: Arc<McpServer>) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    let templates = TemplateRegistry::default_templates();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        // Parse JSON-RPC request
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let error_response = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    &format!("Parse error: {}", e),
                );
                writeln!(stdout, "{}", serde_json::to_string(&error_response)?)?;
                stdout.flush()?;
                continue;
            }
        };

        // Handle request
        let response = handle_request(&server, &templates, request).await;

        // Send response
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }

    Ok(())
}

/// Run MCP server over HTTP
async fn run_http_server(server: Arc<McpServer>, host: &str, port: u16) -> Result<()> {
    use axum::{
        Router,
        routing::post,
        extract::State,
        Json,
    };
    use tower_http::cors::{CorsLayer, Any};

    #[derive(Clone)]
    struct AppState {
        server: Arc<McpServer>,
        templates: Arc<TemplateRegistry>,
    }

    async fn http_handler(
        State(state): State<AppState>,
        Json(request): Json<JsonRpcRequest>,
    ) -> Json<JsonRpcResponse> {
        Json(handle_request(&state.server, &state.templates, request).await)
    }

    let state = AppState {
        server,
        templates: Arc::new(TemplateRegistry::default_templates()),
    };

    let app = Router::new()
        .route("/mcp", post(http_handler))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    println!("MCP Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Handle MCP JSON-RPC request
async fn handle_request(
    server: &McpServer,
    templates: &TemplateRegistry,
    request: JsonRpcRequest,
) -> JsonRpcResponse {
    match request.method.as_str() {
        // Initialize
        "initialize" => {
            JsonRpcResponse::success(request.id, serde_json::json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": server.server_info(),
                "capabilities": server.capabilities()
            }))
        }

        // List tools
        "tools/list" => {
            let result = server.list_tools();
            JsonRpcResponse::success(request.id, serde_json::to_value(result).unwrap())
        }

        // Call tool
        "tools/call" => {
            let params: ToolCallParams = match serde_json::from_value(request.params.clone()) {
                Ok(p) => p,
                Err(e) => {
                    return JsonRpcResponse::error(
                        request.id,
                        -32602,
                        &format!("Invalid params: {}", e),
                    );
                }
            };

            let result = handle_tool_call(server, templates, &params.name, params.arguments).await;
            match result {
                Ok(call_result) => {
                    JsonRpcResponse::success(request.id, serde_json::to_value(call_result).unwrap())
                }
                Err(e) => {
                    JsonRpcResponse::success(request.id, serde_json::to_value(CallToolResult {
                        content: vec![ToolResultContent::Text {
                            text: format!("Error: {}", e),
                        }],
                        is_error: Some(true),
                    }).unwrap())
                }
            }
        }

        // Resources (not implemented)
        "resources/list" => {
            JsonRpcResponse::success(request.id, serde_json::json!({
                "resources": []
            }))
        }

        // Prompts (not implemented)
        "prompts/list" => {
            JsonRpcResponse::success(request.id, serde_json::json!({
                "prompts": []
            }))
        }

        // Ping
        "ping" => {
            JsonRpcResponse::success(request.id, serde_json::json!({}))
        }

        // Unknown method
        _ => {
            JsonRpcResponse::error(
                request.id,
                -32601,
                &format!("Method not found: {}", request.method),
            )
        }
    }
}

#[derive(Deserialize)]
struct ToolCallParams {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

/// Handle tool call
async fn handle_tool_call(
    server: &McpServer,
    templates: &TemplateRegistry,
    name: &str,
    arguments: serde_json::Value,
) -> Result<CallToolResult> {
    match name {
        // Schema introspection tools
        "dpb_list_crates" => {
            let _crates_info: Vec<String> = server.tools()
                .iter()
                .filter_map(|t| {
                    if t.name.starts_with("dpb_encoder_") ||
                       t.name.starts_with("dpb_neuron_") ||
                       t.name.starts_with("dpb_snn_") ||
                       t.name.starts_with("dpb_synth_") {
                        None
                    } else {
                        Some(format!("{}: {}", t.name, t.description))
                    }
                })
                .collect();

            let text = format!(
                "# DPB Framework Crates\n\n\
                 ## Core Crates\n\
                 - **dpb-core**: Signal processing primitives, types, and traits\n\
                 - **dpb-neurons**: 19 spiking neuron models (LIF, Izhikevich, AdEx, etc.)\n\
                 - **dpb-encoders**: 75 event-based encoders for biosignals\n\
                 - **dpb-snn**: SNN architectures, training, and inference\n\
                 - **dpb-synth**: Synthetic biosignal generation\n\n\
                 ## Clinical & Analysis\n\
                 - **dpb-clinical**: HIPAA compliance, de-identification\n\
                 - **dpb-norms**: Age/sex-stratified normative databases\n\
                 - **dpb-cognitive**: Cognitive assessment tasks\n\n\
                 ## Visualization & Export\n\
                 - **dpb-viz**: Raster plots, heatmaps, dashboards\n\
                 - **dpb-export**: ONNX, TensorFlow, neuromorphic export\n\n\
                 ## Integration\n\
                 - **dpb-lsl**: Lab Streaming Layer integration\n\
                 - **dpb-federated**: Privacy-preserving federated learning\n\n\
                 ## Platform Bindings\n\
                 - **dpb-python**: Python bindings (PyO3)\n\
                 - **dpb-ffi**: C FFI bindings\n\
                 - **dpb-wasm**: WebAssembly bindings\n\
                 - **dpb-mobile**: iOS/Android runtime"
            );

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text }],
                is_error: Some(false),
            })
        }

        "dpb_search_types" => {
            let pattern = arguments.get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let results = search_types_in_codebase(pattern);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text: results }],
                is_error: Some(false),
            })
        }

        "dpb_search_functions" => {
            let pattern = arguments.get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let results = search_functions_in_codebase(pattern);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text: results }],
                is_error: Some(false),
            })
        }

        "dpb_get_type_info" => {
            let type_path = arguments.get("type_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let info = get_type_info(type_path);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text: info }],
                is_error: Some(false),
            })
        }

        "dpb_get_function_info" => {
            let function_path = arguments.get("function_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let info = get_function_info(function_path);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text: info }],
                is_error: Some(false),
            })
        }

        "dpb_get_module_tree" => {
            let crate_name = arguments.get("crate_name")
                .and_then(|v| v.as_str())
                .unwrap_or("dpb-core");

            let tree = get_module_tree(crate_name);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text: tree }],
                is_error: Some(false),
            })
        }

        // Template tools
        name if name.starts_with("dpb_template_") => {
            let template_id = name.strip_prefix("dpb_template_").unwrap();
            let action = arguments.get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("info");

            if let Some(template) = templates.get(template_id) {
                let text = match action {
                    "example" => template.example.clone(),
                    "code" => template.code.clone(),
                    _ => format!(
                        "# {}\n\n{}\n\n## Dependencies\n{}\n\n## Imports\n```rust\n{}\n```\n\n## Code Template\n```rust\n{}\n```",
                        template.name,
                        template.description,
                        template.dependencies.join(", "),
                        template.imports.join("\n"),
                        template.code
                    ),
                };

                Ok(CallToolResult {
                    content: vec![ToolResultContent::Text { text }],
                    is_error: Some(false),
                })
            } else {
                Ok(CallToolResult {
                    content: vec![ToolResultContent::Text {
                        text: format!("Template not found: {}", template_id),
                    }],
                    is_error: Some(true),
                })
            }
        }

        // Encoder tools
        name if name.starts_with("dpb_encoder_") => {
            let encoder_name = name.strip_prefix("dpb_encoder_").unwrap();
            let action = arguments.get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("info");

            let text = get_encoder_info(encoder_name, action);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text }],
                is_error: Some(false),
            })
        }

        // Neuron tools
        name if name.starts_with("dpb_neuron_") => {
            let neuron_name = name.strip_prefix("dpb_neuron_").unwrap();
            let action = arguments.get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("info");

            let text = get_neuron_info(neuron_name, action);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text }],
                is_error: Some(false),
            })
        }

        // SNN tools
        name if name.starts_with("dpb_snn_") => {
            let snn_name = name.strip_prefix("dpb_snn_").unwrap();
            let action = arguments.get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("info");

            let text = get_snn_info(snn_name, action);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text }],
                is_error: Some(false),
            })
        }

        // Synth tools
        name if name.starts_with("dpb_synth_") => {
            let synth_name = name.strip_prefix("dpb_synth_").unwrap();
            let action = arguments.get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("info");

            let text = get_synth_info(synth_name, action);

            Ok(CallToolResult {
                content: vec![ToolResultContent::Text { text }],
                is_error: Some(false),
            })
        }

        _ => {
            Ok(CallToolResult {
                content: vec![ToolResultContent::Text {
                    text: format!("Unknown tool: {}", name),
                }],
                is_error: Some(true),
            })
        }
    }
}

// Helper functions for documentation

fn search_types_in_codebase(pattern: &str) -> String {
    let pattern_lower = pattern.to_lowercase();

    // Common types that match the pattern
    let all_types = vec![
        // Core types
        ("TimeSeries", "dpb-core", "Time series signal container"),
        ("SpikeTrain", "dpb-core", "Collection of spike events"),
        ("SpikeEvent", "dpb-core", "Single spike with timestamp and value"),
        ("Context", "dpb-core", "Processing context"),
        ("SignalBuffer", "dpb-core", "Circular buffer for streaming signals"),
        // Neurons
        ("LifNeuron", "dpb-neurons", "Leaky Integrate-and-Fire neuron"),
        ("IzhikevichNeuron", "dpb-neurons", "Izhikevich neuron model"),
        ("AdExNeuron", "dpb-neurons", "Adaptive Exponential neuron"),
        ("HodgkinHuxleyNeuron", "dpb-neurons", "Biophysical Hodgkin-Huxley model"),
        // Encoders
        ("LevelCrossingEncoder", "dpb-encoders", "Level crossing spike encoder"),
        ("EcgRPeakEncoder", "dpb-encoders", "ECG R-peak detection encoder"),
        ("PpgPulseEncoder", "dpb-encoders", "PPG pulse detection encoder"),
        // SNN
        ("FeedforwardSNN", "dpb-snn", "Feedforward spiking neural network"),
        ("ConvolutionalSNN", "dpb-snn", "Convolutional SNN"),
        ("RecurrentSNN", "dpb-snn", "Recurrent SNN"),
    ];

    let matches: Vec<_> = all_types.iter()
        .filter(|(name, _, _)| name.to_lowercase().contains(&pattern_lower))
        .collect();

    if matches.is_empty() {
        format!("No types found matching '{}'", pattern)
    } else {
        let mut result = format!("# Types matching '{}'\n\n", pattern);
        for (name, crate_name, desc) in matches {
            result.push_str(&format!("- **{}** ({}): {}\n", name, crate_name, desc));
        }
        result
    }
}

fn search_functions_in_codebase(pattern: &str) -> String {
    let pattern_lower = pattern.to_lowercase();

    let all_functions = vec![
        ("encode", "EventEncoder::encode", "Encode signal to spikes"),
        ("forward", "SNNNetwork::forward", "Forward pass through network"),
        ("train_step", "Trainer::train_step", "Single training step"),
        ("detect", "QrsDetector::detect", "Detect QRS complexes"),
        ("filter", "Filter::apply", "Apply filter to signal"),
    ];

    let matches: Vec<_> = all_functions.iter()
        .filter(|(name, _, _)| name.to_lowercase().contains(&pattern_lower))
        .collect();

    if matches.is_empty() {
        format!("No functions found matching '{}'", pattern)
    } else {
        let mut result = format!("# Functions matching '{}'\n\n", pattern);
        for (name, path, desc) in matches {
            result.push_str(&format!("- **{}** ({}): {}\n", name, path, desc));
        }
        result
    }
}

fn get_type_info(type_path: &str) -> String {
    format!(
        "# Type: {}\n\n\
         Use `cargo doc --open -p dpb-core` to view full documentation.\n\n\
         Alternatively, search the codebase with:\n\
         ```\n\
         grep -r 'pub struct {}' crates/\n\
         ```",
        type_path,
        type_path.split("::").last().unwrap_or(type_path)
    )
}

fn get_function_info(function_path: &str) -> String {
    format!(
        "# Function: {}\n\n\
         Use `cargo doc --open` to view full documentation.\n\n\
         Alternatively, search the codebase with:\n\
         ```\n\
         grep -r 'fn {}' crates/\n\
         ```",
        function_path,
        function_path.split("::").last().unwrap_or(function_path)
    )
}

fn get_module_tree(crate_name: &str) -> String {
    match crate_name {
        "dpb-core" => {
            "# dpb-core Module Tree\n\n\
             ```\n\
             dpb-core/\n\
             ├── signal/           # Signal processing\n\
             │   ├── ecg.rs       # ECG analysis\n\
             │   ├── ppg.rs       # PPG analysis\n\
             │   ├── eda.rs       # EDA analysis\n\
             │   ├── emg.rs       # EMG analysis\n\
             │   ├── eeg/         # EEG analysis\n\
             │   ├── filter.rs    # Digital filters\n\
             │   ├── fft.rs       # FFT operations\n\
             │   └── resample.rs  # Resampling\n\
             ├── pipeline/         # Streaming pipelines\n\
             ├── io/               # File I/O (WFDB, EDF)\n\
             ├── types/            # Core types\n\
             ├── traits/           # Core traits\n\
             ├── gpu/              # GPU compute\n\
             └── metrics/          # Performance metrics\n\
             ```".to_string()
        }
        "dpb-encoders" => {
            "# dpb-encoders Module Tree\n\n\
             ```\n\
             dpb-encoders/\n\
             ├── base/             # Base encoders\n\
             │   ├── level_crossing.rs\n\
             │   ├── template_deviation.rs\n\
             │   └── derivative.rs\n\
             ├── contact/          # Contact modalities\n\
             │   ├── ecg.rs        # ECG encoders\n\
             │   ├── ppg.rs        # PPG encoders\n\
             │   ├── eda.rs        # EDA encoders\n\
             │   └── emg.rs        # EMG encoders\n\
             ├── pose/             # Movement encoders\n\
             ├── hand/             # Hand movement encoders\n\
             ├── eye/              # Eye tracking encoders\n\
             ├── voice/            # Voice encoders\n\
             └── templates/        # Population templates\n\
             ```".to_string()
        }
        _ => format!("Module tree not available for: {}", crate_name)
    }
}

fn get_encoder_info(encoder_name: &str, action: &str) -> String {
    match action {
        "example" => format!(
            "```rust\n\
             use dpb_encoders::{};\n\
             use dpb_core::traits::EventEncoder;\n\n\
             let encoder = {}::new(/* config */);\n\
             let spikes = encoder.encode(&signal);\n\
             ```",
            encoder_name, encoder_name
        ),
        "config" => format!(
            "# {} Configuration\n\n\
             See the type documentation with `cargo doc -p dpb-encoders --open`",
            encoder_name
        ),
        _ => format!(
            "# {}\n\n\
             Encoder for converting biosignals to spike events.\n\n\
             ## Usage\n\
             ```rust\n\
             use dpb_encoders::{};\n\
             use dpb_core::traits::EventEncoder;\n\n\
             let encoder = {}::default();\n\
             let spikes = encoder.encode(&signal);\n\
             ```",
            encoder_name, encoder_name, encoder_name
        )
    }
}

fn get_neuron_info(neuron_name: &str, action: &str) -> String {
    match action {
        "example" => format!(
            "```rust\n\
             use dpb_neurons::{};\n\n\
             let mut neuron = {}::default();\n\
             let output = neuron.step(input_current, dt);\n\
             ```",
            neuron_name, neuron_name
        ),
        "parameters" => {
            match neuron_name {
                "lif_neuron" => "# LIF Neuron Parameters\n\n\
                    - `tau_m`: Membrane time constant (ms)\n\
                    - `v_rest`: Resting potential (mV)\n\
                    - `v_thresh`: Threshold potential (mV)\n\
                    - `v_reset`: Reset potential (mV)\n\
                    - `r_m`: Membrane resistance".to_string(),
                "izhikevich_neuron" => "# Izhikevich Neuron Parameters\n\n\
                    - `a`: Time scale of recovery variable\n\
                    - `b`: Sensitivity of recovery to subthreshold fluctuations\n\
                    - `c`: After-spike reset value of v\n\
                    - `d`: After-spike reset of recovery variable".to_string(),
                _ => format!("Parameters for {} - see documentation", neuron_name)
            }
        }
        _ => format!(
            "# {}\n\n\
             Spiking neuron model implementation.\n\n\
             ## Usage\n\
             ```rust\n\
             use dpb_neurons::{};\n\n\
             let mut neuron = {}::default();\n\
             for t in 0..1000 {{\n\
                 let spike = neuron.step(input[t], 0.001);\n\
                 if spike {{ println!(\"Spike at t={{}}\", t); }}\n\
             }}\n\
             ```",
            neuron_name, neuron_name, neuron_name
        )
    }
}

fn get_snn_info(snn_name: &str, action: &str) -> String {
    match action {
        "example" => format!(
            "```rust\n\
             use dpb_snn::architectures::{};\n\n\
             let snn = {}::new(vec![784, 256, 10]);\n\
             let output = snn.forward(&input_spikes, 100);\n\
             ```",
            snn_name, snn_name
        ),
        "layers" => format!(
            "# {} Layer Configuration\n\n\
             Layers are configured during construction. Common patterns:\n\
             ```rust\n\
             // Classification network\n\
             let snn = {}::new(vec![input_size, hidden1, hidden2, num_classes]);\n\
             ```",
            snn_name, snn_name
        ),
        _ => format!(
            "# {}\n\n\
             Spiking Neural Network architecture.\n\n\
             ## Usage\n\
             ```rust\n\
             use dpb_snn::architectures::{};\n\n\
             let snn = {}::new(vec![784, 256, 10]);\n\
             let output = snn.forward(&input_spikes, time_steps);\n\
             ```",
            snn_name, snn_name, snn_name
        )
    }
}

fn get_synth_info(synth_name: &str, action: &str) -> String {
    match action {
        "example" => format!(
            "```rust\n\
             use dpb_synth::{};\n\n\
             let generator = {}::default();\n\
             let signal = generator.generate(duration_sec, sample_rate);\n\
             ```",
            synth_name, synth_name
        ),
        "params" => format!(
            "# {} Parameters\n\n\
             See the type's configuration struct for tunable parameters.\n\
             Use `cargo doc -p dpb-synth --open` for details.",
            synth_name
        ),
        _ => format!(
            "# {}\n\n\
             Synthetic biosignal generator.\n\n\
             ## Usage\n\
             ```rust\n\
             use dpb_synth::{};\n\n\
             let generator = {}::default();\n\
             let signal = generator.generate(10.0, 250.0); // 10s at 250Hz\n\
             ```",
            synth_name, synth_name, synth_name
        )
    }
}

fn generate_default_schema() -> ApiSchema {
    let mut schema = ApiSchema::new("0.1.0");

    // Add placeholder schemas for main crates
    use dpb_ai_agent::schema::CrateSchema;

    let crates = [
        ("dpb-core", "Core signal processing primitives"),
        ("dpb-neurons", "Spiking neuron models"),
        ("dpb-encoders", "Event-based signal encoders"),
        ("dpb-snn", "Spiking Neural Networks"),
        ("dpb-synth", "Synthetic signal generation"),
    ];

    for (name, desc) in crates {
        let mut crate_schema = CrateSchema::new(name, "0.1.0");
        crate_schema.docs = Some(desc.to_string());
        schema.crates.insert(name.to_string(), crate_schema);
    }

    schema
}
