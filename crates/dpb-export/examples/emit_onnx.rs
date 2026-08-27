//! Emits ONNX files for each encoder type so they can be checked against a
//! real ONNX implementation. Run: cargo run -p dpb-export --features onnx --example emit_onnx -- <dir>
use dpb_export::encoder_export::MockEncoder;
use dpb_export::metadata::ModelMetadata;
use dpb_export::onnx::OnnxExporter;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: emit_onnx <dir>");
    let cases: Vec<(&str, MockEncoder)> = vec![
        ("level_crossing", MockEncoder::level_crossing(4, 1000.0, 0.1)),
        ("delta", MockEncoder::delta(2, 500.0, 0.05, 8)),
        (
            "temporal_contrast",
            MockEncoder::temporal_contrast(3, 2000.0, 0.2, 0.001),
        ),
    ];
    for (name, enc) in cases {
        let mut meta = ModelMetadata::new();
        meta.name = name.to_string();
        meta.version = "0.1.0".to_string();
        meta.description = format!("DPB {name} encoder");
        let exporter = OnnxExporter::new(meta);
        let path = format!("{dir}/{name}.onnx");
        exporter.export(&path, &enc).expect("export");
        println!("wrote {path}");
    }
}
