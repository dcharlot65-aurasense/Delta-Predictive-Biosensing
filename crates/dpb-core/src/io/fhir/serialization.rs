//! FHIR Serialization and Deserialization
//!
//! Provides JSON and XML serialization support for FHIR R4 resources.

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Trait for FHIR serialization
pub trait FhirSerializer {
    /// Serializes a FHIR resource to a string
    fn serialize<T: Serialize>(&self, resource: &T) -> Result<String>;

    /// Serializes a FHIR resource to a writer
    fn serialize_to_writer<T: Serialize, W: Write>(&self, resource: &T, writer: W) -> Result<()>;

    /// Deserializes a FHIR resource from a string
    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &str) -> Result<T>;

    /// Deserializes a FHIR resource from a reader
    fn deserialize_from_reader<T: for<'de> Deserialize<'de>, R: Read>(&self, reader: R) -> Result<T>;

    /// Returns the MIME type for this serialization format
    fn mime_type(&self) -> &'static str;

    /// Returns the file extension for this format
    fn extension(&self) -> &'static str;
}

/// JSON serializer for FHIR R4
#[derive(Debug, Default, Clone)]
pub struct JsonSerializer {
    pretty: bool,
}

impl JsonSerializer {
    /// Creates a new JSON serializer
    pub fn new() -> Self {
        Self { pretty: false }
    }

    /// Creates a pretty-printing JSON serializer
    pub fn pretty() -> Self {
        Self { pretty: true }
    }

    /// Sets whether to use pretty printing
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }
}

impl FhirSerializer for JsonSerializer {
    fn serialize<T: Serialize>(&self, resource: &T) -> Result<String> {
        if self.pretty {
            serde_json::to_string_pretty(resource)
                .map_err(|e| DpbError::Other(format!("JSON serialization error: {}", e)))
        } else {
            serde_json::to_string(resource)
                .map_err(|e| DpbError::Other(format!("JSON serialization error: {}", e)))
        }
    }

    fn serialize_to_writer<T: Serialize, W: Write>(&self, resource: &T, writer: W) -> Result<()> {
        if self.pretty {
            serde_json::to_writer_pretty(writer, resource)
                .map_err(|e| DpbError::Other(format!("JSON serialization error: {}", e)))
        } else {
            serde_json::to_writer(writer, resource)
                .map_err(|e| DpbError::Other(format!("JSON serialization error: {}", e)))
        }
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &str) -> Result<T> {
        serde_json::from_str(data)
            .map_err(|e| DpbError::Other(format!("JSON deserialization error: {}", e)))
    }

    fn deserialize_from_reader<T: for<'de> Deserialize<'de>, R: Read>(&self, reader: R) -> Result<T> {
        serde_json::from_reader(reader)
            .map_err(|e| DpbError::Other(format!("JSON deserialization error: {}", e)))
    }

    fn mime_type(&self) -> &'static str {
        "application/fhir+json"
    }

    fn extension(&self) -> &'static str {
        "json"
    }
}

/// XML serializer for FHIR R4 (simplified implementation)
///
/// Note: Full FHIR XML serialization requires complex attribute handling and namespace support.
/// This is a simplified version that converts FHIR JSON to basic XML representation.
#[derive(Debug, Default, Clone)]
pub struct XmlSerializer {
    pretty: bool,
}

impl XmlSerializer {
    /// Creates a new XML serializer
    pub fn new() -> Self {
        Self { pretty: false }
    }

    /// Creates a pretty-printing XML serializer
    pub fn pretty() -> Self {
        Self { pretty: true }
    }

    /// Converts a JSON value to XML string (simplified)
    fn json_to_xml(&self, json: &serde_json::Value, tag_name: &str, indent: usize) -> String {
        let indent_str = if self.pretty {
            "  ".repeat(indent)
        } else {
            String::new()
        };
        let newline = if self.pretty { "\n" } else { "" };

        match json {
            serde_json::Value::Object(map) => {
                let mut xml = format!("{}<{}>{}", indent_str, tag_name, newline);
                for (key, value) in map {
                    if key != "resourceType" {
                        xml.push_str(&self.json_to_xml(value, key, indent + 1));
                    }
                }
                xml.push_str(&format!("{}</{}>{}", indent_str, tag_name, newline));
                xml
            }
            serde_json::Value::Array(arr) => {
                let mut xml = String::new();
                for item in arr {
                    xml.push_str(&self.json_to_xml(item, tag_name, indent));
                }
                xml
            }
            serde_json::Value::String(s) => {
                format!("{}<{} value=\"{}\" />{}", indent_str, tag_name,
                    escape_xml(s), newline)
            }
            serde_json::Value::Number(n) => {
                format!("{}<{} value=\"{}\" />{}", indent_str, tag_name, n, newline)
            }
            serde_json::Value::Bool(b) => {
                format!("{}<{} value=\"{}\" />{}", indent_str, tag_name, b, newline)
            }
            serde_json::Value::Null => {
                format!("{}<{} />{}", indent_str, tag_name, newline)
            }
        }
    }
}

impl FhirSerializer for XmlSerializer {
    fn serialize<T: Serialize>(&self, resource: &T) -> Result<String> {
        // First convert to JSON
        let json_str = serde_json::to_string(resource)
            .map_err(|e| DpbError::Other(format!("JSON serialization error: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| DpbError::Other(format!("JSON parsing error: {}", e)))?;

        // Extract resource type
        let resource_type = json_value
            .get("resourceType")
            .and_then(|v| v.as_str())
            .unwrap_or("Resource");

        // Convert to XML
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!("<{} xmlns=\"http://hl7.org/fhir\">\n", resource_type));

        if let serde_json::Value::Object(ref map) = json_value {
            for (key, value) in map {
                if key != "resourceType" {
                    xml.push_str(&self.json_to_xml(value, key, 1));
                }
            }
        }

        xml.push_str(&format!("</{}>\n", resource_type));
        Ok(xml)
    }

    fn serialize_to_writer<T: Serialize, W: Write>(&self, resource: &T, mut writer: W) -> Result<()> {
        let xml_string = self.serialize(resource)?;
        writer.write_all(xml_string.as_bytes())
            .map_err(|e| DpbError::Other(format!("Write error: {}", e)))
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, _data: &str) -> Result<T> {
        // XML deserialization not fully implemented - would require XML parser
        Err(DpbError::Other(
            "XML deserialization not yet implemented. Use JSON format.".to_string()
        ))
    }

    fn deserialize_from_reader<T: for<'de> Deserialize<'de>, R: Read>(&self, _reader: R) -> Result<T> {
        Err(DpbError::Other(
            "XML deserialization not yet implemented. Use JSON format.".to_string()
        ))
    }

    fn mime_type(&self) -> &'static str {
        "application/fhir+xml"
    }

    fn extension(&self) -> &'static str {
        "xml"
    }
}

/// Escapes special XML characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Generates a narrative text section for FHIR resources
pub struct NarrativeGenerator;

impl NarrativeGenerator {
    /// Generates a basic narrative for a patient resource
    pub fn patient_narrative(patient: &super::resources::Patient) -> String {
        let mut narrative = String::from("<div xmlns=\"http://www.w3.org/1999/xhtml\">");

        if let Some(name) = patient.name.first() {
            narrative.push_str("<p><b>");
            if let Some(family) = &name.family {
                narrative.push_str(family);
            }
            if !name.given.is_empty() {
                narrative.push_str(", ");
                narrative.push_str(&name.given.join(" "));
            }
            narrative.push_str("</b></p>");
        }

        if let Some(gender) = &patient.gender {
            narrative.push_str(&format!("<p>Gender: {}</p>", gender));
        }

        if let Some(dob) = &patient.birth_date {
            narrative.push_str(&format!("<p>Date of Birth: {}</p>", dob));
        }

        narrative.push_str("</div>");
        narrative
    }

    /// Generates a basic narrative for an observation
    pub fn observation_narrative(obs: &super::resources::Observation) -> String {
        let mut narrative = String::from("<div xmlns=\"http://www.w3.org/1999/xhtml\">");

        if let Some(text) = &obs.code.text {
            narrative.push_str(&format!("<p><b>{}</b></p>", text));
        }

        if let Some(value) = &obs.value_quantity
            && let (Some(val), Some(unit)) = (value.value, &value.unit) {
                narrative.push_str(&format!("<p>Value: {} {}</p>", val, unit));
            }

        if let Some(time) = &obs.effective_date_time {
            narrative.push_str(&format!("<p>Time: {}</p>", time));
        }

        narrative.push_str("</div>");
        narrative
    }
}

/// Extension handling for custom FHIR extensions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Extension {
    /// URL identifying the extension
    pub url: String,

    /// Value of the extension (simplified - only string values)
    #[serde(rename = "valueString", skip_serializing_if = "Option::is_none")]
    pub value_string: Option<String>,

    /// Value as integer
    #[serde(rename = "valueInteger", skip_serializing_if = "Option::is_none")]
    pub value_integer: Option<i64>,

    /// Value as boolean
    #[serde(rename = "valueBoolean", skip_serializing_if = "Option::is_none")]
    pub value_boolean: Option<bool>,

    /// Value as decimal
    #[serde(rename = "valueDecimal", skip_serializing_if = "Option::is_none")]
    pub value_decimal: Option<f64>,
}

impl Extension {
    /// Creates a string extension
    pub fn string(url: String, value: String) -> Self {
        Self {
            url,
            value_string: Some(value),
            value_integer: None,
            value_boolean: None,
            value_decimal: None,
        }
    }

    /// Creates an integer extension
    pub fn integer(url: String, value: i64) -> Self {
        Self {
            url,
            value_string: None,
            value_integer: Some(value),
            value_boolean: None,
            value_decimal: None,
        }
    }

    /// Creates a decimal extension
    pub fn decimal(url: String, value: f64) -> Self {
        Self {
            url,
            value_string: None,
            value_integer: None,
            value_boolean: None,
            value_decimal: Some(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fhir::resources::{Patient, HumanName, Observation};
    // Only the tests build resources directly; the serializers are generic.
    use crate::io::fhir::{Bundle, FhirResource};
    use crate::io::fhir::observations::VitalSignsObservation;

    #[test]
    fn test_json_serialization() {
        let patient = Patient {
            resource_type: "Patient".to_string(),
            id: Some("patient-001".to_string()),
            name: vec![HumanName {
                family: Some("Smith".to_string()),
                given: vec!["John".to_string()],
                ..Default::default()
            }],
            gender: Some("male".to_string()),
            ..Default::default()
        };

        let serializer = JsonSerializer::new();
        let json = serializer.serialize(&patient).unwrap();

        assert!(json.contains("Patient"));
        assert!(json.contains("Smith"));
        assert!(json.contains("John"));
    }

    #[test]
    fn test_json_pretty() {
        let patient = Patient::new("patient-001".to_string());

        let serializer = JsonSerializer::pretty();
        let json = serializer.serialize(&patient).unwrap();

        assert!(json.contains('\n')); // Pretty format has newlines
    }

    #[test]
    fn test_json_deserialization() {
        let json = r#"{
            "resourceType": "Patient",
            "id": "patient-001",
            "gender": "male"
        }"#;

        let serializer = JsonSerializer::new();
        let patient: Patient = serializer.deserialize(json).unwrap();

        assert_eq!(patient.id, Some("patient-001".to_string()));
        assert_eq!(patient.gender, Some("male".to_string()));
    }

    #[test]
    fn test_xml_serialization() {
        let patient = Patient {
            resource_type: "Patient".to_string(),
            id: Some("patient-001".to_string()),
            gender: Some("male".to_string()),
            ..Default::default()
        };

        let serializer = XmlSerializer::new();
        let xml = serializer.serialize(&patient).unwrap();

        assert!(xml.contains("<?xml"));
        assert!(xml.contains("<Patient"));
        assert!(xml.contains("</Patient>"));
    }

    #[test]
    fn test_mime_types() {
        let json_serializer = JsonSerializer::new();
        assert_eq!(json_serializer.mime_type(), "application/fhir+json");
        assert_eq!(json_serializer.extension(), "json");

        let xml_serializer = XmlSerializer::new();
        assert_eq!(xml_serializer.mime_type(), "application/fhir+xml");
        assert_eq!(xml_serializer.extension(), "xml");
    }

    #[test]
    fn test_escape_xml() {
        let input = "<tag> & \"quote\" 'apos'";
        let escaped = escape_xml(input);
        assert_eq!(escaped, "&lt;tag&gt; &amp; &quot;quote&quot; &apos;apos&apos;");
    }

    #[test]
    fn test_extension_creation() {
        let ext = Extension::string(
            "http://example.org/custom".to_string(),
            "custom value".to_string()
        );
        assert_eq!(ext.url, "http://example.org/custom");
        assert_eq!(ext.value_string, Some("custom value".to_string()));

        let ext_int = Extension::integer(
            "http://example.org/score".to_string(),
            42
        );
        assert_eq!(ext_int.value_integer, Some(42));
    }

    #[test]
    fn test_bundle_serialization() {
        let patient = Patient::new("patient-001".to_string());
        let bundle = Bundle::collection()
            .add_resource(FhirResource::Patient(patient));

        let serializer = JsonSerializer::pretty();
        let json = serializer.serialize(&bundle).unwrap();

        assert!(json.contains("Bundle"));
        assert!(json.contains("collection"));
    }
}
