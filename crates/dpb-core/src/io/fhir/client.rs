//! FHIR Client for Server Interactions
//!
//! Provides a client for interacting with FHIR servers via HTTP/HTTPS.
//! Supports CRUD operations, search, and batch/transaction operations.
//!
//! Note: This is a framework for FHIR client operations. Full HTTP implementation
//! would require an HTTP client library like `reqwest` (not included in base dependencies).

use super::{Bundle, BundleType, FhirResource, ResourceType};
use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// FHIR client configuration
#[derive(Debug, Clone)]
pub struct FhirClientConfig {
    /// Base URL of the FHIR server (e.g., "https://example.com/fhir")
    pub base_url: String,
    /// Authentication token (Bearer token)
    pub auth_token: Option<String>,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Whether to use pretty-printed JSON
    pub pretty_json: bool,
    /// Custom HTTP headers
    pub headers: HashMap<String, String>,
}

impl FhirClientConfig {
    /// Creates a new FHIR client configuration
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token: None,
            api_key: None,
            timeout_seconds: 30,
            pretty_json: false,
            headers: HashMap::new(),
        }
    }

    /// Sets the authentication token
    pub fn with_auth_token(mut self, token: String) -> Self {
        self.auth_token = Some(token);
        self
    }

    /// Sets the API key
    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Sets the request timeout
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Adds a custom header
    pub fn add_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
}

/// FHIR client for server interactions
///
/// This is a framework for FHIR operations. In a production implementation,
/// you would integrate an HTTP client library like `reqwest` or `hyper`.
#[derive(Debug, Clone)]
pub struct FhirClient {
    config: FhirClientConfig,
}

impl FhirClient {
    /// Creates a new FHIR client
    pub fn new(config: FhirClientConfig) -> Self {
        Self { config }
    }

    /// Returns the base URL of the FHIR server
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Builds a resource URL
    fn resource_url(&self, resource_type: ResourceType, id: Option<&str>) -> String {
        match id {
            Some(id) => format!("{}/{}/{}", self.config.base_url, resource_type, id),
            None => format!("{}/{}", self.config.base_url, resource_type),
        }
    }

    /// Builds a search URL with parameters
    fn search_url(&self, resource_type: ResourceType, params: &SearchParams) -> String {
        let base = self.resource_url(resource_type, None);
        let query = params.to_query_string();
        if query.is_empty() {
            base
        } else {
            format!("{}?{}", base, query)
        }
    }

    // === CREATE operations ===

    /// Creates a new resource on the server (POST)
    ///
    /// In a full implementation, this would:
    /// 1. Serialize the resource to JSON
    /// 2. POST to {base}/{resourceType}
    /// 3. Return the created resource with server-assigned ID
    pub fn create(&self, _resource: FhirResource) -> Result<CreatedResource> {
        // Placeholder for HTTP POST implementation
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    // === READ operations ===

    /// Reads a resource by type and ID (GET)
    ///
    /// In a full implementation, this would:
    /// 1. GET {base}/{resourceType}/{id}
    /// 2. Deserialize the JSON response
    /// 3. Return the resource
    pub fn read(&self, _resource_type: ResourceType, _id: &str) -> Result<FhirResource> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    /// Reads the version history of a resource
    pub fn history(&self, _resource_type: ResourceType, _id: &str) -> Result<Bundle> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    // === UPDATE operations ===

    /// Updates an existing resource (PUT)
    ///
    /// In a full implementation, this would:
    /// 1. Serialize the resource to JSON
    /// 2. PUT to {base}/{resourceType}/{id}
    /// 3. Return the updated resource
    pub fn update(&self, _resource: FhirResource) -> Result<FhirResource> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    /// Conditionally updates a resource based on search criteria
    pub fn conditional_update(
        &self,
        _resource: FhirResource,
        _params: SearchParams,
    ) -> Result<FhirResource> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    // === DELETE operations ===

    /// Deletes a resource (DELETE)
    pub fn delete(&self, _resource_type: ResourceType, _id: &str) -> Result<()> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    /// Conditionally deletes resources based on search criteria
    pub fn conditional_delete(&self, _resource_type: ResourceType, _params: SearchParams) -> Result<()> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    // === SEARCH operations ===

    /// Searches for resources
    ///
    /// In a full implementation, this would:
    /// 1. Build query string from params
    /// 2. GET {base}/{resourceType}?{params}
    /// 3. Return search result bundle
    pub fn search(&self, _resource_type: ResourceType, _params: SearchParams) -> Result<Bundle> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    /// Performs a system-level search across all resource types
    pub fn search_all(&self, _params: SearchParams) -> Result<Bundle> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    // === BATCH/TRANSACTION operations ===

    /// Submits a transaction bundle (atomic operation)
    pub fn transaction(&self, _bundle: Bundle) -> Result<Bundle> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    /// Submits a batch bundle (independent operations)
    pub fn batch(&self, _bundle: Bundle) -> Result<Bundle> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }

    // === CAPABILITY operations ===

    /// Retrieves the server's capability statement
    pub fn capabilities(&self) -> Result<CapabilityStatement> {
        Err(DpbError::Other(
            "HTTP client not implemented. Integrate reqwest or similar HTTP library.".to_string()
        ))
    }
}

/// Response from creating a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatedResource {
    /// The created resource
    pub resource: FhirResource,
    /// Location of the created resource
    pub location: String,
    /// ETag for version management
    pub etag: Option<String>,
}

/// Search parameters for FHIR queries
#[derive(Debug, Clone, Default)]
pub struct SearchParams {
    params: HashMap<String, Vec<String>>,
}

impl SearchParams {
    /// Creates new empty search parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a search parameter
    pub fn add(mut self, key: String, value: String) -> Self {
        self.params.entry(key).or_default().push(value);
        self
    }

    /// Adds a parameter for searching by patient
    pub fn patient(self, patient_id: String) -> Self {
        self.add("patient".to_string(), patient_id)
    }

    /// Adds a parameter for searching by subject
    pub fn subject(self, subject_ref: String) -> Self {
        self.add("subject".to_string(), subject_ref)
    }

    /// Adds a date range parameter
    pub fn date_range(self, start: String, end: String) -> Self {
        self.add("date".to_string(), format!("ge{}", start))
            .add("date".to_string(), format!("le{}", end))
    }

    /// Adds a code search parameter
    pub fn code(self, code: String) -> Self {
        self.add("code".to_string(), code)
    }

    /// Sets the page size (_count parameter)
    pub fn count(self, count: usize) -> Self {
        self.add("_count".to_string(), count.to_string())
    }

    /// Sets the page number (_page parameter)
    pub fn page(self, page: usize) -> Self {
        self.add("_page".to_string(), page.to_string())
    }

    /// Adds a sort parameter
    pub fn sort(self, field: String, ascending: bool) -> Self {
        let value = if ascending {
            field
        } else {
            format!("-{}", field)
        };
        self.add("_sort".to_string(), value)
    }

    /// Converts parameters to URL query string
    pub fn to_query_string(&self) -> String {
        let mut parts = Vec::new();
        for (key, values) in &self.params {
            for value in values {
                parts.push(format!("{}={}",
                    urlencoding::encode(key),
                    urlencoding::encode(value)));
            }
        }
        parts.join("&")
    }

    /// Returns the number of parameters
    pub fn len(&self) -> usize {
        self.params.values().map(|v| v.len()).sum()
    }

    /// Checks if parameters are empty
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// Simple URL encoding helper (since we don't have the urlencoding crate)
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

/// Simplified FHIR Capability Statement
///
/// Describes what the FHIR server can do
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityStatement {
    /// Resource type (always "CapabilityStatement")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Status: draft | active | retired
    pub status: String,
    /// FHIR version
    #[serde(rename = "fhirVersion")]
    pub fhir_version: String,
    /// Supported formats (json, xml, etc.)
    pub format: Vec<String>,
    /// Server implementation details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub implementation: Option<ImplementationInfo>,
}

/// Server implementation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationInfo {
    /// Description of the implementation
    pub description: String,
    /// Server URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config() {
        let config = FhirClientConfig::new("https://example.com/fhir".to_string())
            .with_auth_token("token123".to_string())
            .with_timeout(60)
            .add_header("X-Custom".to_string(), "value".to_string());

        assert_eq!(config.base_url, "https://example.com/fhir");
        assert_eq!(config.auth_token, Some("token123".to_string()));
        assert_eq!(config.timeout_seconds, 60);
        assert!(config.headers.contains_key("X-Custom"));
    }

    #[test]
    fn test_client_urls() {
        let config = FhirClientConfig::new("https://example.com/fhir".to_string());
        let client = FhirClient::new(config);

        let url = client.resource_url(ResourceType::Patient, Some("123"));
        assert_eq!(url, "https://example.com/fhir/Patient/123");

        let url = client.resource_url(ResourceType::Observation, None);
        assert_eq!(url, "https://example.com/fhir/Observation");
    }

    #[test]
    fn test_search_params() {
        let params = SearchParams::new()
            .patient("Patient/123".to_string())
            .code("8867-4".to_string())
            .count(10)
            .page(0);

        assert!(!params.is_empty());
        assert_eq!(params.len(), 4);

        let query = params.to_query_string();
        assert!(query.contains("patient="));
        assert!(query.contains("code="));
    }

    #[test]
    fn test_search_params_date_range() {
        let params = SearchParams::new()
            .date_range("2025-01-01".to_string(), "2025-12-31".to_string());

        let query = params.to_query_string();
        assert!(query.contains("date=ge2025-01-01"));
        assert!(query.contains("date=le2025-12-31"));
    }

    #[test]
    fn test_search_params_sort() {
        let params = SearchParams::new()
            .sort("date".to_string(), true)
            .sort("name".to_string(), false);

        let query = params.to_query_string();
        assert!(query.contains("_sort=date"));
        assert!(query.contains("_sort=-name"));
    }

    #[test]
    fn test_url_encoding() {
        let encoded = urlencoding::encode("Patient/123");
        assert_eq!(encoded, "Patient%2F123");

        let encoded = urlencoding::encode("Hello World!");
        assert!(encoded.contains("%20") || encoded.contains("%21"));
    }

    #[test]
    fn test_search_url_building() {
        let config = FhirClientConfig::new("https://example.com/fhir".to_string());
        let client = FhirClient::new(config);

        let params = SearchParams::new()
            .patient("Patient/123".to_string());

        let url = client.search_url(ResourceType::Observation, &params);
        assert!(url.starts_with("https://example.com/fhir/Observation?"));
        assert!(url.contains("patient="));
    }
}
