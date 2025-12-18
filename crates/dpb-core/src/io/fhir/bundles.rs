//! FHIR Bundle Support
//!
//! Bundles are containers for a collection of resources, used for various purposes
//! including documents, transactions, batches, and search results.

use super::FhirResource;
use serde::{Deserialize, Serialize};

/// FHIR Bundle types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BundleType {
    /// A document bundle (immutable collection)
    Document,
    /// A message bundle
    Message,
    /// A transaction bundle (atomic update)
    Transaction,
    /// A transaction response
    #[serde(rename = "transaction-response")]
    TransactionResponse,
    /// A batch bundle (independent updates)
    Batch,
    /// A batch response
    #[serde(rename = "batch-response")]
    BatchResponse,
    /// A history list of versions
    History,
    /// Search results
    Searchset,
    /// A collection of resources
    Collection,
}

impl BundleType {
    /// Returns the bundle type as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            BundleType::Document => "document",
            BundleType::Message => "message",
            BundleType::Transaction => "transaction",
            BundleType::TransactionResponse => "transaction-response",
            BundleType::Batch => "batch",
            BundleType::BatchResponse => "batch-response",
            BundleType::History => "history",
            BundleType::Searchset => "searchset",
            BundleType::Collection => "collection",
        }
    }
}

/// HTTP verb for bundle entry requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpVerb {
    /// HTTP GET
    Get,
    /// HTTP POST (create)
    Post,
    /// HTTP PUT (update/create)
    Put,
    /// HTTP DELETE
    Delete,
    /// HTTP PATCH
    Patch,
}

/// Request information for bundle entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleEntryRequest {
    /// HTTP method
    pub method: HttpVerb,
    /// URL for the request
    pub url: String,
    /// If-None-Match header (for conditional create)
    #[serde(rename = "ifNoneMatch", skip_serializing_if = "Option::is_none")]
    pub if_none_match: Option<String>,
    /// If-Modified-Since header
    #[serde(rename = "ifModifiedSince", skip_serializing_if = "Option::is_none")]
    pub if_modified_since: Option<String>,
    /// If-Match header (for conditional update)
    #[serde(rename = "ifMatch", skip_serializing_if = "Option::is_none")]
    pub if_match: Option<String>,
}

/// Response information for bundle entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleEntryResponse {
    /// HTTP status code
    pub status: String,
    /// Location header (for created resources)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// ETag header
    #[serde(skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
    /// Last-Modified header
    #[serde(rename = "lastModified", skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<String>,
}

/// Search parameters for bundle entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleEntrySearch {
    /// Search mode: match | include | outcome
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Search score (relevance)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// Link in a bundle (for pagination, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleLink {
    /// Link relation type (self, next, previous, etc.)
    pub relation: String,
    /// URL for the link
    pub url: String,
}

impl BundleLink {
    /// Creates a new bundle link
    pub fn new(relation: &str, url: String) -> Self {
        Self {
            relation: relation.to_string(),
            url,
        }
    }

    /// Creates a "self" link
    pub fn self_link(url: String) -> Self {
        Self::new("self", url)
    }

    /// Creates a "next" link for pagination
    pub fn next(url: String) -> Self {
        Self::new("next", url)
    }

    /// Creates a "previous" link for pagination
    pub fn previous(url: String) -> Self {
        Self::new("previous", url)
    }
}

/// Entry in a FHIR bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleEntry {
    /// Full URL of the resource (absolute or relative)
    #[serde(rename = "fullUrl", skip_serializing_if = "Option::is_none")]
    pub full_url: Option<String>,

    /// The resource in this entry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<FhirResource>,

    /// Request information (for transaction/batch)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<BundleEntryRequest>,

    /// Response information (for transaction-response/batch-response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<BundleEntryResponse>,

    /// Search information (for search results)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<BundleEntrySearch>,
}

impl BundleEntry {
    /// Creates a new bundle entry with a resource
    pub fn new(resource: FhirResource) -> Self {
        // Generate full URL from resource type and ID
        let full_url = if let Some(id) = resource.id() {
            Some(format!("{}/{}", resource.resource_type(), id))
        } else {
            None
        };

        Self {
            full_url,
            resource: Some(resource),
            request: None,
            response: None,
            search: None,
        }
    }

    /// Creates a bundle entry with explicit full URL
    pub fn with_url(mut self, url: String) -> Self {
        self.full_url = Some(url);
        self
    }

    /// Adds a request to the entry (for transactions/batches)
    pub fn with_request(mut self, method: HttpVerb, url: String) -> Self {
        self.request = Some(BundleEntryRequest {
            method,
            url,
            if_none_match: None,
            if_modified_since: None,
            if_match: None,
        });
        self
    }

    /// Adds a response to the entry
    pub fn with_response(mut self, status: String) -> Self {
        self.response = Some(BundleEntryResponse {
            status,
            location: None,
            etag: None,
            last_modified: None,
        });
        self
    }

    /// Adds search information to the entry
    pub fn with_search(mut self, mode: String, score: Option<f64>) -> Self {
        self.search = Some(BundleEntrySearch {
            mode: Some(mode),
            score,
        });
        self
    }
}

/// FHIR Bundle resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bundle {
    /// Resource type (always "Bundle")
    #[serde(rename = "resourceType")]
    pub resource_type: String,

    /// Logical ID of the bundle
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Bundle type
    #[serde(rename = "type")]
    pub bundle_type: BundleType,

    /// Total number of matches (for search results)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,

    /// Links (for pagination, self-reference, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub link: Vec<BundleLink>,

    /// Entries in the bundle
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entry: Vec<BundleEntry>,

    /// Digital signature (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl Bundle {
    /// Creates a new empty bundle
    pub fn new(bundle_type: BundleType) -> Self {
        Self {
            resource_type: "Bundle".to_string(),
            id: None,
            bundle_type,
            total: None,
            link: Vec::new(),
            entry: Vec::new(),
            signature: None,
        }
    }

    /// Creates a document bundle
    pub fn document() -> Self {
        Self::new(BundleType::Document)
    }

    /// Creates a transaction bundle
    pub fn transaction() -> Self {
        Self::new(BundleType::Transaction)
    }

    /// Creates a batch bundle
    pub fn batch() -> Self {
        Self::new(BundleType::Batch)
    }

    /// Creates a search result bundle
    pub fn searchset() -> Self {
        Self::new(BundleType::Searchset)
    }

    /// Creates a collection bundle
    pub fn collection() -> Self {
        Self::new(BundleType::Collection)
    }

    /// Sets the bundle ID
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Adds a resource to the bundle
    pub fn add_resource(mut self, resource: FhirResource) -> Self {
        self.entry.push(BundleEntry::new(resource));
        self
    }

    /// Adds a bundle entry
    pub fn add_entry(mut self, entry: BundleEntry) -> Self {
        self.entry.push(entry);
        self
    }

    /// Adds a link to the bundle
    pub fn add_link(mut self, link: BundleLink) -> Self {
        self.link.push(link);
        self
    }

    /// Sets the total count (for search results)
    pub fn with_total(mut self, total: u64) -> Self {
        self.total = Some(total);
        self
    }

    /// Returns the number of entries in the bundle
    pub fn len(&self) -> usize {
        self.entry.len()
    }

    /// Checks if the bundle is empty
    pub fn is_empty(&self) -> bool {
        self.entry.is_empty()
    }

    /// Extracts all resources from the bundle
    pub fn resources(&self) -> Vec<&FhirResource> {
        self.entry
            .iter()
            .filter_map(|e| e.resource.as_ref())
            .collect()
    }

    /// Adds pagination links
    pub fn with_pagination(mut self, self_url: String, next_url: Option<String>, previous_url: Option<String>) -> Self {
        self.link.push(BundleLink::self_link(self_url));
        if let Some(next) = next_url {
            self.link.push(BundleLink::next(next));
        }
        if let Some(prev) = previous_url {
            self.link.push(BundleLink::previous(prev));
        }
        self
    }
}

impl Default for Bundle {
    fn default() -> Self {
        Self::new(BundleType::Collection)
    }
}

/// Builder for creating paginated bundles
pub struct PaginatedBundleBuilder {
    bundle: Bundle,
    page_size: usize,
    page_number: usize,
    base_url: String,
}

impl PaginatedBundleBuilder {
    /// Creates a new paginated bundle builder
    pub fn new(base_url: String, page_size: usize) -> Self {
        Self {
            bundle: Bundle::searchset(),
            page_size,
            page_number: 0,
            base_url,
        }
    }

    /// Sets the page number
    pub fn page(mut self, page_number: usize) -> Self {
        self.page_number = page_number;
        self
    }

    /// Adds resources to the bundle
    pub fn add_resources(mut self, resources: Vec<FhirResource>) -> Self {
        for resource in resources {
            self.bundle = self.bundle.add_resource(resource);
        }
        self
    }

    /// Sets the total count of results
    pub fn total(mut self, total: u64) -> Self {
        self.bundle = self.bundle.with_total(total);
        self
    }

    /// Builds the paginated bundle with appropriate links
    pub fn build(mut self) -> Bundle {
        let self_url = format!("{}?_count={}&_page={}",
            self.base_url, self.page_size, self.page_number);

        let next_url = if let Some(total) = self.bundle.total {
            let max_page = (total as usize + self.page_size - 1) / self.page_size;
            if self.page_number + 1 < max_page {
                Some(format!("{}?_count={}&_page={}",
                    self.base_url, self.page_size, self.page_number + 1))
            } else {
                None
            }
        } else {
            None
        };

        let prev_url = if self.page_number > 0 {
            Some(format!("{}?_count={}&_page={}",
                self.base_url, self.page_size, self.page_number - 1))
        } else {
            None
        };

        self.bundle = self.bundle.with_pagination(self_url, next_url, prev_url);
        self.bundle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fhir::resources::Patient;

    #[test]
    fn test_bundle_type_str() {
        assert_eq!(BundleType::Document.as_str(), "document");
        assert_eq!(BundleType::Transaction.as_str(), "transaction");
        assert_eq!(BundleType::Searchset.as_str(), "searchset");
    }

    #[test]
    fn test_bundle_creation() {
        let bundle = Bundle::collection().with_id("bundle-001".to_string());
        assert_eq!(bundle.resource_type, "Bundle");
        assert_eq!(bundle.id, Some("bundle-001".to_string()));
        assert_eq!(bundle.bundle_type, BundleType::Collection);
    }

    #[test]
    fn test_bundle_add_resource() {
        let patient = Patient::new("patient-001".to_string());
        let bundle = Bundle::collection()
            .add_resource(FhirResource::Patient(patient));

        assert_eq!(bundle.len(), 1);
        assert!(!bundle.is_empty());
    }

    #[test]
    fn test_bundle_entry() {
        let patient = Patient::new("patient-001".to_string());
        let entry = BundleEntry::new(FhirResource::Patient(patient))
            .with_url("http://example.org/Patient/patient-001".to_string());

        assert!(entry.resource.is_some());
        assert_eq!(entry.full_url, Some("http://example.org/Patient/patient-001".to_string()));
    }

    #[test]
    fn test_bundle_link() {
        let link = BundleLink::next("http://example.org/Patient?page=2".to_string());
        assert_eq!(link.relation, "next");
    }

    #[test]
    fn test_paginated_bundle() {
        let patient1 = Patient::new("patient-001".to_string());
        let patient2 = Patient::new("patient-002".to_string());

        let bundle = PaginatedBundleBuilder::new("http://example.org/Patient".to_string(), 10)
            .page(0)
            .total(2)
            .add_resources(vec![
                FhirResource::Patient(patient1),
                FhirResource::Patient(patient2),
            ])
            .build();

        assert_eq!(bundle.len(), 2);
        assert_eq!(bundle.total, Some(2));
        assert!(!bundle.link.is_empty());
    }

    #[test]
    fn test_transaction_bundle() {
        let patient = Patient::new("patient-001".to_string());
        let entry = BundleEntry::new(FhirResource::Patient(patient))
            .with_request(HttpVerb::Post, "Patient".to_string());

        let bundle = Bundle::transaction().add_entry(entry);
        assert_eq!(bundle.bundle_type, BundleType::Transaction);
    }
}
