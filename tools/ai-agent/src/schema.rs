//! API Schema extraction from rustdoc JSON
//!
//! This module processes rustdoc JSON output and produces a simplified,
//! AI-friendly schema describing all public functions, types, and their documentation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result};

/// Complete API schema for all DPB crates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSchema {
    /// Schema version for compatibility checking
    pub schema_version: String,
    /// Generation timestamp
    pub generated_at: String,
    /// Framework version
    pub framework_version: String,
    /// Individual crate schemas
    pub crates: HashMap<String, CrateSchema>,
}

impl ApiSchema {
    /// Create a new empty API schema
    pub fn new(framework_version: &str) -> Self {
        Self {
            schema_version: "1.0.0".to_string(),
            generated_at: chrono_lite_now(),
            framework_version: framework_version.to_string(),
            crates: HashMap::new(),
        }
    }

    /// Load from a JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read schema from {:?}", path))?;
        serde_json::from_str(&content)
            .with_context(|| "Failed to parse API schema JSON")
    }

    /// Save to a JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize API schema")?;
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write schema to {:?}", path))?;
        Ok(())
    }

    /// Get a flattened list of all types across all crates
    pub fn all_types(&self) -> Vec<(&str, &TypeDef)> {
        self.crates
            .iter()
            .flat_map(|(crate_name, schema)| {
                schema.types.iter().map(move |t| (crate_name.as_str(), t))
            })
            .collect()
    }

    /// Get a flattened list of all functions across all crates
    pub fn all_functions(&self) -> Vec<(&str, &FunctionDef)> {
        self.crates
            .iter()
            .flat_map(|(crate_name, schema)| {
                schema.functions.iter().map(move |f| (crate_name.as_str(), f))
            })
            .collect()
    }

    /// Search for types by name pattern
    pub fn search_types(&self, pattern: &str) -> Vec<(&str, &TypeDef)> {
        let pattern_lower = pattern.to_lowercase();
        self.all_types()
            .into_iter()
            .filter(|(_, t)| t.name.to_lowercase().contains(&pattern_lower))
            .collect()
    }

    /// Search for functions by name pattern
    pub fn search_functions(&self, pattern: &str) -> Vec<(&str, &FunctionDef)> {
        let pattern_lower = pattern.to_lowercase();
        self.all_functions()
            .into_iter()
            .filter(|(_, f)| f.name.to_lowercase().contains(&pattern_lower))
            .collect()
    }
}

/// Schema for a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateSchema {
    /// Crate name (e.g., "dpb-core")
    pub name: String,
    /// Crate version
    pub version: String,
    /// Crate-level documentation
    pub docs: Option<String>,
    /// Public modules
    pub modules: Vec<ModuleDef>,
    /// Public types (structs, enums, traits)
    pub types: Vec<TypeDef>,
    /// Public functions
    pub functions: Vec<FunctionDef>,
    /// Re-exports
    pub reexports: Vec<ReexportDef>,
}

impl CrateSchema {
    /// Create a new empty crate schema
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            docs: None,
            modules: Vec::new(),
            types: Vec::new(),
            functions: Vec::new(),
            reexports: Vec::new(),
        }
    }
}

/// Module definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDef {
    /// Module path (e.g., "signal::ecg")
    pub path: String,
    /// Module documentation
    pub docs: Option<String>,
    /// Visibility
    pub visibility: Visibility,
}

/// Type definition (struct, enum, or trait)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDef {
    /// Type name
    pub name: String,
    /// Full path (e.g., "dpb_core::signal::ecg::RpeakDetector")
    pub path: String,
    /// Kind of type
    pub kind: TypeKind,
    /// Type documentation
    pub docs: Option<String>,
    /// Generic parameters
    pub generics: Vec<GenericParam>,
    /// Fields (for structs) or variants (for enums)
    pub fields: Vec<FieldDef>,
    /// Methods
    pub methods: Vec<FunctionDef>,
    /// Implemented traits
    pub traits: Vec<String>,
    /// Visibility
    pub visibility: Visibility,
}

/// Kind of type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Struct,
    Enum,
    Trait,
    TypeAlias,
    Union,
}

/// Field or variant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    /// Field name (None for tuple structs)
    pub name: Option<String>,
    /// Field type
    pub ty: String,
    /// Field documentation
    pub docs: Option<String>,
    /// Default value (if any)
    pub default: Option<String>,
}

/// Function/method definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name
    pub name: String,
    /// Full path
    pub path: String,
    /// Function documentation
    pub docs: Option<String>,
    /// Function signature
    pub signature: String,
    /// Parameters
    pub params: Vec<ParamDef>,
    /// Return type
    pub return_type: Option<String>,
    /// Generic parameters
    pub generics: Vec<GenericParam>,
    /// Whether this is async
    pub is_async: bool,
    /// Whether this is unsafe
    pub is_unsafe: bool,
    /// Whether this is const
    pub is_const: bool,
    /// Visibility
    pub visibility: Visibility,
}

/// Parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: String,
    /// Whether this is a reference
    pub is_ref: bool,
    /// Whether this is mutable
    pub is_mut: bool,
}

/// Generic parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericParam {
    /// Parameter name
    pub name: String,
    /// Bounds (trait constraints)
    pub bounds: Vec<String>,
    /// Default type (if any)
    pub default: Option<String>,
}

/// Re-export definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReexportDef {
    /// Public name
    pub name: String,
    /// Original path
    pub original_path: String,
}

/// Visibility level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Crate,
    Super,
    Private,
}

// ============================================================================
// Rustdoc JSON parsing
// ============================================================================

/// Rustdoc JSON format (simplified representation)
#[derive(Debug, Clone, Deserialize)]
pub struct RustdocJson {
    pub root: String,
    pub crate_version: Option<String>,
    pub includes_private: bool,
    pub index: HashMap<String, RustdocItem>,
    pub paths: HashMap<String, RustdocPath>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocItem {
    pub id: String,
    pub crate_id: u32,
    pub name: Option<String>,
    pub span: Option<RustdocSpan>,
    pub visibility: String,
    pub docs: Option<String>,
    pub attrs: Vec<String>,
    pub inner: RustdocInner,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocSpan {
    pub filename: String,
    pub begin: (u32, u32),
    pub end: (u32, u32),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", content = "inner", rename_all = "snake_case")]
pub enum RustdocInner {
    Module(RustdocModule),
    Struct(RustdocStruct),
    Enum(RustdocEnum),
    Trait(RustdocTrait),
    Function(RustdocFunction),
    Method(RustdocMethod),
    TypeAlias(RustdocTypeAlias),
    Impl(RustdocImpl),
    Import(RustdocImport),
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocModule {
    pub items: Vec<String>,
    pub is_stripped: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocStruct {
    pub kind: RustdocStructKind,
    pub generics: RustdocGenerics,
    pub impls: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RustdocStructKind {
    Unit,
    Tuple { fields: Vec<Option<String>> },
    Plain { fields: Vec<String>, has_stripped_fields: bool },
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocEnum {
    pub generics: RustdocGenerics,
    pub variants: Vec<String>,
    pub impls: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocTrait {
    pub is_auto: bool,
    pub is_unsafe: bool,
    pub items: Vec<String>,
    pub generics: RustdocGenerics,
    pub bounds: Vec<RustdocGenericBound>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocFunction {
    pub sig: RustdocFnSig,
    pub generics: RustdocGenerics,
    pub has_body: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocMethod {
    pub sig: RustdocFnSig,
    pub generics: RustdocGenerics,
    pub has_body: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocFnSig {
    pub inputs: Vec<(String, RustdocType)>,
    pub output: Option<RustdocType>,
    pub is_c_variadic: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocTypeAlias {
    pub r#type: RustdocType,
    pub generics: RustdocGenerics,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocImpl {
    pub is_unsafe: bool,
    pub generics: RustdocGenerics,
    pub provided_trait_methods: Vec<String>,
    pub trait_: Option<RustdocPath>,
    pub for_: RustdocType,
    pub items: Vec<String>,
    pub is_negative: bool,
    pub is_synthetic: bool,
    pub blanket_impl: Option<RustdocType>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocImport {
    pub source: String,
    pub name: String,
    pub id: Option<String>,
    pub is_glob: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocGenerics {
    pub params: Vec<RustdocGenericParam>,
    pub where_predicates: Vec<RustdocWherePredicate>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocGenericParam {
    pub name: String,
    pub kind: RustdocGenericParamKind,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RustdocGenericParamKind {
    Lifetime { outlives: Vec<String> },
    Type { bounds: Vec<RustdocGenericBound>, default: Option<RustdocType>, is_synthetic: bool },
    Const { type_: RustdocType, default: Option<String> },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RustdocGenericBound {
    TraitBound { trait_: RustdocPath, generic_params: Vec<RustdocGenericParam>, modifier: String },
    Outlives { lifetime: String },
    Use { args: Vec<String> },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RustdocWherePredicate {
    BoundPredicate { type_: RustdocType, bounds: Vec<RustdocGenericBound>, generic_params: Vec<RustdocGenericParam> },
    LifetimePredicate { lifetime: String, outlives: Vec<String> },
    EqPredicate { lhs: RustdocType, rhs: RustdocType },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RustdocType {
    Primitive(String),
    Complex(Box<RustdocTypeComplex>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocTypeComplex {
    // Simplified - actual rustdoc JSON has many type variants
    pub name: Option<String>,
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustdocPath {
    pub name: String,
    pub id: Option<String>,
    pub args: Option<RustdocGenericArgs>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RustdocGenericArgs {
    AngleBracketed { args: Vec<RustdocGenericArg>, bindings: Vec<serde_json::Value> },
    Parenthesized { inputs: Vec<RustdocType>, output: Option<RustdocType> },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RustdocGenericArg {
    Lifetime { lifetime: String },
    Type { type_: RustdocType },
    Const { const_: serde_json::Value },
    Infer,
}

// ============================================================================
// Schema extraction from rustdoc JSON
// ============================================================================

/// Extract a CrateSchema from rustdoc JSON
pub fn extract_crate_schema(json_path: &Path) -> Result<CrateSchema> {
    let content = std::fs::read_to_string(json_path)
        .with_context(|| format!("Failed to read rustdoc JSON from {:?}", json_path))?;

    let rustdoc: RustdocJson = serde_json::from_str(&content)
        .with_context(|| "Failed to parse rustdoc JSON")?;

    let root_item = rustdoc.index.get(&rustdoc.root)
        .with_context(|| "Root item not found in rustdoc JSON")?;

    let crate_name = root_item.name.clone().unwrap_or_else(|| "unknown".to_string());
    let version = rustdoc.crate_version.clone().unwrap_or_else(|| "0.0.0".to_string());

    let mut schema = CrateSchema::new(&crate_name, &version);
    schema.docs = root_item.docs.clone();

    // Process all items
    extract_items(&rustdoc, &rustdoc.root, "", &mut schema);

    Ok(schema)
}

fn extract_items(rustdoc: &RustdocJson, item_id: &str, path_prefix: &str, schema: &mut CrateSchema) {
    let Some(item) = rustdoc.index.get(item_id) else { return };

    let name = item.name.clone().unwrap_or_default();
    let current_path = if path_prefix.is_empty() {
        name.clone()
    } else {
        format!("{}::{}", path_prefix, name)
    };

    let visibility = parse_visibility(&item.visibility);
    if visibility == Visibility::Private {
        return;
    }

    match &item.inner {
        RustdocInner::Module(m) => {
            if !m.is_stripped.unwrap_or(false) {
                schema.modules.push(ModuleDef {
                    path: current_path.clone(),
                    docs: item.docs.clone(),
                    visibility: visibility.clone(),
                });

                for child_id in &m.items {
                    extract_items(rustdoc, child_id, &current_path, schema);
                }
            }
        }
        RustdocInner::Struct(s) => {
            let fields = extract_struct_fields(rustdoc, s);
            let methods = extract_impl_methods(rustdoc, &s.impls);
            let traits = extract_impl_traits(rustdoc, &s.impls);

            schema.types.push(TypeDef {
                name: name.clone(),
                path: current_path,
                kind: TypeKind::Struct,
                docs: item.docs.clone(),
                generics: extract_generics(&s.generics),
                fields,
                methods,
                traits,
                visibility,
            });
        }
        RustdocInner::Enum(e) => {
            let fields = extract_enum_variants(rustdoc, &e.variants);
            let methods = extract_impl_methods(rustdoc, &e.impls);
            let traits = extract_impl_traits(rustdoc, &e.impls);

            schema.types.push(TypeDef {
                name: name.clone(),
                path: current_path,
                kind: TypeKind::Enum,
                docs: item.docs.clone(),
                generics: extract_generics(&e.generics),
                fields,
                methods,
                traits,
                visibility,
            });
        }
        RustdocInner::Trait(t) => {
            let methods = extract_trait_methods(rustdoc, &t.items);

            schema.types.push(TypeDef {
                name: name.clone(),
                path: current_path,
                kind: TypeKind::Trait,
                docs: item.docs.clone(),
                generics: extract_generics(&t.generics),
                fields: Vec::new(),
                methods,
                traits: Vec::new(),
                visibility,
            });
        }
        RustdocInner::Function(f) => {
            schema.functions.push(FunctionDef {
                name: name.clone(),
                path: current_path,
                docs: item.docs.clone(),
                signature: format_fn_signature(&name, &f.sig, &f.generics),
                params: extract_params(&f.sig.inputs),
                return_type: format_type(&f.sig.output),
                generics: extract_generics(&f.generics),
                is_async: false, // Would need to check attrs
                is_unsafe: false,
                is_const: false,
                visibility,
            });
        }
        RustdocInner::TypeAlias(ta) => {
            schema.types.push(TypeDef {
                name: name.clone(),
                path: current_path,
                kind: TypeKind::TypeAlias,
                docs: item.docs.clone(),
                generics: extract_generics(&ta.generics),
                fields: Vec::new(),
                methods: Vec::new(),
                traits: Vec::new(),
                visibility,
            });
        }
        RustdocInner::Import(i) => {
            if !i.is_glob {
                schema.reexports.push(ReexportDef {
                    name: i.name.clone(),
                    original_path: i.source.clone(),
                });
            }
        }
        _ => {}
    }
}

fn parse_visibility(vis: &str) -> Visibility {
    match vis {
        "public" => Visibility::Public,
        "crate" => Visibility::Crate,
        s if s.starts_with("restricted") => Visibility::Super,
        _ => Visibility::Private,
    }
}

fn extract_generics(generics: &RustdocGenerics) -> Vec<GenericParam> {
    generics.params.iter().filter_map(|p| {
        match &p.kind {
            RustdocGenericParamKind::Type { bounds, default, .. } => {
                Some(GenericParam {
                    name: p.name.clone(),
                    bounds: bounds.iter().filter_map(|b| format_generic_bound(b)).collect(),
                    default: default.as_ref().and_then(|t| format_type(&Some(t.clone()))),
                })
            }
            RustdocGenericParamKind::Const { type_, default } => {
                Some(GenericParam {
                    name: p.name.clone(),
                    bounds: vec![format!("const: {}", format_type(&Some(type_.clone())).unwrap_or_default())],
                    default: default.clone(),
                })
            }
            _ => None,
        }
    }).collect()
}

fn format_generic_bound(bound: &RustdocGenericBound) -> Option<String> {
    match bound {
        RustdocGenericBound::TraitBound { trait_, .. } => Some(trait_.name.clone()),
        RustdocGenericBound::Outlives { lifetime } => Some(format!("'{}", lifetime)),
        _ => None,
    }
}

fn extract_struct_fields(rustdoc: &RustdocJson, s: &RustdocStruct) -> Vec<FieldDef> {
    match &s.kind {
        RustdocStructKind::Plain { fields, .. } => {
            fields.iter().filter_map(|field_id| {
                let item = rustdoc.index.get(field_id)?;
                Some(FieldDef {
                    name: item.name.clone(),
                    ty: "unknown".to_string(), // Would need full type parsing
                    docs: item.docs.clone(),
                    default: None,
                })
            }).collect()
        }
        RustdocStructKind::Tuple { fields } => {
            fields.iter().enumerate().map(|(i, _)| {
                FieldDef {
                    name: Some(format!("{}", i)),
                    ty: "unknown".to_string(),
                    docs: None,
                    default: None,
                }
            }).collect()
        }
        RustdocStructKind::Unit => Vec::new(),
    }
}

fn extract_enum_variants(rustdoc: &RustdocJson, variants: &[String]) -> Vec<FieldDef> {
    variants.iter().filter_map(|variant_id| {
        let item = rustdoc.index.get(variant_id)?;
        Some(FieldDef {
            name: item.name.clone(),
            ty: "variant".to_string(),
            docs: item.docs.clone(),
            default: None,
        })
    }).collect()
}

fn extract_impl_methods(rustdoc: &RustdocJson, impls: &[String]) -> Vec<FunctionDef> {
    impls.iter().flat_map(|impl_id| {
        let Some(item) = rustdoc.index.get(impl_id) else { return Vec::new() };
        let RustdocInner::Impl(imp) = &item.inner else { return Vec::new() };

        // Skip trait impls for method extraction
        if imp.trait_.is_some() {
            return Vec::new();
        }

        imp.items.iter().filter_map(|method_id| {
            let method_item = rustdoc.index.get(method_id)?;
            let name = method_item.name.clone()?;

            let (sig, generics) = match &method_item.inner {
                RustdocInner::Method(m) => (&m.sig, &m.generics),
                RustdocInner::Function(f) => (&f.sig, &f.generics),
                _ => return None,
            };

            Some(FunctionDef {
                name: name.clone(),
                path: name.clone(),
                docs: method_item.docs.clone(),
                signature: format_fn_signature(&name, sig, generics),
                params: extract_params(&sig.inputs),
                return_type: format_type(&sig.output),
                generics: extract_generics(generics),
                is_async: false,
                is_unsafe: false,
                is_const: false,
                visibility: parse_visibility(&method_item.visibility),
            })
        }).collect()
    }).collect()
}

fn extract_impl_traits(rustdoc: &RustdocJson, impls: &[String]) -> Vec<String> {
    impls.iter().filter_map(|impl_id| {
        let item = rustdoc.index.get(impl_id)?;
        let RustdocInner::Impl(imp) = &item.inner else { return None };
        imp.trait_.as_ref().map(|t| t.name.clone())
    }).collect()
}

fn extract_trait_methods(rustdoc: &RustdocJson, items: &[String]) -> Vec<FunctionDef> {
    items.iter().filter_map(|item_id| {
        let item = rustdoc.index.get(item_id)?;
        let name = item.name.clone()?;

        let (sig, generics) = match &item.inner {
            RustdocInner::Method(m) => (&m.sig, &m.generics),
            RustdocInner::Function(f) => (&f.sig, &f.generics),
            _ => return None,
        };

        Some(FunctionDef {
            name: name.clone(),
            path: name.clone(),
            docs: item.docs.clone(),
            signature: format_fn_signature(&name, sig, generics),
            params: extract_params(&sig.inputs),
            return_type: format_type(&sig.output),
            generics: extract_generics(generics),
            is_async: false,
            is_unsafe: false,
            is_const: false,
            visibility: parse_visibility(&item.visibility),
        })
    }).collect()
}

fn extract_params(inputs: &[(String, RustdocType)]) -> Vec<ParamDef> {
    inputs.iter().map(|(name, ty)| {
        ParamDef {
            name: name.clone(),
            ty: format_type(&Some(ty.clone())).unwrap_or_else(|| "unknown".to_string()),
            is_ref: false, // Would need deeper type analysis
            is_mut: false,
        }
    }).collect()
}

fn format_type(ty: &Option<RustdocType>) -> Option<String> {
    ty.as_ref().map(|t| match t {
        RustdocType::Primitive(s) => s.clone(),
        RustdocType::Complex(c) => c.name.clone().unwrap_or_else(|| "complex".to_string()),
    })
}

fn format_fn_signature(name: &str, sig: &RustdocFnSig, _generics: &RustdocGenerics) -> String {
    let params: Vec<String> = sig.inputs.iter()
        .map(|(name, ty)| format!("{}: {}", name, format_type(&Some(ty.clone())).unwrap_or_default()))
        .collect();

    let ret = sig.output.as_ref()
        .and_then(|t| format_type(&Some(t.clone())))
        .map(|t| format!(" -> {}", t))
        .unwrap_or_default();

    format!("fn {}({}){}", name, params.join(", "), ret)
}

/// Simple timestamp without chrono dependency
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_schema_new() {
        let schema = ApiSchema::new("0.1.0");
        assert_eq!(schema.framework_version, "0.1.0");
        assert_eq!(schema.schema_version, "1.0.0");
    }

    #[test]
    fn test_crate_schema_new() {
        let schema = CrateSchema::new("dpb-core", "0.1.0");
        assert_eq!(schema.name, "dpb-core");
        assert_eq!(schema.version, "0.1.0");
    }
}
