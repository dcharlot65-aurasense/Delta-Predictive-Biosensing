//! Build script for the Python extension module.
//!
//! When the `extension-module` feature is on, pyo3 deliberately does not link
//! against libpython: the Python symbols are left undefined for the host
//! interpreter to supply when the module is imported. Apple's linker rejects
//! undefined symbols by default, so a `cargo build --features extension-module`
//! fails at link time even though the code is correct.
//!
//! Wheel builders (maturin, setuptools-rust) pass these flags themselves, which
//! is why this only ever showed up on a plain cargo build -- and why the crate
//! could not be checked in CI without one.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let extension_module = std::env::var("CARGO_FEATURE_EXTENSION_MODULE").is_ok();
    let apple = std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple");

    if extension_module && apple {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
