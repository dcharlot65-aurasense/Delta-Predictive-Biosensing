//! Build script for dpb-lsl
//!
//! This build script configures linking to the native liblsl library
//! when the `native` feature is enabled.

fn main() {
    // Only link liblsl when native feature is enabled
    #[cfg(feature = "native")]
    {
        // Try pkg-config first (Linux)
        if let Ok(_) = std::process::Command::new("pkg-config")
            .args(&["--exists", "lsl"])
            .status()
        {
            println!("cargo:rustc-link-lib=lsl");

            // Get lib path from pkg-config
            if let Ok(output) = std::process::Command::new("pkg-config")
                .args(&["--libs-only-L", "lsl"])
                .output()
            {
                let path = String::from_utf8_lossy(&output.stdout);
                for lib_path in path.split_whitespace() {
                    if lib_path.starts_with("-L") {
                        println!("cargo:rustc-link-search=native={}", &lib_path[2..]);
                    }
                }
            }

            return;
        }

        // Try common locations
        let search_paths = vec![
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/homebrew/lib",
            "/usr/local/opt/lsl/lib",
        ];

        for path in &search_paths {
            let lib_path = std::path::Path::new(path);
            if lib_path.join("liblsl.so").exists()
                || lib_path.join("liblsl.dylib").exists()
                || lib_path.join("lsl.dll").exists()
            {
                println!("cargo:rustc-link-search=native={}", path);
                println!("cargo:rustc-link-lib=lsl");
                return;
            }
        }

        // Check LSL_LIB environment variable
        if let Ok(lsl_lib) = std::env::var("LSL_LIB") {
            println!("cargo:rustc-link-search=native={}", lsl_lib);
            println!("cargo:rustc-link-lib=lsl");
            return;
        }

        // Provide helpful error message
        println!(
            "cargo:warning=liblsl not found. Install liblsl or set LSL_LIB environment variable."
        );
        println!("cargo:warning=On Ubuntu: sudo apt-get install liblsl-dev");
        println!("cargo:warning=On macOS: brew install labstreaminglayer/tap/lsl");
    }
}
