.PHONY: help test example build-ffi clean

help:
	@echo "DPB.jl Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build-ffi    - Build the DPB FFI library (Rust)"
	@echo "  test         - Run Julia tests"
	@echo "  example      - Run basic usage example"
	@echo "  clean        - Clean Julia build artifacts"
	@echo "  install      - Install package in Julia"
	@echo "  help         - Show this help message"

build-ffi:
	@echo "Building DPB FFI library..."
	cd ../../.. && cargo build --release -p dpb-ffi
	@echo "Done! Library at: ../../../target/release/libdpb_ffi.so"

test: build-ffi
	@echo "Running Julia tests..."
	julia --project=. -e 'using Pkg; Pkg.test()'

example: build-ffi
	@echo "Running basic usage example..."
	julia --project=. examples/basic_usage.jl

clean:
	@echo "Cleaning Julia artifacts..."
	rm -f Manifest.toml
	rm -rf .julia

install:
	@echo "Installing DPB.jl package..."
	julia --project=. -e 'using Pkg; Pkg.instantiate()'
	@echo "Done! Use 'using DPB' in Julia to load the package."

# Development helpers
dev:
	julia --project=.

repl: build-ffi
	@echo "Starting Julia REPL with DPB loaded..."
	julia --project=. -e 'using DPB'
