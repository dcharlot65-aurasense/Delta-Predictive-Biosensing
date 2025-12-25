#!/bin/bash
# Generate API Schema from rustdoc JSON
#
# This script generates rustdoc JSON for all DPB crates and processes them
# into a simplified API schema that AI agents can consume.
#
# Usage:
#   ./scripts/generate-schema.sh
#   ./scripts/generate-schema.sh --openapi  # Generate OpenAPI spec
#
# Requirements:
#   - Rust nightly toolchain (for rustdoc JSON output)
#   - DPB workspace built successfully

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$WORKSPACE_ROOT/tools/ai-agent/output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}DPB API Schema Generator${NC}"
echo "========================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for nightly toolchain
if ! rustup run nightly rustc --version &> /dev/null; then
    echo -e "${YELLOW}Warning: Nightly toolchain not found. Installing...${NC}"
    rustup install nightly
fi

# List of crates to document
CRATES=(
    "dpb-core"
    "dpb-neurons"
    "dpb-encoders"
    "dpb-snn"
    "dpb-synth"
    "dpb-clinical"
    "dpb-norms"
    "dpb-viz"
    "dpb-lsl"
    "dpb-export"
    "dpb-federated"
    "dpb-cognitive"
    "dpb-bench"
)

echo "Generating rustdoc JSON for ${#CRATES[@]} crates..."
echo ""

# Generate rustdoc JSON for each crate
cd "$WORKSPACE_ROOT"
for crate in "${CRATES[@]}"; do
    echo -n "  Processing $crate... "
    if RUSTDOCFLAGS='-Z unstable-options --output-format json' \
       cargo +nightly doc -p "$crate" --no-deps 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}SKIP${NC}"
    fi
done

echo ""
echo "Processing JSON into API schema..."

# Build the schema generator
cd "$WORKSPACE_ROOT/tools/ai-agent"
cargo build --release --bin dpb-schema-gen 2>/dev/null

# Run the schema generator
OPENAPI_FLAG=""
if [[ "$1" == "--openapi" ]]; then
    OPENAPI_FLAG="--openapi"
fi

./target/release/dpb-schema-gen \
    --all \
    --output "$OUTPUT_DIR/api-schema.json" \
    --templates \
    --templates-output "$OUTPUT_DIR/code-templates.json" \
    --workspace "$WORKSPACE_ROOT" \
    $OPENAPI_FLAG

echo ""
echo -e "${GREEN}Schema generation complete!${NC}"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/api-schema.json"
echo "  - $OUTPUT_DIR/code-templates.json"
if [[ "$1" == "--openapi" ]]; then
    echo "  - Generated as OpenAPI 3.0 spec"
fi
