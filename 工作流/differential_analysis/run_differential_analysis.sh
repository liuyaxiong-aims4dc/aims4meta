#!/bin/bash
#
# Master script for differential compound analysis
# Traditional Chinese Medicine: Authentic vs Counterfeit
#
# Usage:
#   bash run_differential_analysis.sh --group-a <dir> --group-b <dir> --output <dir> [options]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./differential_analysis_results"
SIMILARITY_THRESHOLD=0.85
L2_DATABASES="tcmbank herb coconut"
CONFIG_FILE="config.yaml"

# Print header
echo ""
echo "========================================================================"
echo "  中药正品与伪品差异化合物分析"
echo "  Traditional Chinese Medicine Differential Analysis"
echo "  Powered by DreaMS"
echo "========================================================================"
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --group-a)
            GROUP_A="$2"
            shift 2
            ;;
        --group-b)
            GROUP_B="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --similarity-threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        --l2-databases)
            L2_DATABASES="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_differential_analysis.sh [OPTIONS]"
            echo ""
            echo "Required options:"
            echo "  --group-a DIR          Directory containing Group A (authentic) MSP files"
            echo "  --group-b DIR          Directory containing Group B (counterfeit) MSP files"
            echo ""
            echo "Optional options:"
            echo "  --output DIR           Output directory (default: ./differential_analysis_results)"
            echo "  --similarity-threshold Similarity threshold for unique compounds (default: 0.85)"
            echo "  --l2-databases LIST    L2 databases to search (default: 'tcmbank herb coconut')"
            echo "  --config FILE          DreaMS config file (default: config.yaml)"
            echo "  --help                 Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$GROUP_A" ] || [ -z "$GROUP_B" ]; then
    echo -e "${RED}Error: --group-a and --group-b are required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Check if directories exist
if [ ! -d "$GROUP_A" ]; then
    echo -e "${RED}Error: Group A directory does not exist: $GROUP_A${NC}"
    exit 1
fi

if [ ! -d "$GROUP_B" ]; then
    echo -e "${RED}Error: Group B directory does not exist: $GROUP_B${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Group A (Authentic): $GROUP_A"
echo "  Group B (Counterfeit): $GROUP_B"
echo "  Output directory: $OUTPUT_DIR"
echo "  Similarity threshold: $SIMILARITY_THRESHOLD"
echo "  L2 databases: $L2_DATABASES"
echo "  Config file: $CONFIG_FILE"
echo ""

# Activate conda environment
echo -e "${YELLOW}[Setup] Activating conda environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dreams
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Step 1: Generate DreaMS embeddings
echo -e "${YELLOW}[Step 1/3] Generating DreaMS embeddings...${NC}"
python "${SCRIPT_DIR}/01_generate_embeddings.py" \
    --group-a "$GROUP_A" \
    --group-b "$GROUP_B" \
    --output "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 1 complete${NC}"
else
    echo -e "${RED}✗ Step 1 failed${NC}"
    exit 1
fi
echo ""

# Step 2: Differential analysis
echo -e "${YELLOW}[Step 2/3] Performing differential analysis...${NC}"
python "${SCRIPT_DIR}/02_differential_analysis.py" \
    --embeddings "${OUTPUT_DIR}/embeddings/embeddings.npz" \
    --metadata "${OUTPUT_DIR}/spectrum_metadata.csv" \
    --output "$OUTPUT_DIR" \
    --similarity-threshold "$SIMILARITY_THRESHOLD"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 2 complete${NC}"
else
    echo -e "${RED}✗ Step 2 failed${NC}"
    exit 1
fi
echo ""

# Step 3: Generate final report
echo -e "${YELLOW}[Step 3/3] Generating final report...${NC}"
python "${SCRIPT_DIR}/04_generate_report.py" \
    --results-dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 3 complete${NC}"
else
    echo -e "${RED}✗ Step 3 failed${NC}"
    exit 1
fi
echo ""

# Final summary
echo ""
echo "========================================================================"
echo -e "${GREEN}✓ DIFFERENTIAL ANALYSIS COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - Comprehensive report: ${OUTPUT_DIR}/COMPREHENSIVE_DIFFERENTIAL_REPORT.txt"
echo "  - Excel summary: ${OUTPUT_DIR}/DIFFERENTIAL_ANALYSIS_SUMMARY.xlsx"
echo "  - PCA plot: ${OUTPUT_DIR}/visualization/pca_plot.png"
echo "  - Statistics: ${OUTPUT_DIR}/statistics/"
echo ""
echo "Next steps:"
echo "  1. Review the comprehensive report"
echo "  2. Examine the Excel summary for detailed data"
echo "  3. For compound identification, use the separate identification workflow"
echo ""
