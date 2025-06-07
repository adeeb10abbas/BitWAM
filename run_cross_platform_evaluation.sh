#!/bin/bash

# Cross-Platform BitNet Evaluation Helper Script
# 
# This script helps you run BitNet evaluations across different platforms
# and compare the results.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

# Check if Python script exists
if [ ! -f "cross_platform_bitnet_evaluation.py" ]; then
    print_error "cross_platform_bitnet_evaluation.py not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

# Check if required dependencies are available
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check PyTorch
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch is not installed."
        echo "Please install PyTorch: https://pytorch.org/get-started/locally/"
        exit 1
    fi
    
    # Check psutil
    if ! python3 -c "import psutil" &> /dev/null; then
        print_warning "psutil not found. Installing..."
        pip install psutil
    fi
    
    print_success "Dependencies checked!"
}

# Detect current platform
detect_platform() {
    print_info "Detecting platform..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            PLATFORM="Apple Silicon (M1/M2/M3)"
            PLATFORM_CODE="apple_arm"
        else
            PLATFORM="macOS Intel"
            PLATFORM_CODE="macos_intel"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v nvidia-smi &> /dev/null; then
            PLATFORM="Linux with NVIDIA GPU"
            PLATFORM_CODE="linux_nvidia"
        else
            PLATFORM="Linux CPU"
            PLATFORM_CODE="linux_cpu"
        fi
    else
        PLATFORM="Unknown"
        PLATFORM_CODE="unknown"
    fi
    
    print_success "Platform detected: $PLATFORM"
}

# Run benchmark
run_benchmark() {
    print_header "Running BitNet Cross-Platform Benchmark"
    
    print_info "This will test BitNet diffusion policy performance on your platform."
    print_info "Platform: $PLATFORM"
    
    # Ask for number of runs
    read -p "Number of benchmark runs (default: 50): " NUM_RUNS
    NUM_RUNS=${NUM_RUNS:-50}
    
    read -p "Number of warmup runs (default: 5): " WARMUP_RUNS
    WARMUP_RUNS=${WARMUP_RUNS:-5}
    
    print_info "Starting benchmark with $NUM_RUNS runs and $WARMUP_RUNS warmup runs..."
    
    # Run the benchmark
    python3 cross_platform_bitnet_evaluation.py \
        --mode benchmark \
        --runs $NUM_RUNS \
        --warmup $WARMUP_RUNS \
        --save-dir cross_platform_results
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed successfully!"
        
        # Find the most recent result file
        RESULT_FILE=$(ls -t cross_platform_results/bitnet_eval_*.json | head -1)
        print_success "Results saved to: $RESULT_FILE"
        
        echo ""
        print_info "To compare with other platforms:"
        echo "1. Copy this script and the result file to another machine"
        echo "2. Run the benchmark on that machine"
        echo "3. Use the compare mode to analyze differences"
        echo ""
        echo "Commands for next steps:"
        echo "  # On this machine - copy files:"
        echo "  scp cross_platform_bitnet_evaluation.py user@othermachine:~/"
        echo "  scp $RESULT_FILE user@othermachine:~/"
        echo ""
        echo "  # On other machine - run benchmark:"
        echo "  python3 cross_platform_bitnet_evaluation.py --mode benchmark"
        echo ""
        echo "  # Compare results:"
        echo "  python3 cross_platform_bitnet_evaluation.py --mode compare --compare-files result1.json result2.json"
        
    else
        print_error "Benchmark failed!"
        exit 1
    fi
}

# Compare results
compare_results() {
    print_header "Comparing Cross-Platform Results"
    
    # Check if results directory exists
    if [ ! -d "cross_platform_results" ]; then
        print_error "No results directory found. Run benchmark first."
        exit 1
    fi
    
    # List available result files
    RESULT_FILES=($(ls cross_platform_results/bitnet_eval_*.json 2>/dev/null))
    
    if [ ${#RESULT_FILES[@]} -eq 0 ]; then
        print_error "No result files found in cross_platform_results/"
        exit 1
    fi
    
    echo "Available result files:"
    for i in "${!RESULT_FILES[@]}"; do
        echo "  $((i+1)). ${RESULT_FILES[$i]}"
    done
    echo ""
    
    if [ ${#RESULT_FILES[@]} -lt 2 ]; then
        print_warning "Need at least 2 result files for comparison."
        print_info "Copy result files from other machines to cross_platform_results/"
        exit 1
    fi
    
    # Let user select files to compare
    echo "Select files to compare (enter numbers separated by spaces, e.g., '1 2'):"
    read -p "Files to compare: " SELECTED
    
    # Parse selected files
    COMPARE_FILES=()
    for num in $SELECTED; do
        if [ $num -gt 0 ] && [ $num -le ${#RESULT_FILES[@]} ]; then
            COMPARE_FILES+=("${RESULT_FILES[$((num-1))]}")
        fi
    done
    
    if [ ${#COMPARE_FILES[@]} -lt 2 ]; then
        print_error "Need to select at least 2 files for comparison."
        exit 1
    fi
    
    print_info "Comparing ${#COMPARE_FILES[@]} result files..."
    
    # Run comparison
    python3 cross_platform_bitnet_evaluation.py \
        --mode compare \
        --compare-files "${COMPARE_FILES[@]}"
    
    if [ $? -eq 0 ]; then
        print_success "Comparison completed!"
    else
        print_error "Comparison failed!"
        exit 1
    fi
}

# Install dependencies automatically
install_deps() {
    print_header "Installing Dependencies"
    
    print_info "Installing required Python packages..."
    
    # Create requirements list
    REQUIREMENTS=(
        "torch"
        "psutil"
        "numpy"
    )
    
    for req in "${REQUIREMENTS[@]}"; do
        print_info "Installing $req..."
        pip install "$req"
    done
    
    print_success "Dependencies installed!"
}

# Show help
show_help() {
    echo "Cross-Platform BitNet Evaluation Helper"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  benchmark    Run benchmark on current platform (default)"
    echo "  compare      Compare results from multiple platforms"
    echo "  install      Install required dependencies"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run benchmark with interactive prompts"
    echo "  $0 benchmark          # Same as above"
    echo "  $0 compare            # Compare existing results"
    echo "  $0 install            # Install dependencies"
    echo ""
    echo "Cross-platform workflow:"
    echo "  1. Run '$0 benchmark' on first machine (e.g., NVIDIA GPU)"
    echo "  2. Copy script and result file to second machine (e.g., Apple Silicon)"
    echo "  3. Run '$0 benchmark' on second machine"
    echo "  4. Run '$0 compare' to analyze differences"
}

# Main script logic
main() {
    # Parse command line arguments
    COMMAND=${1:-benchmark}
    
    case $COMMAND in
        "benchmark")
            check_dependencies
            detect_platform
            run_benchmark
            ;;
        "compare")
            compare_results
            ;;
        "install")
            install_deps
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 