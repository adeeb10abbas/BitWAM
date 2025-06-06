#!/bin/bash
# 🤖 1bit_vla + LeRobot Integration Examples
# 
# This script provides easy commands to run various integration examples
# and comparisons between BitNet and standard VLA models.

set -e  # Exit on any error

# Handle OpenMP conflicts automatically
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🤖 1bit_vla + LeRobot Integration Examples"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
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

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ] || [ ! -d "src/bit_vla" ]; then
        print_error "Please run this script from the 1bit_vla root directory"
        exit 1
    fi
    
    # Check if lerobot is available
    if [ ! -d "../lerobot" ]; then
        print_warning "LeRobot not found in ../lerobot"
        echo "Please clone LeRobot:"
        echo "cd .. && git clone https://github.com/huggingface/lerobot.git"
        exit 1
    fi
    
    # Check Python packages
    python -c "import torch" 2>/dev/null || {
        print_error "PyTorch not installed. Please install: pip install torch"
        exit 1
    }
    
    print_success "Dependencies check passed"
}

# Install 1bit_vla in development mode
install_package() {
    print_header "Installing 1bit_vla Package"
    pip install -e . || {
        print_error "Failed to install 1bit_vla package"
        exit 1
    }
    print_success "Package installed successfully"
}

# Run basic integration example
run_basic_example() {
    print_header "Running Basic Integration Example"
    print_warning "This will download the PushT dataset (~100MB)"
    
    cd examples
    python lerobot_integration_example.py || {
        print_error "Basic example failed"
        exit 1
    }
    cd ..
    
    print_success "Basic example completed! Check lerobot_integration_results.json"
}

# Run training with comparison
run_training_comparison() {
    local dataset=${1:-"lerobot/pusht"}
    local epochs=${2:-5}
    
    print_header "Running Training Comparison on $dataset"
    print_warning "This will train both BitNet and standard models"
    
    python scripts/train_with_lerobot.py \
        --dataset "$dataset" \
        --model bitact \
        --use_bitnet \
        --compare_standard \
        --num_epochs "$epochs" \
        --output_dir "outputs/comparison_$(date +%Y%m%d_%H%M%S)" || {
        print_error "Training comparison failed"
        exit 1
    }
    
    print_success "Training comparison completed!"
}

# Run with W&B logging
run_with_wandb() {
    local dataset=${1:-"lerobot/pusht"}
    
    print_header "Running with Weights & Biases Logging"
    
    # Check if wandb is available
    python -c "import wandb" 2>/dev/null || {
        print_warning "W&B not installed. Installing..."
        pip install wandb
    }
    
    python scripts/train_with_lerobot.py \
        --dataset "$dataset" \
        --model bitact \
        --use_bitnet \
        --num_epochs 10 \
        --use_wandb \
        --wandb_project "1bit_vla_lerobot_$(whoami)" || {
        print_error "W&B training failed"
        exit 1
    }
    
    print_success "W&B training completed!"
}

# Quick test on multiple datasets
run_dataset_sweep() {
    print_header "Running Quick Test on Multiple Datasets"
    
    datasets=("lerobot/pusht" "lerobot/pusht_image")
    
    for dataset in "${datasets[@]}"; do
        echo -e "\n${YELLOW}Testing $dataset...${NC}"
        
        python scripts/train_with_lerobot.py \
            --dataset "$dataset" \
            --model bitact \
            --use_bitnet \
            --num_epochs 2 \
            --output_dir "outputs/sweep_$(basename $dataset)_$(date +%Y%m%d_%H%M%S)" || {
            print_warning "Failed on $dataset, continuing..."
            continue
        }
        
        print_success "Completed $dataset"
    done
    
    print_success "Dataset sweep completed!"
}

# Analyze existing results
analyze_results() {
    print_header "Analyzing Existing Results"
    
    if [ -d "outputs" ]; then
        echo "Found output directories:"
        find outputs -name "results.json" -exec echo "📊 {}" \;
        find outputs -name "comparison_report.json" -exec echo "📈 {}" \;
        
        # Show latest comparison if available
        latest_comparison=$(find outputs -name "comparison_report.json" | head -1)
        if [ -n "$latest_comparison" ]; then
            echo -e "\n${BLUE}Latest Comparison Results:${NC}"
            python -c "
import json
try:
    with open('$latest_comparison') as f:
        data = json.load(f)
    print('Performance Gap:', data.get('detailed_comparison', {}).get('performance_gap_percent', 'N/A'), '%')
    print('Recommendations:')
    for rec in data.get('recommendations', []):
        print('  -', rec)
except Exception as e:
    print('Error reading results:', e)
"
        fi
    else
        print_warning "No output directory found. Run some experiments first!"
    fi
}

# Run speed benchmark
run_speed_benchmark() {
    local device=${1:-"cpu"}
    
    print_header "Running Speed Benchmark"
    print_warning "This will measure inference speed on $device"
    
    python examples/speed_benchmark.py \
        --device "$device" \
        --save_results \
        --output_file "outputs/speed_benchmark_${device}_$(date +%Y%m%d_%H%M%S).json" || {
        print_error "Speed benchmark failed"
        exit 1
    }
    
    print_success "Speed benchmark completed!"
}

# Run speed comparison on multiple datasets
run_speed_comparison() {
    print_header "Running Speed Comparison on Multiple Datasets"
    
    datasets=("lerobot/pusht")
    
    for dataset in "${datasets[@]}"; do
        echo -e "\n${YELLOW}Speed testing $dataset...${NC}"
        
        # Create a temporary script to test speed with dataset
        cat > temp_speed_test.py << EOF
import sys
sys.path.append('src')
sys.path.append('../lerobot')

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from bit_vla import BitACTPolicy, BitACTConfig
import torch
import time

# Load dataset to get dimensions
dataset = LeRobotDataset("$dataset", video_backend="pyav")
sample = dataset[0]
obs_dim = sample["observation.state"].flatten().shape[0]

print(f"Dataset: $dataset")
print(f"Observation dimension: {obs_dim}")

# Quick speed test
device = torch.device("cpu")
config = BitACTConfig(action_dim=2, use_bitnet=True)
model = BitACTPolicy(config, observation_dim=obs_dim).to(device)

# Time single inference
model.eval()
dummy_input = torch.randn(1, obs_dim, device=device)

with torch.no_grad():
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Time it
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = model(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

mean_time = sum(times) / len(times)
print(f"Average inference time: {mean_time:.2f}ms")
print(f"Max frequency: {1000/mean_time:.1f}Hz")
EOF
        
        python temp_speed_test.py || {
            print_warning "Speed test failed for $dataset, continuing..."
            continue
        }
        
        rm -f temp_speed_test.py
        print_success "Completed $dataset"
    done
    
    print_success "Speed comparison completed!"
}

# Show usage information
show_usage() {
    echo -e "\n${BLUE}Usage:${NC}"
    echo "./run_lerobot_examples.sh [command] [args...]"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo "  setup                          - Check dependencies and install package"
    echo "  basic                          - Run basic integration example"
    echo "  compare [dataset] [epochs]     - Run training comparison (default: lerobot/pusht, 5 epochs)"
    echo "  wandb [dataset]                - Run with W&B logging (default: lerobot/pusht)"
    echo "  sweep                          - Quick test on multiple datasets"
    echo "  analyze                        - Analyze existing results"
    echo "  all                            - Run setup + basic + compare"
    echo "  speed [device]                 - Run speed benchmark (default: cpu)"
    echo "  speed_comparison               - Run speed comparison on multiple datasets"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  ./run_lerobot_examples.sh setup"
    echo "  ./run_lerobot_examples.sh basic"
    echo "  ./run_lerobot_examples.sh compare lerobot/aloha_static_coffee 10"
    echo "  ./run_lerobot_examples.sh wandb lerobot/pusht"
    echo "  ./run_lerobot_examples.sh all"
    echo "  ./run_lerobot_examples.sh speed cpu"
    echo "  ./run_lerobot_examples.sh speed_comparison"
}

# Main command handling
case ${1:-help} in
    "setup")
        check_dependencies
        install_package
        ;;
    "basic")
        check_dependencies
        run_basic_example
        ;;
    "compare")
        check_dependencies
        run_training_comparison "$2" "$3"
        ;;
    "wandb")
        check_dependencies
        run_with_wandb "$2"
        ;;
    "sweep")
        check_dependencies
        run_dataset_sweep
        ;;
    "analyze")
        analyze_results
        ;;
    "all")
        check_dependencies
        install_package
        run_basic_example
        run_training_comparison "lerobot/pusht" 5
        analyze_results
        ;;
    "speed")
        check_dependencies
        run_speed_benchmark "$2"
        ;;
    "speed_comparison")
        check_dependencies
        run_speed_comparison
        ;;
    "help"|*)
        show_usage
        ;;
esac

echo -e "\n${GREEN}🎉 Done! Check the outputs/ directory for results.${NC}"
echo -e "${BLUE}📚 See README_LEROBOT_INTEGRATION.md for detailed documentation.${NC}" 