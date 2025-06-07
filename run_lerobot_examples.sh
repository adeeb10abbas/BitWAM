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
    pip install --user -e . || {
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
        pip install --user wandb
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
    local device=${1:-"auto"}
    
    # Auto-detect device if not specified
    if [ "$device" = "auto" ]; then
        gpu_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
        if [ "$gpu_available" = "True" ]; then
            device="cuda"
            print_success "Auto-detected CUDA GPU, using GPU for benchmark"
        else
            device="cpu"
            print_warning "No CUDA GPU detected, using CPU for benchmark"
        fi
    fi
    
    print_header "Running Speed Benchmark on $device"
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
    
    # Check if CUDA is available
    gpu_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    
    if [ "$gpu_available" = "True" ]; then
        print_success "CUDA GPU detected! Will test both CPU and GPU performance"
        devices=("cpu" "cuda")
    else
        print_warning "No CUDA GPU detected, testing CPU only"
        devices=("cpu")
    fi
    
    datasets=("lerobot/pusht")
    
    for dataset in "${datasets[@]}"; do
        echo -e "\n${YELLOW}Speed testing $dataset...${NC}"
        
        for device in "${devices[@]}"; do
            echo -e "${BLUE}Testing on $device...${NC}"
            
            # Create a temporary script to test speed with dataset
            cat > temp_speed_test.py << EOF
import sys
sys.path.append('src')
sys.path.append('../lerobot')

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from bit_vla import BitACTPolicy, BitACTConfig
import torch
import time
import gc

# Force cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Dataset: $dataset")
print(f"Device: $device")

# Load dataset to get dimensions
try:
    dataset = LeRobotDataset("$dataset", video_backend="pyav")
    sample = dataset[0]
    obs_dim = sample["observation.state"].flatten().shape[0]
    action_dim = sample["action"].shape[-1] if sample["action"].dim() > 0 else 2
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Test both BitNet and standard models
configs = [
    ("BitNet", BitACTConfig(action_dim=action_dim, use_bitnet=True)),
    ("Standard", BitACTConfig(action_dim=action_dim, use_bitnet=False))
]

device_torch = torch.device("$device")
print(f"PyTorch device: {device_torch}")

for model_name, config in configs:
    print(f"\n--- {model_name} Model ---")
    
    try:
        model = BitACTPolicy(config, observation_dim=obs_dim).to(device_torch)
        model.eval()
        
        # Model size info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Memory usage (for GPU)
        if device_torch.type == "cuda":
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        
        # Create test inputs
        batch_sizes = [1, 8, 32]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, obs_dim, device=device_torch)
            
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(dummy_input)
                
                if device_torch.type == "cuda":
                    torch.cuda.synchronize()
                
                # Time it
                times = []
                for _ in range(100):
                    if device_torch.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = model(dummy_input)
                    if device_torch.type == "cuda":
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
            
            mean_time = sum(times) / len(times)
            std_time = (sum((t - mean_time)**2 for t in times) / len(times))**0.5
            throughput = batch_size * 1000 / mean_time
            
            print(f"Batch size {batch_size}: {mean_time:.2f}±{std_time:.2f}ms, {throughput:.1f} samples/s")
        
        # Memory usage (for GPU)
        if device_torch.type == "cuda":
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2  # MB
            print(f"GPU memory used: {memory_used:.1f} MB")
        
        # Clean up
        del model
        if device_torch.type == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        continue

print(f"\n✅ Speed test completed for $dataset on $device")
EOF
            
            python temp_speed_test.py || {
                print_warning "Speed test failed for $dataset on $device, continuing..."
                continue
            }
            
            echo ""  # Add spacing between devices
        done
        
        rm -f temp_speed_test.py
        print_success "Completed $dataset"
    done
    
    # Summary message
    if [ "$gpu_available" = "True" ]; then
        print_success "Speed comparison completed on CPU and GPU!"
        echo -e "${BLUE}💡 GPU should show significant speedup for larger batch sizes${NC}"
    else
        print_success "Speed comparison completed on CPU!"
        echo -e "${YELLOW}💡 Install CUDA PyTorch for GPU acceleration: pip install torch --index-url https://download.pytorch.org/whl/cu121${NC}"
    fi
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
    echo "  speed [device]                 - Run speed benchmark (default: auto)"
    echo "  speed_comparison               - Run speed comparison on multiple datasets"
    echo "  cuda_optimize                  - Test CUDA optimizations for BitNet"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  ./run_lerobot_examples.sh setup"
    echo "  ./run_lerobot_examples.sh basic"
    echo "  ./run_lerobot_examples.sh compare lerobot/aloha_static_coffee 10"
    echo "  ./run_lerobot_examples.sh wandb lerobot/pusht"
    echo "  ./run_lerobot_examples.sh all"
    echo "  ./run_lerobot_examples.sh speed                    # Auto-detect GPU/CPU"
    echo "  ./run_lerobot_examples.sh speed cuda               # Force GPU"
    echo "  ./run_lerobot_examples.sh speed cpu                # Force CPU"
    echo "  ./run_lerobot_examples.sh speed_comparison         # Test both CPU and GPU if available"
    echo "  ./run_lerobot_examples.sh cuda_optimize            # Test CUDA optimizations for BitNet"
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
    "cuda_optimize")
        check_dependencies
        print_header "Testing CUDA Optimizations for BitNet"
        
        # Check if CUDA is available
        gpu_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
        
        if [ "$gpu_available" = "True" ]; then
            print_success "CUDA GPU detected! Running optimization tests..."
            python test_cuda_optimizations.py || {
                print_error "CUDA optimization tests failed"
                exit 1
            }
            print_success "CUDA optimization tests completed!"
        else
            print_error "No CUDA GPU detected. These tests require a GPU."
            print_warning "Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121"
            exit 1
        fi
        ;;
    "help"|*)
        show_usage
        ;;
esac

echo -e "\n${GREEN}🎉 Done! Check the outputs/ directory for results.${NC}"
echo -e "${BLUE}📚 See README_LEROBOT_INTEGRATION.md for detailed documentation.${NC}" 