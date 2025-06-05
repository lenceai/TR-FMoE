import os
import subprocess
import requests
from distributed_setup import MultiMachineSetup

def setup_environment():
    """Setup training environment"""
    print("Setting up TR-FMoE training environment...")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("pdfs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Install requirements
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    
    print("Environment setup complete!")

def download_sample_data():
    """Download sample PDFs for training"""
    # Sample academic papers (replace with actual URLs)
    sample_pdfs = [
        "https://arxiv.org/pdf/1706.03762.pdf",  # Attention is All You Need
        "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
        "https://arxiv.org/pdf/2101.03961.pdf",  # Switch Transformer
    ]
    
    print("Downloading sample PDFs...")
    for i, url in enumerate(sample_pdfs):
        try:
            response = requests.get(url)
            with open(f"pdfs/sample_{i}.pdf", "wb") as f:
                f.write(response.content)
            print(f"Downloaded sample_{i}.pdf")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def run_single_gpu_training():
    """Run single GPU training with optimal settings for RTX 3090"""
    print("Starting single GPU training optimized for RTX 3090...")
    
    # RTX 3090 optimized config
    config = {
        "batch_size": 12,  # Optimized for 24GB VRAM
        "gradient_accumulation_steps": 4,
        "mixed_precision": True,
        "num_experts": 8,
        "dim": 768,
        "num_layers": 12,
        "max_seq_len": 1024
    }
    
    # Set environment variables for optimization
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # Run training
    subprocess.run([
        "python", "tr_fmoe_mvp.py", 
        "--mode", "single",
        "--pdf_dir", "./pdfs"
    ], check=True)

def setup_multi_machine():
    """Setup for multi-machine distributed training"""
    print("Setting up multi-machine distributed training...")
    
    setup = MultiMachineSetup()
    
    # Setup SSH keys
    setup.setup_ssh_keys()
    
    # Launch distributed training
    setup.launch_distributed_training() 