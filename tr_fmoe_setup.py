# requirements.txt
"""
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
PyPDF2>=3.0.1
wandb>=0.15.0
tqdm>=4.65.0
numpy>=1.24.0
aiohttp>=3.8.0
torch-audio>=2.0.0
accelerate>=0.20.0
"""

# config.yaml
"""
# TR-FMoE Configuration
model:
  vocab_size: 32000
  dim: 768
  num_layers: 12
  num_heads: 12
  num_experts: 8
  max_seq_len: 1024
  hidden_dim: 3072

training:
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0
  num_epochs: 5
  eval_steps: 500
  save_steps: 1000

data:
  fineweb_samples: 10000
  pdf_dir: "./pdfs"
  max_seq_len: 1024
  train_test_split: 0.9

distributed:
  backend: "nccl"
  master_addr: "localhost"
  master_port: "12355"
  world_size: 3
  
logging:
  wandb_project: "tr-fmoe-mvp"
  checkpoint_dir: "./checkpoints"
  log_level: "INFO"
"""

# ============================================================================
# MULTI-MACHINE DISTRIBUTED SETUP
# ============================================================================

import os
import socket
import yaml
from typing import Dict, List
import subprocess
import time
import threading

class MultiMachineSetup:
    """Setup for training across multiple machines"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.machines = [
            {"host": "192.168.1.10", "port": 22, "gpu_count": 1, "rank_start": 0},
            {"host": "192.168.1.11", "port": 22, "gpu_count": 1, "rank_start": 1}, 
            {"host": "192.168.1.12", "port": 22, "gpu_count": 1, "rank_start": 2}
        ]
        
        self.master_machine = self.machines[0]
        self.world_size = sum(m["gpu_count"] for m in self.machines)

    def setup_ssh_keys(self):
        """Setup SSH keys for passwordless access"""
        for machine in self.machines[1:]:  # Skip master
            print(f"Setting up SSH for {machine['host']}")
            # Generate SSH key if not exists
            subprocess.run([
                "ssh-keygen", "-t", "rsa", "-N", "", 
                "-f", f"~/.ssh/id_rsa_{machine['host']}", "-q"
            ], check=False)
            
            # Copy public key to remote machine
            subprocess.run([
                "ssh-copy-id", "-i", f"~/.ssh/id_rsa_{machine['host']}.pub",
                f"user@{machine['host']}"
            ], check=False)

    def sync_code_to_machines(self):
        """Sync code to all machines"""
        for machine in self.machines:
            if machine == self.master_machine:
                continue
                
            print(f"Syncing code to {machine['host']}")
            subprocess.run([
                "rsync", "-avz", "--exclude=checkpoints", "--exclude=__pycache__",
                "./", f"user@{machine['host']}:~/tr-fmoe/"
            ], check=True)

    def launch_training_on_machine(self, machine: Dict, master_addr: str, master_port: str):
        """Launch training process on a specific machine"""
        host = machine["host"]
        rank_start = machine["rank_start"]
        gpu_count = machine["gpu_count"]
        
        if machine == self.master_machine:
            # Run locally on master
            cmd = [
                "python", "-m", "torch.distributed.launch",
                f"--nproc_per_node={gpu_count}",
                f"--nnodes={len(self.machines)}",
                f"--node_rank={self.machines.index(machine)}",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                "tr_fmoe_distributed.py"
            ]
            subprocess.run(cmd, check=True)
        else:
            # Run remotely via SSH
            remote_cmd = [
                "ssh", f"user@{host}",
                f"cd ~/tr-fmoe && python -m torch.distributed.launch "
                f"--nproc_per_node={gpu_count} "
                f"--nnodes={len(self.machines)} "
                f"--node_rank={self.machines.index(machine)} "
                f"--master_addr={master_addr} "
                f"--master_port={master_port} "
                f"tr_fmoe_distributed.py"
            ]
            subprocess.run(remote_cmd, check=True)

    def launch_distributed_training(self):
        """Launch distributed training across all machines"""
        master_addr = self.master_machine["host"]
        master_port = str(self.config["distributed"]["master_port"])
        
        print(f"Starting distributed training across {len(self.machines)} machines")
        print(f"Master: {master_addr}:{master_port}")
        print(f"World size: {self.world_size}")
        
        # Sync code to all machines
        self.sync_code_to_machines()
        
        # Launch training processes
        threads = []
        for machine in self.machines:
            thread = threading.Thread(
                target=self.launch_training_on_machine,
                args=(machine, master_addr, master_port)
            )
            threads.append(thread)
            thread.start()
            time.sleep(5)  # Stagger launches
        
        # Wait for all processes to complete
        for thread in threads:
            thread.join()
        
        print("Distributed training completed!")

# ============================================================================
# FEDERATED SIMULATION COMPONENTS
# ============================================================================

class FederatedNode:
    """Simulates a federated learning node with experts"""
    
    def __init__(self, node_id: str, expert_ids: List[int], host: str, port: int):
        self.node_id = node_id
        self.expert_ids = expert_ids
        self.host = host
        self.port = port
        self.experts = {}
        self.performance_metrics = {
            "latency": 0.0,
            "reliability": 1.0,
            "load": 0.0
        }

    def load_experts(self, model_checkpoint: str):
        """Load expert networks from checkpoint"""
        checkpoint = torch.load(model_checkpoint, map_location="cpu")
        model_state = checkpoint["model_state_dict"]
        
        for expert_id in self.expert_ids:
            expert_state = {}
            for key, value in model_state.items():
                if f"experts.{expert_id}" in key:
                    new_key = key.replace(f"layers.0.moe.experts.{expert_id}.", "")
                    expert_state[new_key] = value
            
            # Create expert instance
            expert = Expert(dim=768, hidden_dim=3072, expert_id=expert_id)
            expert.load_state_dict(expert_state)
            self.experts[expert_id] = expert

    async def process_expert_request(self, expert_id: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Process expert inference request"""
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not available on node {self.node_id}")
        
        start_time = time.time()
        
        with torch.no_grad():
            output = self.experts[expert_id](input_tensor)
        
        # Update performance metrics
        latency = time.time() - start_time
        self.performance_metrics["latency"] = 0.9 * self.performance_metrics["latency"] + 0.1 * latency
        
        return output

class FederatedCoordinator:
    """Coordinates federated training and inference"""
    
    def __init__(self):
        self.nodes = {}
        self.expert_placement = {}  # expert_id -> [node_ids with replicas]
        self.routing_cache = {}

    def register_node(self, node: FederatedNode):
        """Register a federated node"""
        self.nodes[node.node_id] = node
        
        # Update expert placement
        for expert_id in node.expert_ids:
            if expert_id not in self.expert_placement:
                self.expert_placement[expert_id] = []
            self.expert_placement[expert_id].append(node.node_id)

    def place_experts_tri_redundant(self, num_experts: int):
        """Place experts with tri-redundancy"""
        node_list = list(self.nodes.keys())
        
        if len(node_list) < 3:
            raise ValueError("Need at least 3 nodes for tri-redundancy")
        
        for expert_id in range(num_experts):
            # Select 3 nodes for this expert using round-robin with offset
            selected_nodes = []
            for i in range(3):
                node_idx = (expert_id * 3 + i) % len(node_list)
                selected_nodes.append(node_list[node_idx])
            
            self.expert_placement[expert_id] = selected_nodes
            
            # Update node expert assignments
            for node_id in selected_nodes:
                if expert_id not in self.nodes[node_id].expert_ids:
                    self.nodes[node_id].expert_ids.append(expert_id)

    async def route_expert_request(self, expert_id: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Route expert request with tri-redundancy"""
        if expert_id not in self.expert_placement:
            raise ValueError(f"Expert {expert_id} not placed in federation")
        
        available_nodes = self.expert_placement[expert_id]
        
        # Select best 2 nodes based on performance
        node_scores = []
        for node_id in available_nodes:
            node = self.nodes[node_id]
            score = (node.performance_metrics["reliability"] / 
                    (1.0 + node.performance_metrics["latency"]) /
                    (1.0 + node.performance_metrics["load"]))
            node_scores.append((score, node_id))
        
        # Sort by score and select top 2
        node_scores.sort(reverse=True)
        selected_nodes = [node_id for _, node_id in node_scores[:2]]
        
        # Request inference from selected nodes
        tasks = []
        for node_id in selected_nodes:
            node = self.nodes[node_id]
            task = node.process_expert_request(expert_id, input_tensor)
            tasks.append(task)
        
        # Wait for responses
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Consensus mechanism (simplified)
        valid_outputs = [out for out in outputs if isinstance(out, torch.Tensor)]
        
        if len(valid_outputs) == 0:
            raise RuntimeError("No valid responses from expert replicas")
        elif len(valid_outputs) == 1:
            return valid_outputs[0]
        else:
            # Average the outputs (Byzantine fault tolerance would be more complex)
            return torch.stack(valid_outputs).mean(dim=0)

# ============================================================================
# DEPLOYMENT SCRIPTS
# ============================================================================

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
    import requests
    
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

# ============================================================================
# MONITORING AND UTILITIES
# ============================================================================

class TrainingMonitor:
    """Monitor training progress and system resources"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []

    def log_metrics(self, metrics: Dict):
        """Log training metrics"""
        metrics["timestamp"] = time.time() - self.start_time
        self.metrics_history.append(metrics)
        
        # Print progress
        print(f"Step {metrics.get('step', 0):>6} | "
              f"Loss: {metrics.get('loss', 0):.4f} | "
              f"LR: {metrics.get('lr', 0):.2e} | "
              f"GPU Mem: {self.get_gpu_memory():.1f}GB")

    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def get_training_summary(self) -> Dict:
        """Get training summary statistics"""
        if not self.metrics_history:
            return {}
        
        losses = [m.get("loss", 0) for m in self.metrics_history]
        
        return {
            "total_steps": len(self.metrics_history),
            "training_time": time.time() - self.start_time,
            "final_loss": losses[-1] if losses else 0,
            "best_loss": min(losses) if losses else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0
        }

def validate_setup():
    """Validate training setup"""
    print("Validating TR-FMoE setup...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will be slow")
    else:
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1e9:.0f}GB)")
    
    # Check required packages
    required_packages = ["torch", "transformers", "datasets", "PyPDF2", "wandb"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} missing - install with: pip install {package}")
    
    # Check directories
    required_dirs = ["checkpoints", "logs", "pdfs"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing - creating...")
            os.makedirs(dir_name, exist_ok=True)
    
    print("Setup validation complete!")

# ============================================================================
# MAIN CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TR-FMoE Setup and Training")
    parser.add_argument("command", choices=[
        "setup", "validate", "download-data", "train-single", 
        "train-distributed", "train-multi-machine"
    ], help="Command to execute")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_environment()
    elif args.command == "validate":
        validate_setup()
    elif args.command == "download-data":
        download_sample_data()
    elif args.command == "train-single":
        run_single_gpu_training()
    elif args.command == "train-distributed":
        subprocess.run(["python", "tr_fmoe_mvp.py", "--mode", "distributed"])
    elif args.command == "train-multi-machine":
        setup_multi_machine()
