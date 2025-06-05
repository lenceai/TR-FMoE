import time
import torch
from typing import Dict
import os

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