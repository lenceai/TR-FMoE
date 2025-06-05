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
            {"host": "192.168.77.53", "port": 22, "gpu_count": 1, "rank_start": 0},
            {"host": "192.168.77.54", "port": 22, "gpu_count": 1, "rank_start": 1}, 
            {"host": "192.168.77.55", "port": 22, "gpu_count": 1, "rank_start": 2}
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