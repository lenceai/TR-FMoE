import argparse
from deployment import (
    setup_environment,
    download_sample_data,
    run_single_gpu_training,
    setup_multi_machine
)
from monitoring import validate_setup

def main():
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
        import subprocess
        subprocess.run(["python", "tr_fmoe_mvp.py", "--mode", "distributed"])
    elif args.command == "train-multi-machine":
        setup_multi_machine()

if __name__ == "__main__":
    main() 