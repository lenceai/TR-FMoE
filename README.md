# TR-FMoE: Tri-Redundant Federated Mixture of Experts

A complete implementation of Tri-Redundant Federated Mixture of Experts (TR-FMoE) for large-scale language model training with efficient expert routing and distributed training capabilities.

## 🚀 Features

- **Advanced Architecture**: Implements state-of-the-art MoE with RoPE attention, RMSNorm, and SwiGLU experts
- **Flexible Training**: Supports both single GPU and multi-GPU distributed training
- **Multi-Source Data**: Processes PDFs and FineWeb dataset for diverse training data
- **Expert Routing**: Intelligent top-k expert selection with load balancing
- **Performance Monitoring**: Built-in metrics tracking and wandb integration
- **Robust Checkpointing**: Automatic model saving and resuming capabilities

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM for single GPU, multiple GPUs for distributed training)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for datasets and checkpoints

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd TR-FMoE
```

### 2. Create Virtual Environment
```bash
python -m venv tr_fmoe_env
source tr_fmoe_env/bin/activate  # On Windows: tr_fmoe_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup

Create a `.env` file in the project root:
```bash
# Required: Hugging Face authentication token
HUGGING_FACE_HUB_TOKEN=your_token_here

# Optional: Weights & Biases API key for experiment tracking
WANDB_API_KEY=your_wandb_key_here
```

**Getting your Hugging Face Token:**
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token to your `.env` file

## 🗂️ Project Structure

```
TR-FMoE/
├── tr_fmoe_mvp.py          # Main implementation with model, training, and data processing
├── requirements.txt        # Python dependencies
├── config.yaml            # Configuration settings
├── main.py                # Entry point script
├── federated.py           # Federated learning components
├── distributed_setup.py   # Distributed training utilities
├── deployment.py          # Model deployment scripts
├── monitoring.py          # Performance monitoring tools
├── .env                   # Environment variables (create this)
├── pdfs/                  # Optional: Place PDF files here for training
├── checkpoints/           # Model checkpoints (auto-created)
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: Single GPU Training (Recommended for getting started)

```bash
python tr_fmoe_mvp.py --mode single
```

### Option 2: Distributed Multi-GPU Training

```bash
python tr_fmoe_mvp.py --mode distributed
```

### Option 3: Custom PDF Training Data

```bash
# Place your PDF files in the pdfs/ directory
mkdir pdfs
# Copy your PDF files to pdfs/

# Run training with PDFs
python tr_fmoe_mvp.py --mode single --pdf_dir ./pdfs
```

## ⚙️ Configuration

The training configuration can be modified in the `train_single_gpu()` or `train_distributed()` functions in `tr_fmoe_mvp.py`:

```python
config = {
    "vocab_size": 32000,        # Vocabulary size (auto-detected from tokenizer)
    "dim": 768,                 # Model dimension
    "num_layers": 12,           # Number of transformer layers
    "num_heads": 12,            # Number of attention heads
    "num_experts": 8,           # Number of experts in MoE layers
    "max_seq_len": 1024,        # Maximum sequence length
    "batch_size": 8,            # Batch size (reduce if OOM)
    "learning_rate": 1e-4,      # Learning rate
    "num_epochs": 3,            # Number of training epochs
    "fineweb_samples": 5000,    # Number of FineWeb samples to use
}
```

### Key Parameters to Adjust:

- **`batch_size`**: Reduce if you encounter out-of-memory errors
- **`num_experts`**: More experts = better specialization but higher memory usage
- **`dim`**: Model size (512, 768, 1024, etc.)
- **`fineweb_samples`**: More samples = longer training but better performance

## 📊 Training Workflow

### 1. Data Preparation
The system automatically:
- Downloads FineWeb dataset samples from Hugging Face
- Processes any PDFs in the `pdfs/` directory
- Tokenizes all text using DialoGPT or GPT-2 tokenizer
- Creates training sequences of specified length

### 2. Model Initialization
- Initializes TR-FMoE model with specified architecture
- Sets up expert networks with SwiGLU activation
- Configures RoPE attention and RMSNorm layers

### 3. Training Process
- **Single GPU**: Direct training on available GPU
- **Distributed**: Automatic multi-GPU coordination using DDP
- Implements gradient clipping and learning rate scheduling
- Saves checkpoints every 1000 steps
- Evaluates model every 500 steps

### 4. Monitoring
- Real-time loss tracking via progress bars
- Weights & Biases integration for experiment tracking
- Expert utilization statistics
- Performance metrics logging

## 📈 Monitoring Training

### Terminal Output
Monitor training progress through:
- Loss values (vocabulary loss + auxiliary loss)
- Learning rate changes
- Evaluation metrics
- Expert usage statistics

### Weights & Biases (Optional)
If you have wandb configured:
1. Training metrics are automatically logged
2. View real-time charts at [wandb.ai](https://wandb.ai)
3. Compare different training runs

### Checkpoints
- Saved automatically in `./checkpoints/` directory
- Best model saved when validation loss improves
- Resume training from checkpoints if interrupted

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
"batch_size": 4  # or even 2
```

**2. Hugging Face Authentication Error**
```bash
# Verify your token is set correctly
echo $HUGGING_FACE_HUB_TOKEN
# Or check your .env file
```

**3. Dataset Download Issues**
- Check internet connection
- Verify Hugging Face token permissions
- Try reducing `fineweb_samples` for testing

**4. PDF Processing Errors**
- Ensure PDFs are readable (not image-only)
- Check PDF file permissions
- Verify PyPDF2 compatibility

### Performance Optimization

**Single GPU:**
- Use mixed precision training (add to trainer)
- Reduce sequence length if memory constrained
- Use gradient checkpointing for larger models

**Distributed Training:**
- Ensure all GPUs have similar memory
- Use appropriate batch size per GPU
- Monitor GPU utilization across all devices

## 🎯 Advanced Usage

### Custom Expert Specialization
Modify expert initialization in `Expert` class to create specialized experts for different domains.

### Custom Data Sources
Extend `DatasetBuilder` class to add new data sources beyond PDFs and FineWeb.

### Model Architecture Changes
Modify `TRFMoEModel` to experiment with different architectures, expert counts, or attention mechanisms.

## 📄 Model Architecture Details

- **Transformer Blocks**: RMSNorm + Multi-head attention + MoE feed-forward
- **Attention**: RoPE (Rotary Position Embedding) for better position encoding
- **Experts**: SwiGLU activation for improved performance
- **Routing**: Top-k expert selection with load balancing loss
- **Normalization**: RMSNorm for stable training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review closed issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy Training! 🚀**