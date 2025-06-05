# TR-FMoE MVP: Tri-Redundant Federated Mixture of Experts
# Complete implementation for single GPU and distributed training

import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import math
import time
import logging
import pickle
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import PyPDF2
import io
import requests
import wandb
from tqdm import tqdm
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Hugging Face authentication
token = os.getenv('HUGGING_FACE_HUB_TOKEN')
if not token:
    logger.error("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    logger.info("Please set your token using: export HUGGING_FACE_HUB_TOKEN=your_token")
    exit(1)

try:
    login(token=token)
    logger.info("Successfully authenticated with Hugging Face")
except Exception as e:
    logger.error(f"Failed to authenticate with Hugging Face: {e}")
    exit(1)

# ============================================================================
# 1. MODEL ARCHITECTURE
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=x.device)
        freqs = torch.outer(t, self.inv_freq)
        # Shape: [seq_len, dim//2]
        cos = freqs.cos()  # [seq_len, dim//2]
        sin = freqs.sin()  # [seq_len, dim//2]
        return cos, sin

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: [B, num_heads, seq_len, head_dim]
        B, num_heads, seq_len, head_dim = x.shape
        
        # Split into real and imaginary parts
        x1, x2 = x[..., ::2], x[..., 1::2]  # Each: [B, num_heads, seq_len, head_dim//2]
        
        # Reshape cos/sin to match: [1, 1, seq_len, head_dim//2]
        cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim//2]
        sin = sin[None, None, :, :]  # [1, 1, seq_len, head_dim//2]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return rotated

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3, 4)  # [B, 3, L, num_heads, head_dim]
        q, k, v = qkv.unbind(dim=1)  # Each: [B, L, num_heads, head_dim]
        
        # Transpose to get [B, num_heads, L, head_dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn.masked_fill_(mask, -torch.inf)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)

class Expert(nn.Module):
    """Individual Expert Network"""
    def __init__(self, dim: int, hidden_dim: int, expert_id: int, specialization: str = "general"):
        super().__init__()
        self.expert_id = expert_id
        self.specialization = specialization
        self.dim = dim
        
        # SwiGLU architecture
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        # Performance tracking
        self.call_count = 0
        self.total_compute_time = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        start_time = time.time()
        
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        
        self.total_compute_time += time.time() - start_time
        return output

    def get_stats(self) -> Dict:
        avg_time = self.total_compute_time / max(1, self.call_count)
        return {
            "expert_id": self.expert_id,
            "specialization": self.specialization,
            "call_count": self.call_count,
            "avg_compute_time": avg_time
        }

class Router(nn.Module):
    """Router for expert selection"""
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, dim]
        gate_scores = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Top-k selection
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_values, dim=-1)
        
        # Load balancing loss
        expert_usage = F.softmax(gate_scores, dim=-1).mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(expert_usage ** 2)
        
        return top_k_indices, top_k_weights, aux_loss

class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    def __init__(self, dim: int, num_experts: int, hidden_dim: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        
        self.router = Router(dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim, expert_id=i, specialization=f"expert_{i}")
            for i in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        
        # Get routing decisions
        expert_indices, expert_weights, aux_loss = self.router(x)
        
        # Reshape for easier processing
        x_flat = x.view(-1, dim)  # [batch*seq, dim]
        expert_indices_flat = expert_indices.view(-1, self.top_k)
        expert_weights_flat = expert_weights.view(-1, self.top_k)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices_flat == expert_idx)
            if not expert_mask.any():
                continue
                
            # Get token indices and weights for this expert
            token_indices, expert_positions = expert_mask.nonzero(as_tuple=True)
            weights = expert_weights_flat[token_indices, expert_positions]
            
            # Process tokens through expert
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Add weighted contribution to output
            output[token_indices] += weights.unsqueeze(1) * expert_output
        
        return output.view(batch_size, seq_len, dim), aux_loss

class TransformerBlock(nn.Module):
    """Transformer block with MoE"""
    def __init__(self, dim: int, num_heads: int, num_experts: int, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, max_seq_len)
        self.moe = MoELayer(dim, num_experts, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # MoE with residual
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, aux_loss

class TRFMoEModel(nn.Module):
    """Complete TR-FMoE Model"""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 8,
        max_seq_len: int = 2048,
        hidden_dim: int = None
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, num_experts, hidden_dim, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.embedding(tokens)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
        
        x = self.norm(x)
        logits = self.output(x)
        
        outputs = {"logits": logits, "aux_loss": total_aux_loss}
        
        if targets is not None:
            # Compute loss
            vocab_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
            total_loss = vocab_loss + 0.01 * total_aux_loss
            outputs.update({
                "loss": total_loss,
                "vocab_loss": vocab_loss,
                "aux_loss": total_aux_loss
            })
        
        return outputs

# ============================================================================
# 2. DATA PROCESSING
# ============================================================================

class PDFProcessor:
    """Process PDF files for training"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return ""

    def process_pdf_directory(self, pdf_dir: str, max_files: int = None) -> List[str]:
        """Process all PDFs in directory"""
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        texts = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                texts.append(text)
        
        return texts

class DatasetBuilder:
    """Build training dataset from multiple sources"""
    def __init__(self, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def load_fineweb(self, num_samples: int = 10000) -> List[str]:
        """Load FineWeb dataset from Hugging Face"""
        logger.info(f"Loading FineWeb dataset with {num_samples} samples...")
        
        try:
            # Load FineWeb dataset with authentication
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                "sample-10BT",
                split="train",
                streaming=True,
                token=os.getenv('HUGGING_FACE_HUB_TOKEN')
            )
            
            texts = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(example["text"])
                if i % 1000 == 0:
                    logger.info(f"Loaded {i} FineWeb samples")
            
            return texts
        except Exception as e:
            logger.error(f"Failed to load FineWeb dataset: {e}")
            logger.info("Falling back to sample text data...")
            # Return some sample text data as fallback
            return [
                "This is a sample text for training.",
                "Another sample text for the model.",
                "More sample text data for training."
            ]

    def tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """Tokenize texts and create training sequences"""
        logger.info("Tokenizing texts...")
        
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            # Tokenize text
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Split into chunks of max_seq_len
            for i in range(0, len(tokens) - self.max_seq_len + 1, self.max_seq_len):
                chunk = tokens[i:i + self.max_seq_len]
                if len(chunk) == self.max_seq_len:
                    all_tokens.append(torch.tensor(chunk, dtype=torch.long))
        
        return all_tokens

    def create_dataset(self, pdf_dir: str = None, fineweb_samples: int = 5000) -> torch.utils.data.Dataset:
        """Create complete training dataset"""
        all_texts = []
        
        # Load FineWeb data
        fineweb_texts = self.load_fineweb(fineweb_samples)
        all_texts.extend(fineweb_texts)
        
        # Load PDF data if provided
        if pdf_dir and os.path.exists(pdf_dir):
            pdf_processor = PDFProcessor(self.tokenizer)
            pdf_texts = pdf_processor.process_pdf_directory(pdf_dir)
            all_texts.extend(pdf_texts)
            logger.info(f"Loaded {len(pdf_texts)} PDF documents")
        
        # Tokenize all texts
        tokenized_sequences = self.tokenize_texts(all_texts)
        
        logger.info(f"Created dataset with {len(tokenized_sequences)} sequences")
        return TokenDataset(tokenized_sequences)

class TokenDataset(torch.utils.data.Dataset):
    """Dataset for tokenized sequences"""
    def __init__(self, sequences: List[torch.Tensor]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:]
        }

# ============================================================================
# 3. TRAINING INFRASTRUCTURE
# ============================================================================

class TRFMoETrainer:
    """Training infrastructure for TR-FMoE"""
    def __init__(
        self,
        model: TRFMoEModel,
        tokenizer,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Optimizer with different learning rates for router vs experts
        self.optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if "gate" in n], "lr": learning_rate * 0.1},
            {"params": [p for n, p in model.named_parameters() if "gate" not in n], "lr": learning_rate}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=warmup_steps * 10,  # Will be updated based on actual training steps
            pct_start=0.1
        )
        
        # Metrics tracking
        self.step = 0
        self.best_loss = float('inf')

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, labels)
        loss = outputs["loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        
        return {
            "loss": loss.item(),
            "vocab_loss": outputs["vocab_loss"].item(),
            "aux_loss": outputs["aux_loss"].item(),
            "lr": self.scheduler.get_last_lr()[0]
        }

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_vocab_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, labels)
                
                total_loss += outputs["loss"].item()
                total_vocab_loss += outputs["vocab_loss"].item()
                total_aux_loss += outputs["aux_loss"].item()
                num_batches += 1
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_vocab_loss": total_vocab_loss / num_batches,
            "eval_aux_loss": total_aux_loss / num_batches
        }

    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "best_loss": self.best_loss
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epochs: int,
        save_steps: int = 1000,
        eval_steps: int = 500,
        checkpoint_dir: str = "./checkpoints"
    ):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Initialize wandb if available
        try:
            wandb.init(project="tr-fmoe-mvp", config={
                "num_epochs": num_epochs,
                "model_dim": self.model.dim,
                "num_layers": self.model.num_layers,
                "vocab_size": self.model.vocab_size
            })
        except:
            logger.warning("wandb not available, skipping logging")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}"
                })
                
                # Evaluation
                if self.step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    logger.info(f"Step {self.step} - Eval Loss: {eval_metrics['eval_loss']:.4f}")
                    
                    # Log to wandb
                    try:
                        wandb.log({**metrics, **eval_metrics, "step": self.step})
                    except:
                        pass
                    
                    # Save best model
                    if eval_metrics["eval_loss"] < self.best_loss:
                        self.best_loss = eval_metrics["eval_loss"]
                        self.save_checkpoint(checkpoint_dir, epoch)
                
                # Regular checkpointing
                if self.step % save_steps == 0:
                    self.save_checkpoint(checkpoint_dir, epoch)
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")

# ============================================================================
# 4. SINGLE GPU TRAINING SCRIPT
# ============================================================================

def train_single_gpu():
    """Train TR-FMoE on single GPU"""
    # Configuration
    config = {
        "vocab_size": 32000,
        "dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "num_experts": 8,
        "max_seq_len": 1024,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "fineweb_samples": 5000,
        "pdf_dir": "./pdfs"  # Optional: add PDF files here
    }
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")
    
    # Initialize tokenizer with authentication
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-medium",
            token=os.getenv('HUGGING_FACE_HUB_TOKEN'),
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        config["vocab_size"] = len(tokenizer)
        logger.info("Successfully loaded DialoGPT-medium tokenizer")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.info("Falling back to GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config["vocab_size"] = len(tokenizer)
        logger.info("Successfully loaded GPT-2 tokenizer")

    # Build dataset
    dataset_builder = DatasetBuilder(tokenizer, config["max_seq_len"])
    full_dataset = dataset_builder.create_dataset(
        pdf_dir=config.get("pdf_dir"),
        fineweb_samples=config["fineweb_samples"]
    )
    
    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = TRFMoEModel(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_experts=config["num_experts"],
        max_seq_len=config["max_seq_len"]
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Initialize trainer
    trainer = TRFMoETrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=config["learning_rate"]
    )
    
    # Start training
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=config["num_epochs"],
        checkpoint_dir="./checkpoints"
    )
    
    logger.info("Single GPU training completed!")

# ============================================================================
# 5. DISTRIBUTED TRAINING SETUP
# ============================================================================

def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_distributed_worker(rank: int, world_size: int, config: Dict):
    """Distributed training worker function"""
    setup_distributed(rank, world_size)
    
    # Setup device
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Build dataset (only on rank 0 to avoid conflicts)
    if rank == 0:
        dataset_builder = DatasetBuilder(tokenizer, config["max_seq_len"])
        full_dataset = dataset_builder.create_dataset(
            pdf_dir=config.get("pdf_dir"),
            fineweb_samples=config["fineweb_samples"]
        )
        
        # Save dataset
        torch.save(full_dataset, "distributed_dataset.pt")
    
    # Wait for rank 0 to finish dataset creation
    dist.barrier()
    
    # Load dataset on all ranks
    full_dataset = torch.load("distributed_dataset.pt")
    
    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = TRFMoEModel(
        vocab_size=len(tokenizer),
        dim=config["dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_experts=config["num_experts"],
        max_seq_len=config["max_seq_len"]
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Initialize trainer (modified for distributed)
    class DistributedTrainer(TRFMoETrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rank = rank
            self.world_size = world_size
        
        def save_checkpoint(self, checkpoint_dir: str, epoch: int):
            # Only save on rank 0
            if self.rank == 0:
                super().save_checkpoint(checkpoint_dir, epoch)
        
        def train(self, train_dataloader, eval_dataloader, num_epochs, **kwargs):
            # Set epoch for sampler
            for epoch in range(num_epochs):
                train_dataloader.sampler.set_epoch(epoch)
                super().train(train_dataloader, eval_dataloader, 1, **kwargs)
    
    trainer = DistributedTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=config["learning_rate"]
    )
    
    # Start training
    if rank == 0:
        logger.info(f"Starting distributed training on {world_size} GPUs...")
    
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=config["num_epochs"],
        checkpoint_dir="./distributed_checkpoints"
    )
    
    cleanup_distributed()

def train_distributed():
    """Launch distributed training"""
    config = {
        "dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "num_experts": 8,
        "max_seq_len": 1024,
        "batch_size": 4,  # Smaller batch size per GPU
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "fineweb_samples": 5000,
        "pdf_dir": "./pdfs"
    }
    
    world_size = torch.cuda.device_count()
    logger.info(f"Starting distributed training on {world_size} GPUs")
    
    mp.spawn(
        train_distributed_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TR-FMoE Training")
    parser.add_argument("--mode", choices=["single", "distributed"], default="single",
                       help="Training mode: single GPU or distributed")
    parser.add_argument("--pdf_dir", type=str, default=None,
                       help="Directory containing PDF files for training")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        train_single_gpu()
    elif args.mode == "distributed":
        train_distributed()
