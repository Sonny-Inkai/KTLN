"""
Configurator for ViLegalJERE Training
Điều chỉnh tham số training tại đây
"""

# Override training parameters if needed
import os

# Check if we're in Kaggle environment
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("🏃‍♂️ Running in Kaggle environment - adjusting parameters...")
    
    # Kaggle-specific optimizations
    batch_size = 4  # Smaller batch size for memory
    gradient_accumulation_steps = 8  # Compensate with more accumulation
    max_iters = 8000  # Shorter training for Kaggle time limits
    eval_interval = 200  # More frequent evaluation
    log_interval = 20   # More frequent logging
    
    # Memory optimizations
    dtype = 'float16'  # Use mixed precision
    compile = False    # Disable torch.compile for compatibility
    
    print(f"📊 Kaggle Config: batch_size={batch_size}, grad_accum={gradient_accumulation_steps}")
    print(f"⏱️  Training iters: {max_iters}, eval every {eval_interval} steps")

# Performance tuning based on model size  
if 'n_embd' in globals() and n_embd <= 512:
    # Small model optimizations
    learning_rate = 5e-4  # Higher LR for small models
    warmup_iters = 1000   # Shorter warmup
    weight_decay = 0.01   # Less regularization
    print(f"🤏 Small model detected (n_embd={n_embd}) - using optimized config")

elif 'n_embd' in globals() and n_embd >= 1024:
    # Large model optimizations  
    learning_rate = 1e-4  # Lower LR for large models
    warmup_iters = 4000   # Longer warmup
    weight_decay = 0.1    # More regularization
    print(f"🦣 Large model detected (n_embd={n_embd}) - using stable config")

# Fine-tuning vs Pre-training adjustments
if 'finetune' in globals() and finetune:
    learning_rate = learning_rate * 0.3  # Much lower LR for fine-tuning
    weight_decay = weight_decay * 0.1    # Less weight decay
    eval_interval = min(eval_interval, 100)  # More frequent eval for fine-tuning
    print(f"🎯 Fine-tuning mode: LR={learning_rate:.2e}, eval_interval={eval_interval}")

print("✅ Configurator loaded successfully!")
