import os
import time
import math
import pickle
from contextlib import nullcontext
import importlib
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
import json
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
finetune = False # True: fine-tuning, False: pre-training
# -----------------------------------------------------------------------------

# --- Cấu hình chung ---
eval_only = False
always_save_checkpoint = True

# Trong Kaggle notebook, thêm vào cell đầu tiên:
import os
os.environ['WANDB_API_KEY'] = 'bcc183326224decc1f9fee116ccfd509e740fab1'

# --- Cấu hình riêng cho từng giai đoạn ---
if finetune:
    # --- CẤU HÌNH CHO FINE-TUNING ---
    init_from = 'resume' # Bắt buộc phải resume từ model đã pre-trained
    data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning" # Nơi chứa file finetune.txt
    finetune_file_name = "finetune.json"
    out_dir = '/kaggle/working/out_vilegal_t5small' # Thư mục chứa checkpoint pre-trained
    
    # ✅ FIXED: Siêu tham số cho fine-tuning tối ưu cho T4x2
    learning_rate = 5e-5 # T5 fine-tuning standard (thấp hơn cho stability)
    max_iters = 6000     # Giảm để fit trong Kaggle time limit
    batch_size = 32      # ✅ FIXED: Giảm cho T4 memory (16GB VRAM)
    gradient_accumulation_steps = 2  # ✅ FIXED: Tăng để maintain effective batch size
    weight_decay = 0.001  # ✅ FIXED: Standard weight decay cho T5
    eval_interval = 200  # ✅ FIXED: Tăng để save time
    log_interval = 10    # ✅ FIXED: Reduce logging frequency
    eval_iters = 100     # ✅ FIXED: Giảm để save time
    
else:
    # --- CẤU HÌNH CHO PRE-TRAINING ---
    init_from = 'scratch' # 'scratch' or 'resume'
    data_path = "/kaggle/input/vietnamese-legal-dataset"  # Kaggle dataset path
    out_dir = '/kaggle/working/out_vilegal_t5small'
    
    # ✅ FIXED: Siêu tham số cho pre-training tối ưu cho T4x2
    learning_rate = 3e-4  # ✅ FIXED: Standard for T5-small pre-training
    max_iters = 10000     # ✅ FIXED: Reasonable for T5-small
    batch_size = 32       # ✅ FIXED: Safe for T4 memory
    gradient_accumulation_steps = 4   # ✅ FIXED: Maintain large effective batch
    weight_decay = 1e-2   # ✅ FIXED: Standard T5 weight decay
    eval_interval = 500  # ✅ FIXED: Less frequent for pre-training
    log_interval = 10     # ✅ FIXED: Reduce logging overhead
    eval_iters = 200      # ✅ FIXED: Keep reasonable for evaluation
    
# wandb logging
wandb_log = True    # Enable for better tracking
wandb_project = 'ViLegalJERE-T5Small'
wandb_run_name = 'vilegal_t5small_kaggle'
# data
dataset = 'vietnamese_legal'
block_size = 512    # Keep same
max_source_length = 512  # encoder max length
max_target_length = 512  # decoder max length
# model - T5-small architecture (~60M parameters)
n_layer = 6         # T5-small has 6 layers each for encoder/decoder
n_head = 8          # T5-small uses 8 attention heads
head_dim = 64       # 512/8 = 64
rank = 4            # ✅ GOOD: Reasonable CP rank for T6
q_rank = 8          # ✅ GOOD: Reasonable query rank for T6
n_embd = 512        # T5-small hidden size
dropout = 0.1       # Standard dropout for T5
bias = False
using_groupnorm = True
# optimizer
optimizer_name = 'adamw'
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000   # ✅ FIXED: Longer warmup for stability (was 1000)
lr_decay_iters = 10000  # ✅ FIXED: Match max_iters for full decay (was 10000)
min_lr = 5e-6        # ✅ FIXED: Higher min_lr to avoid vanishing gradients (was 1e-6)
# DDP settings for Kaggle T4x2
backend = 'gloo'  # Use gloo instead of nccl for better Kaggle compatibility
schedule = 'cosine'
model_type = 'ViLegalJERE'
# system
device = 'cuda'  
dtype = 'float16'   
compile = False     # Disable compile for Kaggle compatibility
scale_attn_by_inverse_layer_idx = False
# -----------------------------------------------------------------------------

# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
# config = {k: globals()[k] for k in config_keys}

# Import ViLegalJERE model
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')

def get_num_params(model, non_embedding=False):
    """Return the number of parameters in the model."""
    n_params = sum(p.numel() for p in model.parameters())
    if non_embedding and hasattr(model, 'shared'):
        n_params -= model.shared.weight.numel()
    return n_params

# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    world_size = 1
    # Keep gradient_accumulation_steps as is for single GPU

# Calculate total tokens
tokens_per_iter = batch_size * (max_source_length + max_target_length) * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)
tokens_trained = 0

# Initialize random seed
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ✅ FIXED: Memory optimization for T4 GPU
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear cache before training
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Data loading for Vietnamese legal text
def load_legal_data():
    """Tải dữ liệu cho pre-training"""
    data_file = os.path.join(data_path, 'dataset.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    articles = text.split('\n')
    tokenized_data = [tokenizer.encode(art, truncation=True, max_length=block_size) for art in articles]
    return tokenized_data

def load_finetune_data():
    """Tải và xử lý dữ liệu từ file finetune.json (JSON) cho fine-tuning"""
    data_file = os.path.join(data_path, finetune_file_name)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Finetune dataset not found at {data_file}")
    
    processed_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, value in data.items():
        source_text = value.get("formatted_context_sent", "")
        target_text = value.get("extracted_relations_text", "")
        
        # Chỉ lấy các cặp dữ liệu có cả input và output
        if source_text and target_text:
            processed_data.append((source_text, target_text))
    
    if master_process:
        print(f"Loaded {len(processed_data)} fine-tuning pairs")
    return processed_data

# ✅ Helper functions cho T5 span corruption theo chuẩn Google
def random_spans_helper(inputs_length, noise_density, mean_noise_span_length, 
                       extra_tokens_per_span_inputs=1, extra_tokens_per_span_targets=1):
    """Calculate input and target lengths for span corruption"""
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), tokens_length - 1)
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        
        # Calculate input length (original tokens - noise + sentinel tokens)
        inputs_length = num_nonnoise_tokens + num_noise_spans * extra_tokens_per_span_inputs
        
        # Calculate target length (noise tokens + sentinel tokens)  
        targets_length = num_noise_tokens + num_noise_spans * extra_tokens_per_span_targets
        
        return inputs_length, targets_length
    
    return _tokens_length_to_inputs_length_targets_length(inputs_length)

def create_noise_mask(length, noise_density, mean_noise_span_length):
    """Create random spans noise mask like Google T5"""
    if noise_density == 0.0:
        return [False] * length
    
    # Increase length to avoid degeneracy    
    length = max(length, 2)
    
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = max(1, int(round(num_noise_tokens / mean_noise_span_length)))
    num_nonnoise_tokens = length - num_noise_tokens
    
    def random_segmentation(num_items, num_segments):
        """Partition a sequence randomly into non-empty segments"""
        if num_segments >= num_items:
            return [1] * num_items
        
        # Create random breakpoints
        breaks = sorted(np.random.choice(num_items - 1, num_segments - 1, replace=False))
        breaks = [0] + [b + 1 for b in breaks] + [num_items]
        
        # Calculate segment lengths
        lengths = [breaks[i+1] - breaks[i] for i in range(len(breaks) - 1)]
        return lengths
    
    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)
    
    # Interleave spans starting with non-noise
    interleaved_span_lengths = []
    for i in range(num_noise_spans):
        interleaved_span_lengths.append(nonnoise_span_lengths[i])
        interleaved_span_lengths.append(noise_span_lengths[i])
    
    # Create mask
    mask = []
    is_noise = False
    for span_length in interleaved_span_lengths:
        mask.extend([is_noise] * span_length)
        is_noise = not is_noise
    
    return mask[:length]

def create_t5_spans(tokens, noise_density=0.15, mean_noise_span_length=3.0):
    """
    Tạo dữ liệu theo kiểu span corruption của T5 chuẩn Google.
    """
    import numpy as np
    
    num_tokens = len(tokens)
    if num_tokens <= 1:
        return tokens, tokens
        
    # ✅ Tạo noise mask theo chuẩn Google T5
    noise_mask = create_noise_mask(num_tokens, noise_density, mean_noise_span_length)
    
    # ✅ Lấy sentinel token theo chuẩn Google T5
    try:
        sentinel_start_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        if sentinel_start_id == tokenizer.unk_token_id:
            raise ValueError("Sentinel token not found")
    except (KeyError, ValueError):
        # ✅ CORRECT FALLBACK: Hard-coded based on tokenizer config
        sentinel_start_id = 10099  # <extra_id_0> confirmed by tokenizer analysis
        if master_process:
            print(f"Warning: Using hard-coded sentinel_start_id = {sentinel_start_id}")
    
    # ✅ Create input sequence with sentinels (noise_span_to_unique_sentinel)
    input_ids = []
    labels = []
    
    prev_token_is_noise = False
    sentinel_idx = 0
    
    for i, token in enumerate(tokens):
        is_noise = noise_mask[i] if i < len(noise_mask) else False
        
        if is_noise:
            if not prev_token_is_noise:
                # First token of a noise span - add sentinel to input
                sentinel_id = sentinel_start_id - sentinel_idx
                input_ids.append(sentinel_id)
                labels.append(sentinel_id)  # Start target with same sentinel
                sentinel_idx += 1
            # Add noise token to target only
            labels.append(token)
        else:
            # Non-noise token goes to input
            input_ids.append(token)
            
        prev_token_is_noise = is_noise
    
    # ✅ Thêm EOS vào cuối labels
    labels.append(tokenizer.eos_token_id)
    
    return input_ids, labels

# Load data based on mode
if master_process: 
    print(f"Running in '{'Fine-tuning' if finetune else 'Pre-training'}' mode.")

if finetune:
    print("Loading fine-tuning data...")
    all_data = load_finetune_data()
else:
    print("Loading Vietnamese legal data for pre-training...")
    all_data = load_legal_data()
    print(f"Loaded {len(all_data)} legal articles")

# Split train/val
split_idx = int(0.9 * len(all_data))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]
print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

def get_batch(split):
    """
    HÀM GET_BATCH ĐA NĂNG - Hỗ trợ cả pre-training và fine-tuning
    """
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError(f"Data split '{split}' is empty. Check data loading.")

    if finetune:
        # --- LẤY BATCH CHO FINE-TUNING ---
        ix = np.random.randint(len(data), size=(batch_size,))
        batch_pairs = [data[i] for i in ix]
        
        # Lấy các cặp (context, relations)
        source_texts = [pair[0] for pair in batch_pairs]
        target_texts = [pair[1] for pair in batch_pairs]
        
        # Tokenize source và target
        input_encodings = tokenizer(source_texts, padding=True, truncation=True, 
                                  max_length=max_source_length, return_tensors="pt")
        target_encodings = tokenizer(target_texts, padding=True, truncation=True, 
                                   max_length=max_target_length, return_tensors="pt")
        
        input_ids = input_encodings.input_ids
        attention_mask = input_encodings.attention_mask.to(torch.bool)
        labels = target_encodings.input_ids
        
        # ✅ Tạo decoder_input_ids đúng cách với eos_token_id
        temp_decoder_input_ids = torch.cat([torch.full((labels.shape[0], 1), tokenizer.eos_token_id), labels[:, :-1]], dim=-1)
        decoder_attention_mask = (temp_decoder_input_ids != tokenizer.pad_token_id)
        decoder_input_ids = temp_decoder_input_ids
        
    else:
        # --- LẤY BATCH CHO PRE-TRAINING (logic cũ) ---
        batch_input_ids = []
        batch_labels = []
        
        for _ in range(batch_size):
            article_tokens = data[np.random.randint(len(data))]
            input_tokens, target_tokens = create_t5_spans(article_tokens)
            
            # Cắt bớt và đệm
            input_padded = input_tokens[:max_source_length] + [tokenizer.pad_token_id] * (max_source_length - len(input_tokens))
            labels_padded = target_tokens[:max_target_length] + [tokenizer.pad_token_id] * (max_target_length - len(target_tokens))
            
            batch_input_ids.append(input_padded)
            batch_labels.append(labels_padded)
        
        # Chuyển thành tensor
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        labels = torch.tensor(batch_labels, dtype=torch.long)
        
        # ✅ Tạo attention mask đúng kiểu boolean
        attention_mask = (input_ids != tokenizer.pad_token_id)
        
        # ✅ Tạo decoder_input_ids đúng cách cho pre-training  
        temp_decoder_input_ids = torch.cat([torch.full((labels.shape[0], 1), tokenizer.eos_token_id), labels[:, :-1]], dim=-1)
        decoder_attention_mask = (temp_decoder_input_ids != tokenizer.pad_token_id)
        decoder_input_ids = temp_decoder_input_ids

    # Chuyển lên GPU
    if device_type == 'cuda':
        input_ids, labels, attention_mask, decoder_attention_mask, decoder_input_ids = (
            input_ids.pin_memory().to(device, non_blocking=True),
            labels.pin_memory().to(device, non_blocking=True),
            attention_mask.pin_memory().to(device, non_blocking=True),
            decoder_attention_mask.pin_memory().to(device, non_blocking=True),
            decoder_input_ids.pin_memory().to(device, non_blocking=True)
        )
    else:
        input_ids, labels, attention_mask, decoder_attention_mask, decoder_input_ids = (
            input_ids.to(device), labels.to(device), attention_mask.to(device), decoder_attention_mask.to(device), decoder_input_ids.to(device)
        )
    
    # Trả về dữ liệu theo format model cần
    return input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask

# Model initialization arguments
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    head_dim=head_dim, 
    rank=rank, 
    q_rank=q_rank, 
    using_groupnorm=using_groupnorm,
    vocab_size=len(tokenizer),  # Use actual tokenizer size after adding special tokens
    dropout=dropout,
    pad_token_id=tokenizer.pad_token_id,      # 0 - Correct
    eos_token_id=tokenizer.eos_token_id,      # 3 - Correct (not 1!)
    decoder_start_token_id=tokenizer.eos_token_id  # 3 - Use EOS as decoder start
)

print(f"🔧 TOKENIZER DEBUG INFO:")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
print(f"  EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"  UNK token: '{tokenizer.unk_token}' (id: {tokenizer.unk_token_id})")

# ✅ Verify sentinel tokens
try:
    sentinel_test = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    print(f"  Sentinel <extra_id_0>: {sentinel_test}")
except:
    print(f"  ⚠️ Could not find <extra_id_0> token!")

print(f"🎯 MODEL CONFIG:")
print(f"  decoder_start_token_id: {tokenizer.eos_token_id} (should match EOS)")

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9

# --- KHỐI KHỞI TẠO MODEL VÀ RESUME ĐÃ SỬA LẠI ---
if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    # Khởi tạo model từ đầu với các tham số đã định nghĩa
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    
    # Kiểm tra xem thư mục checkpoint có tồn tại không
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}. Cannot resume.")

    # Tải lại model từ checkpoint đã lưu. 
    # from_pretrained sẽ tự động tải cả config và trọng số
    model = ViLegalJERE.from_pretrained(out_dir)
    
    # Tải lại trạng thái của optimizer và các biến tiến trình
    optimizer_state_path = os.path.join(out_dir, 'optimizer.pt')
    if os.path.exists(optimizer_state_path):
        checkpoint = torch.load(optimizer_state_path, map_location=device)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        if master_process:
            print(f"Resumed successfully from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")
    else:
        # Nếu không tìm thấy file optimizer, bắt đầu từ đầu nhưng vẫn dùng model đã tải
        print(f"Warning: optimizer.pt not found in {out_dir}. Starting optimizer from scratch.")
        # iter_num và best_val_loss sẽ giữ giá trị mặc định là 0 và 1e9

# Fix vocabulary size mismatch between model and tokenizer
if model.config.vocab_size != len(tokenizer):
    if master_process:
        print(f"Vocab size mismatch: model={model.config.vocab_size}, tokenizer={len(tokenizer)}")
        print("Resizing model embeddings to match tokenizer...")
    model.resize_token_embeddings(len(tokenizer))
    if master_process:
        print(f"Model embeddings resized to {len(tokenizer)}")

model.to(device)

# Calculate parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000

print(f"Model initialized with {param_count_m:.1f}M parameters")

# Update output directory for Kaggle
if init_from != 'resume':
    wandb_run_name = f"ViLegal_{int(param_count_m)}m_T5small_Kaggle_{current_date}"
    out_dir = f"/kaggle/working/out_vilegal_t5small"

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize scaler and optimizer
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

# Tải lại trạng thái optimizer nếu resume và file tồn tại
if init_from == 'resume':
    optimizer_state_path = os.path.join(out_dir, 'optimizer.pt')
    if os.path.exists(optimizer_state_path):
        checkpoint = torch.load(optimizer_state_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process:
            print("Optimizer state loaded successfully")

# Compile model
if compile:
    print("Compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap with DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train/val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch(split)
            with ctx:
                # ✅ FIXED: Pass encoder attention mask correctly in evaluation
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # Encoder attention mask
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,  # Decoder attention mask
                    labels=labels
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, schedule='cosine'):
    """Learning rate scheduler với warmup và decay cải thiện"""
    if it < warmup_iters:
        # ✅ Linear warmup
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    
    # ✅ Decay phase  
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    if schedule == 'cosine':
        # Cosine annealing
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    elif schedule == 'linear':
        # Linear decay
        coeff = 1.0 - decay_ratio
    elif schedule == 'constant':
        # Constant after warmup
        coeff = 1.0
    else:
        # Default to cosine
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (learning_rate - min_lr)

# Logging setup
if wandb_log and master_process:
    import wandb
    
    # Tự động đăng nhập wandb
    try:
        # Thử đăng nhập bằng API key từ environment hoặc file config
        wandb.login()
        print("✅ Đăng nhập wandb thành công!")
    except Exception as e:
        print(f"⚠️ Không thể đăng nhập wandb với API key: {e}")
        print("🔄 Chuyển sang chế độ anonymous...")
        # Fallback sang chế độ anonymous nếu không có API key
        wandb.login(anonymous="allow")
        print("✅ Sử dụng wandb ở chế độ anonymous!")
    
    wandb_config = {
        'model_args': model_args,
        'training_args': {
            'batch_size': batch_size,
            'max_source_length': max_source_length,
            'max_target_length': max_target_length,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'max_iters': max_iters,
            'lr_decay_iters': lr_decay_iters,
            'eval_interval': eval_interval,
            'eval_iters': eval_iters,
            'log_interval': log_interval
        },
        'optimizer_args': {
            'optimizer_name': optimizer_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'grad_clip': grad_clip,
            'decay_lr': decay_lr,
            'warmup_iters': warmup_iters,
            'min_lr': min_lr,
            'schedule': schedule
        }
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

# Training loop
mode_text = "Fine-tuning" if finetune else "Pre-training"
print(f"Starting {mode_text} ViLegalJERE with {param_count_m:.1f}M parameters...")
print(f"Training data size: {len(train_data)}, Val data size: {len(val_data)}")
print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
print(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
print(f"Mode: {mode_text}, Learning rate: {learning_rate}, Max iters: {max_iters}")

input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch('train')
print(f"First batch shapes - Input: {input_ids.shape}, Decoder: {decoder_input_ids.shape}, Labels: {labels.shape}")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
clip_time = 0

while True:
    # Set learning rate
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation (skip first iteration)
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }, step=iter_num)
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"Saving checkpoint to {out_dir}")
                raw_model.save_pretrained(out_dir, safe_serialization=False)
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(optimizer_state, os.path.join(out_dir, 'optimizer.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            # ✅ FIXED: Pass encoder attention mask correctly for T5 architecture
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Encoder attention mask
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,  # Decoder attention mask
                labels=labels
            )
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            loss = loss / gradient_accumulation_steps
        
        # Get next batch
        input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_trained += tokens_per_iter
    tokens_trained_B = tokens_trained / 1e9

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.get_num_params() * batch_size * gradient_accumulation_steps * (max_source_length + max_target_length) * 6 / (dt * 1e12)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        tokens_per_sec = tokens_per_iter / dt
        tokens_per_sec_M = tokens_per_sec / 1_000_000
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tps (M) {tokens_per_sec_M:.2f}, tokens trained {tokens_trained:.2f}B")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "train/clip_rate": clip_time / (iter_num + 1),
                "train/grad_norm": total_norm.item() if grad_clip != 0.0 else 0.0,
                "train/iter_time_ms": dt * 1000,
                "train/mfu": running_mfu * 100,
                "train/tokens_per_sec_M": tokens_per_sec_M,
                "train/tokens_trained_B": tokens_trained_B,
                "gpu/memory_allocated_MB": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu/max_memory_allocated_MB": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }, step=iter_num)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group() 