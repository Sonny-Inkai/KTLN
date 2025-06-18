import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig

@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None

@dataclass
class DecoderOutput:
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().type_as(x)
            self.sin_cached = freqs.sin().type_as(x)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CPLinear(nn.Module):
    # Bilinear form of x using CP decomposition
    def __init__(self, in_features, n_head, head_dim, rank: int = 2, q_rank: int = 8):
        super(CPLinear, self).__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank
        self.q_rank = q_rank

        self.W_A_q = nn.Linear(in_features, n_head * q_rank, bias=False)
        self.W_A_k = nn.Linear(in_features, n_head * rank, bias=False)
        self.W_A_v = nn.Linear(in_features, n_head * rank, bias=False)

        self.W_B_q = nn.Linear(in_features, q_rank * head_dim, bias=False)
        self.W_B_k = nn.Linear(in_features, rank * head_dim, bias=False)
        self.W_B_v = nn.Linear(in_features, rank * head_dim, bias=False)
        
        # ✅ FIXED: Only add rotary for self-attention, NOT cross-attention
        self.rotary = Rotary(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        W_A_q_tensor = self.W_A_q.weight.view(self.in_features, self.n_head, self.q_rank)
        W_A_k_tensor = self.W_A_k.weight.view(self.in_features, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.in_features, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_q_tensor)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_q.weight.data = W_A_q_tensor.view_as(self.W_A_q.weight)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_q_tensor = self.W_B_q.weight.view(self.in_features, self.q_rank, self.head_dim)
        W_B_k_tensor = self.W_B_k.weight.view(self.in_features, self.rank, self.head_dim)
        W_B_v_tensor = self.W_B_v.weight.view(self.in_features, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_q_tensor)
        nn.init.xavier_uniform_(W_B_k_tensor)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_q.weight.data = W_B_q_tensor.view_as(self.W_B_q.weight)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)
        
    def forward(self, x, apply_rope=True):
        # ✅ FIXED: Add apply_rope parameter to control RoPE usage
        batch_size, seq_len, _ = x.size()

        A_q = self.W_A_q(x).view(batch_size, seq_len, self.n_head, self.q_rank)
        A_k = self.W_A_k(x).view(batch_size, seq_len, self.n_head, self.rank)
        A_v = self.W_A_v(x).view(batch_size, seq_len, self.n_head, self.rank)

        B_q = self.W_B_q(x).view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(x).view(batch_size, seq_len, self.rank, self.head_dim)
        B_v = self.W_B_v(x).view(batch_size, seq_len, self.rank, self.head_dim)
        
        # ✅ FIXED: Only apply RoPE when requested (for self-attention)
        if apply_rope:
            cos, sin = self.rotary(B_q)
            B_q, B_k = apply_rotary_emb(B_q, cos, sin), apply_rotary_emb(B_k, cos, sin)
        
        A_q = A_q.view(batch_size * seq_len, self.n_head, self.q_rank)
        A_k = A_k.view(batch_size * seq_len, self.n_head, self.rank)
        A_v = A_v.view(batch_size * seq_len, self.n_head, self.rank)

        B_q = B_q.view(batch_size * seq_len, self.q_rank, self.head_dim)
        B_k = B_k.view(batch_size * seq_len, self.rank, self.head_dim)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.head_dim)
        
        q = torch.bmm(A_q, B_q).div_(self.q_rank).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank).view(batch_size, seq_len, self.n_head, self.head_dim)

        return q, k, v

class ViLegalSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, is_causal=False):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.rank = config.rank
        self.q_rank = config.q_rank
        self.is_cross_attention = is_cross_attention
        self.is_causal = is_causal

        # ✅ FIXED: Proper Q/K/V setup for cross-attention
        if is_cross_attention:
            # Cross-attention: Q from decoder, K/V from encoder
            self.c_q = CPLinear(self.n_embd, self.n_head, self.head_dim, self.q_rank, self.q_rank)
            self.c_kv = CPLinear(self.n_embd, self.n_head, self.head_dim, self.rank, self.rank)
            # ✅ NO RoPE for cross-attention
        else:
            # Self-attention: Q/K/V from same input
            self.c_qkv = CPLinear(self.n_embd, self.n_head, self.head_dim, self.rank, self.q_rank)

        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        
        self.using_groupnorm = getattr(config, 'using_groupnorm', False)
        if self.using_groupnorm:
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, encoder_hidden_states=None, attention_mask=None):
        B, T, C = x.size()

        if self.is_cross_attention and encoder_hidden_states is not None:
            # ✅ FIXED: Proper cross-attention implementation 
            q, _, _ = self.c_q(x, apply_rope=False)  # Query from decoder input, NO RoPE
            _, k, v = self.c_kv(encoder_hidden_states, apply_rope=False)  # Key/Value from encoder, NO RoPE
        else:
            # Self-attention with RoPE
            q, k, v = self.c_qkv(x, apply_rope=True)  # Apply RoPE for self-attention

        # ✅ FIXED: Proper attention mask handling
        attn_mask_for_spda = None
        if attention_mask is not None:
            # Convert boolean mask to additive mask for scaled_dot_product_attention
            # True = keep token, False = mask token
            if attention_mask.dtype == torch.bool:
                attn_mask_for_spda = ~attention_mask  # Invert for SDPA (True = mask)
            else:
                attn_mask_for_spda = attention_mask == 0  # 0 = mask, 1 = keep
            
            # Ensure proper shape for SDPA: [batch, 1, seq_len, seq_len] or [batch, heads, seq_len, seq_len]
            if attn_mask_for_spda.dim() == 2:
                attn_mask_for_spda = attn_mask_for_spda.unsqueeze(1).unsqueeze(1)
        
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, n_head, T, head_dim)
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask_for_spda,
            is_causal=self.is_causal and not self.is_cross_attention  # ✅ No causal for cross-attention
        )
        
        if self.using_groupnorm:
            y = self.subln(y)
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        return y

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = math.floor(8 / 3 * config.n_embd)
        
        self.c_fc1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        return x

class ViLegalEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ViLegalSelfAttention(config, is_causal=False)
        self.mlp = SwiGLU(config)
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class ViLegalDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ViLegalSelfAttention(config, is_causal=True)
        self.cross_attn = ViLegalSelfAttention(config, is_cross_attention=True)
        self.mlp = SwiGLU(config)
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ln_3 = RMSNorm(config.n_embd)

    def forward(self, x, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None):
        # ✅ FIXED: T5 decoder block order per Google reference
        # 1. Self-attention (causal) - sử dụng attention_mask cho decoder tokens
        x = x + self.self_attn(self.ln_1(x), attention_mask=attention_mask)
        
        # 2. Cross-attention with encoder - sử dụng encoder_attention_mask cho encoder tokens
        if encoder_hidden_states is not None:
            x = x + self.cross_attn(self.ln_2(x), encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask)
        
        # 3. Feed-forward MLP
        x = x + self.mlp(self.ln_3(x))
        return x

class ViLegalConfig(PretrainedConfig):
    model_type = "vilegal_jere"
    
    def __init__(
        self,
        vocab_size: int = 10100,
        n_layer: int = 12,
        n_head: int = 16,
        head_dim: int = 64,
        n_embd: int = 1024,
        rank: int = 4,
        q_rank: int = 8,
        block_size: int = 2048,
        bias: bool = False,
        dropout: float = 0.0,
        using_groupnorm: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
        decoder_start_token_id: int = 3,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_embd = n_embd
        self.rank = rank
        self.q_rank = q_rank
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout
        self.using_groupnorm = using_groupnorm
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs
        )

class ViLegalJERE(PreTrainedModel):
    config_class = ViLegalConfig
    base_model_prefix = "vilegal_jere"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Shared embedding
        self.shared = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            ViLegalEncoderBlock(config) for _ in range(config.n_layer)
        ])
        self.encoder_ln = RMSNorm(config.n_embd)
        
        # Decoder  
        self.decoder_blocks = nn.ModuleList([
            ViLegalDecoderBlock(config) for _ in range(config.n_layer)
        ])
        self.decoder_ln = RMSNorm(config.n_embd)
        
        # LM head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings to match new vocabulary size"""
        old_embeddings = self.get_input_embeddings()
        if old_embeddings.num_embeddings == new_num_tokens:
            return
        
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy existing embeddings
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        # Initialize new tokens with small random values
        if new_num_tokens > old_embeddings.num_embeddings:
            with torch.no_grad():
                new_embeddings.weight.data[old_embeddings.num_embeddings:, :].normal_(mean=0.0, std=0.02)
        
        self.set_input_embeddings(new_embeddings)
        self.lm_head.weight = new_embeddings.weight  # Tie weights
        
        # Update config vocab_size
        self.config.vocab_size = new_num_tokens

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Encoder
        if input_ids is not None:
            encoder_outputs = self.encode(input_ids, attention_mask, output_hidden_states)
            encoder_hidden_states = encoder_outputs.last_hidden_state
        else:
            encoder_hidden_states = None
            
        # Decoder - ✅ FIXED: Pass attention masks correctly
        decoder_outputs = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask=decoder_attention_mask,  # Decoder self-attention mask
            encoder_attention_mask=attention_mask,  # Encoder cross-attention mask  
            output_hidden_states=output_hidden_states,
        )
        
        logits = decoder_outputs.logits
        loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            
            # ✅ FIXED: Properly mask pad tokens (id=0) to ignore in loss
            shift_labels_masked = shift_labels.clone()
            shift_labels_masked[shift_labels_masked == self.config.pad_token_id] = -100
            
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels_masked.view(-1))

        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'encoder_last_hidden_state': encoder_hidden_states,
            'decoder_hidden_states': decoder_outputs.hidden_states,
        }

    def encode(self, input_ids, attention_mask=None, output_hidden_states=None):
        x = self.shared(input_ids)
        
        all_hidden_states = () if output_hidden_states else None
        
        for block in self.encoder_blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x = block(x, attention_mask=attention_mask)
            
        x = self.encoder_ln(x)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)
            
        return EncoderOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
        )

    def decode(
        self,
        input_ids,
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
        output_hidden_states=None,
    ):
        x = self.shared(input_ids)
        
        all_hidden_states = () if output_hidden_states else None
        
        for block in self.decoder_blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x = block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,  # For decoder self-attention (causal)
                encoder_attention_mask=encoder_attention_mask,  # For encoder cross-attention
            )
            
        x = self.decoder_ln(x)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)
            
        logits = self.lm_head(x)
        
        return DecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
        )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=None,
        eos_token_id=None,
    ):
        # ✅ FIXED: Handle None values properly by falling back to config
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        
        # Fallback to sensible defaults if config values are somehow still None
        if pad_token_id is None:
            pad_token_id = 0
        if eos_token_id is None:
            eos_token_id = 3
        if decoder_start_token_id is None:
            decoder_start_token_id = eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encode input
        encoder_outputs = self.encode(input_ids, attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Initialize decoder input with proper token type
        decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        
        # Generate tokens
        for _ in range(max_length - 1):
            decoder_outputs = self.decode(
                decoder_input_ids,
                encoder_hidden_states,
                encoder_attention_mask=attention_mask
            )
            
            logits = decoder_outputs.logits[:, -1, :]
            
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-p filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check for EOS token
            if (next_token == eos_token_id).all():
                break
                
        return decoder_input_ids

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.shared.weight.numel()
        return n_params 