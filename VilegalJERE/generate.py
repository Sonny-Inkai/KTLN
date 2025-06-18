import torch
import json
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalJERE

def generate_relations(model, tokenizer, device, context_text, max_length=512):
    """Generate relation extraction from context"""
    # Tokenize input (encoder input)
    inputs = tokenizer(
        context_text,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate using the model's custom generate method
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output (skip the start token)
    generated_text = tokenizer.decode(outputs[0, 1:], skip_special_tokens=True)
    return generated_text

