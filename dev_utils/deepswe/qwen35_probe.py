"""Real tiny LoRA update/export of the designated model; synthetic input only."""
import json
import os
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

source = os.environ['MODEL_SOURCE']
model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16,
    attn_implementation='eager', local_files_only=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
model.train()
model.config.use_cache = False
inputs = tokenizer('Write a Python function that adds two integers.\ndef add(a, b):\n    return a + b', return_tensors='pt').to('cuda')
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
loss = model(**inputs, labels=inputs['input_ids']).loss
assert torch.isfinite(loss), loss
loss.backward()
assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters() if p.requires_grad)
optimizer.step()
merged = model.merge_and_unload()
merged.config.use_cache = True
merged.save_pretrained('/output/checkpoint', safe_serialization=True)
tokenizer.save_pretrained('/output/checkpoint')
AutoProcessor.from_pretrained(source, local_files_only=True).save_pretrained('/output/checkpoint')
Path('/output/train_probe.json').write_text(json.dumps({'loss': loss.item(), 'cuda': torch.cuda.get_device_name()}))
