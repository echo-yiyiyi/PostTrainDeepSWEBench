"""Inside-container compatibility probe; uses random weights, never benchmark data."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def main():
    phase = sys.argv[1]
    out = Path('/output')
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in md.distributions()}
    (out / f'{phase}_packages.json').write_text(json.dumps(versions, indent=2, sort_keys=True))
    assert torch.cuda.is_available()
    (out / f'{phase}_runtime.json').write_text(json.dumps({
        'python': sys.version, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'uid': os.getuid(),
        'device': torch.cuda.get_device_name(0)}, indent=2))
    print('CUDA:', torch.cuda.get_device_name(0), flush=True)
    template = Path('/templates/qwen3.jinja').read_text()
    messages = [{'role': 'system', 'content': 'Return JSON.'},
                {'role': 'user', 'content': 'Print hello.'}]

    if phase == 'train':
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(os.environ['COMPAT_BASE'], local_files_only=True)
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=128,
            intermediate_size=256, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=64, max_position_embeddings=8192,
            eos_token_id=tok.eos_token_id, tie_word_embeddings=True)).cuda()
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        batch = tok('This is a synthetic compatibility check.', return_tensors='pt').to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        loss = model(**batch, labels=batch['input_ids']).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        model = model.merge_and_unload().eval()
        model.save_pretrained(Path('/checkpoint'), safe_serialization=True)
        tok.save_pretrained(Path('/checkpoint'))
    else:
        tok = AutoTokenizer.from_pretrained(Path('/checkpoint'), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(Path('/checkpoint'), local_files_only=True).cuda().eval()

    rendered = tok.apply_chat_template(messages, chat_template=template,
        tokenize=False, add_generation_prompt=True)
    ids = tok(rendered, return_tensors='pt').to('cuda')
    with torch.no_grad():
        logits = model(**ids).logits[:, -1].cpu()
    signature = {'rendered': rendered, 'ids': ids['input_ids'].tolist(),
        'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
        'vocab_size': len(tok)}
    if phase == 'train':
        (out / 'tokenizer_reference.json').write_text(json.dumps(signature))
        torch.save(logits, out / 'logits.pt')
    else:
        assert signature == json.loads((out / 'tokenizer_reference.json').read_text())
        torch.testing.assert_close(logits, torch.load(out / 'logits.pt', weights_only=True), rtol=1e-4, atol=1e-5)
        del model, logits
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(Path('/checkpoint')), max_model_len=8192,
                  gpu_memory_utilization=0.25, enforce_eager=True, dtype='bfloat16')
        result = llm.generate([rendered], SamplingParams(max_tokens=8, temperature=0))
        assert len(result) == 1 and result[0].outputs[0].token_ids
        (out / 'vllm_pass.json').write_text(json.dumps({'tokens': result[0].outputs[0].token_ids}))
    print('PASS:', phase, 'export/load, tokenizer, Qwen3 template, CUDA', flush=True)


if __name__ == '__main__':
    main()
