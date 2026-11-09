"""Export the exact saved adapter merged into its stage initialization model."""
import argparse
from pathlib import Path


def export(base, checkpoint, target):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    adapter = Path(checkpoint) / 'lora_adapter'
    assert (adapter / 'adapter_config.json').is_file(), adapter
    assert (adapter / 'adapter_model.safetensors').is_file(), adapter
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.float32,
                                               device_map='cpu', trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(target), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base), trust_remote_code=True).save_pretrained(str(target))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--target', required=True)
    a = p.parse_args()
    export(a.base, a.checkpoint, a.target)
