"""LoRA teacher updates must match a dense EMA, including consecutive updates."""
import torch
from torch import nn

from experiments.continual.lora_sdpo_train import update_dense_teacher


class LowRankLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(3, 2)
        self.lora_A = nn.ParameterDict({'default': nn.Parameter(torch.randn(1, 3))})
        self.lora_B = nn.ParameterDict({'default': nn.Parameter(torch.randn(2, 1))})

    def get_delta_weight(self, name):
        return 2.0 * self.lora_B[name] @ self.lora_A[name]


def test_dense_teacher_matches_effective_weight_ema():
    torch.manual_seed(7)
    student = nn.Module()
    student.proj = LowRankLinear()
    teacher = nn.Module()
    teacher.proj = nn.Linear(3, 2)
    for _ in range(3):
        before_weight = teacher.proj.weight.detach().clone()
        before_bias = teacher.proj.bias.detach().clone()
        effective = student.proj.base_layer.weight + student.proj.get_delta_weight('default')
        expected = before_weight * .95 + effective * .05
        update_dense_teacher(teacher, student, .05)
        torch.testing.assert_close(teacher.proj.weight, expected)
        torch.testing.assert_close(teacher.proj.bias, before_bias * .95 + student.proj.base_layer.bias * .05)
        with torch.no_grad():
            student.proj.lora_A['default'].add_(.1)
            student.proj.lora_B['default'].sub_(.2)
    assert all(p.grad is None for p in teacher.parameters())


def test_export_contains_adapter_update(tmp_path):
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM
    from experiments.continual.lora_export import export

    torch.manual_seed(3)
    base = tmp_path / 'base'
    checkpoint = tmp_path / 'actor'
    target = tmp_path / 'merged'
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         tie_word_embeddings=False)
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    model.save_pretrained(base)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'<unk>': 0}, unk_token='<unk>')))
    tokenizer.save_pretrained(base)
    student = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
                                             task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name, param in student.named_parameters():
            if 'lora_B' in name:
                param.fill_(.2)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = student(ids).logits.float()
    student.save_pretrained(checkpoint / 'lora_adapter')
    layer = student.get_base_model().model.layers[0].self_attn.q_proj
    expected_weight = layer.base_layer.weight.float() + layer.get_delta_weight('default').float()
    teacher = LlamaForCausalLM.from_pretrained(base)
    before = teacher.model.layers[0].self_attn.q_proj.weight.detach().clone()
    update_dense_teacher(teacher, student, .05)
    torch.testing.assert_close(teacher.model.layers[0].self_attn.q_proj.weight,
                               .95 * before + .05 * expected_weight, atol=1e-6, rtol=1e-5)
    export(base, checkpoint, target)
    merged = AutoModelForCausalLM.from_pretrained(target).eval()
    torch.testing.assert_close(merged.model.layers[0].self_attn.q_proj.weight,
                               expected_weight, atol=1e-7, rtol=1e-6)
    with torch.no_grad():
        actual = merged(ids).logits.float()
    torch.testing.assert_close(actual, expected, atol=.004, rtol=.03)
    assert not any('lora_' in n for n, _ in merged.named_parameters())
