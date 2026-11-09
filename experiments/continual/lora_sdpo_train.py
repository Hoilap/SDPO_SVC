"""Run the frozen SDPO trainer with LoRA and a dense effective-weight EMA teacher."""
from contextlib import ExitStack
import runpy


def canonical(name):
    return name.replace('_fsdp_wrapped_module.', '').removeprefix('base_model.model.').replace('.base_layer.', '.')


def update_dense_teacher(teacher, student, rate):
    """Called with full parameters: EMA(W_base + scaling * B @ A), not EMA(A/B)."""
    import torch
    student_params = {canonical(n): p for n, p in student.named_parameters() if 'lora_' not in n}
    deltas = {}
    for name, module in student.named_modules():
        if hasattr(module, 'get_delta_weight') and hasattr(module, 'lora_A'):
            deltas[canonical(name) + '.weight'] = module
    with torch.no_grad():
        for name, target in teacher.named_parameters():
            key = canonical(name)
            source = student_params[key]
            if target.shape != source.shape:
                raise ValueError(f'EMA shape mismatch: {key}')
            target.mul_(1 - rate).add_(source.to(target), alpha=rate)
            if key in deltas:
                delta = deltas[key].get_delta_weight('default')
                target.add_(delta.to(target), alpha=rate)


def install():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.workers import fsdp_workers
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    if getattr(DataParallelPPOActor, '_lora_sdpo_installed', False):
        return
    DataParallelPPOActor._lora_sdpo_installed = True
    original_build = fsdp_workers.ActorRolloutRefWorker._build_model_optimizer
    original_update = DataParallelPPOActor._update_teacher

    def build(self, *args, **kwargs):
        is_lora = self._is_lora
        if kwargs.get('role') == 'ref':
            self._is_lora = False
        try:
            return original_build(self, *args, **kwargs)
        finally:
            self._is_lora = is_lora

    def update(self):
        cfg = getattr(self.config, 'self_distillation', None)
        if not cfg or self.config.policy_loss.get('loss_mode') != 'sdpo':
            return original_update(self)
        if cfg.get('teacher_regularization', 'ema') != 'ema':
            return original_update(self)
        rate = cfg.get('teacher_update_rate', 0.0)
        if not rate:
            return
        with ExitStack() as stack:
            for model, writeback in [(self.actor_module, False), (self.teacher_module, True)]:
                if isinstance(model, FSDP):
                    stack.enter_context(FSDP.summon_full_params(model, writeback=writeback))
            student = self.actor_module.module if isinstance(self.actor_module, FSDP) else self.actor_module
            teacher = self.teacher_module.module if isinstance(self.teacher_module, FSDP) else self.teacher_module
            if not hasattr(student, 'peft_config'):
                return original_update(self)
            update_dense_teacher(teacher, student, rate)

    fsdp_workers.ActorRolloutRefWorker._build_model_optimizer = build
    DataParallelPPOActor._update_teacher = update
    # Ray workers import their class in a fresh interpreter; patch installation
    # is also supplied through the Ray runtime_env worker_process_setup_hook.


def worker_setup():
    install()


if __name__ == '__main__':
    install()
    runpy.run_module('experiments.continual.warmup_ab_train', run_name='__main__')
