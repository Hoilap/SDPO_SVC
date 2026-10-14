"""Shared full stage test for continual and tail-causal experiments."""

import argparse
from collections import Counter
import json
from pathlib import Path
import os
import subprocess
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--val-files", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--scorer", default="verl/utils/reward_score/feedback/__init__.py")
    parser.add_argument("--dump-generations", action="store_true")
    parser.add_argument("--agent-loop-workers", type=int, default=int(os.environ.get("EVAL_AGENT_LOOP_WORKERS", "8")))
    args = parser.parse_args()
    if args.agent_loop_workers < 1:
        parser.error("--agent-loop-workers must be positive")
    manifest = {"stages": [{"name": args.name, "model": args.model, "val_files": args.val_files}]}
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = {}
    plans = []
    for stage in manifest["stages"]:
        model = Path(stage["model"])
        assert (model / "config.json").is_file(), model
        assert list(model.glob("*.safetensors")), model
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        counts = Counter()
        longest = 0
        for file in stage["val_files"]:
            # Include tokenizer identity in the cache; every boundary is checked.
            key = (file, tokenizer.backend_tokenizer.to_str())
            if key not in cache:
                rows = pq.read_table(file, columns=["prompt", "data_source"]).to_pylist()
                file_counts = Counter()
                file_longest = 0
                for row in rows:
                    prompt = tokenizer.apply_chat_template(
                        row["prompt"], tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    file_longest = max(file_longest, length)
                    file_counts[row["data_source"]] += 1
                cache[key] = (file_counts, file_longest)
            file_counts, file_longest = cache[key]
            counts.update(file_counts)
            longest = max(longest, file_longest)
        # Retain all examples, including those previously filtered above 2048.
        max_prompt = max(2048, ((longest + 127) // 128) * 128)
        plan = dict(stage, counts=dict(counts), questions=sum(counts.values()),
                    responses=sum(counts.values()) * 8, longest_prompt=longest,
                    max_prompt_length=max_prompt, max_response_length=8192)
        plans.append(plan)
        print("FULL TEST PREFLIGHT " + json.dumps(plan), flush=True)
    (root / "plan.json").write_text(json.dumps(plans, indent=2) + "\n")
    if args.preflight_only:
        return
    for plan in plans:
        out = root / plan["name"]
        if (out / "0.metrics.json").exists():
            raise FileExistsError(f"Refusing to overwrite completed evaluation: {out}")
        env = dict(os.environ, TASK="evaluation", EXPERIMENT="full-test-" + plan["name"],
                   VLLM_USE_V1="1", TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="2")
        train_file = max(plan["val_files"], key=lambda file: pq.read_metadata(file).num_rows)
        command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-name", "baseline_grpo",
            "ray_kwargs.ray_init.num_cpus=24",
            "data.train_files=" + json.dumps([train_file]),
            "data.train_max_samples=32", "data.shuffle=False",
            "data.val_files=" + json.dumps(plan["val_files"]),
            "data.val_max_samples=-1", "data.training_val_fraction=1.0",
            "data.filter_overlong_prompts=False", "data.truncation=error",
            "data.max_prompt_length=" + str(plan["max_prompt_length"]),
            "data.max_response_length=8192", "data.val_batch_size=8",
            "data.dataloader_num_workers=2",
            "max_model_len=" + str(plan["max_prompt_length"] + 8192),
            "actor_rollout_ref.model.path=" + plan["model"],
            "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
            "actor_rollout_ref.rollout.agent.num_workers=" + str(args.agent_loop_workers),
            "actor_rollout_ref.rollout.val_kwargs.n=8",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
            "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
            "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
            "trainer.val_only=True", "trainer.val_before_train=True",
            "trainer.resume_mode=disable", "trainer.logger=['console']",
            "trainer.n_gpus_per_node=4", "trainer.nnodes=1",
            "trainer.validation_dump_generations=" + str(args.dump_generations),
            "trainer.validation_data_dir=" + str(out),
            "trainer.default_local_dir=" + str(root / "runtime" / plan["name"]),
            "custom_reward_function.path=" + str(Path(args.scorer).resolve()),
            "reward_model.launch_reward_fn_async=False"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "invocation.json").write_text(json.dumps(command, indent=2) + "\n")
        print("START FULL TEST " + plan["name"], flush=True)
        subprocess.run(command, env=env, check=True)
        result = json.loads((out / "0.metrics.json").read_text())
        assert result["num_samples"] == plan["responses"], (plan, result["num_samples"])
        result.update(evaluation_kind="test", sampling_fraction=1.0, num_questions=plan["questions"],
                      responses_per_question=8, questions_by_source=plan["counts"])
        (out / "0.metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print("COMPLETE FULL TEST " + plan["name"], flush=True)


if __name__ == "__main__":
    main()
