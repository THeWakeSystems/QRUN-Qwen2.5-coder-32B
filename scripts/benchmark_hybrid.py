"""
benchmark_hybrid.py — Comprehensive testing of Hybrid checkpoint for dialogue/coding/reasoning capabilities

Usage:
    python benchmark_hybrid.py --checkpoint /private/Springhead-v1.0/checkpoints_hybrid_v2/epoch_2.pt
    python3 benchmark_hybrid.py --checkpoint /private/Springhead-v1.0/checkpoints_hybrid_v2/epoch_2.pt --model_path /private/models/Qwen2.5-coder-32B --device cuda --dtype auto --max_memory_per_device 14GiB
"""
import argparse
import json
import os
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer
from accelerate import dispatch_model
from model.CustomQwen32B_hybrid import create_hybrid_model


def resolve_runtime_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def get_device_count(device):
    if device == "cuda":
        return torch.cuda.device_count()
    if device == "musa" and hasattr(torch, "musa"):
        return torch.musa.device_count()
    if device == "npu" and hasattr(torch, "npu"):
        return torch.npu.device_count()
    return 1


def resolve_checkpoint_path(checkpoint_path):
    candidates = []
    raw = checkpoint_path.strip()

    if raw:
        candidates.append(raw)

    if raw.startswith("archive/"):
        candidates.append(raw[len("archive/"):])

    if raw.startswith("/archive/"):
        candidates.append(raw[len("/archive"):])

    # Allow passing file name only
    if raw and os.path.sep not in raw:
        candidates.append(os.path.join("checkpoints_hybrid_v2", raw))

    expanded = list(candidates)
    for candidate in expanded:
        if not os.path.isabs(candidate):
            candidates.append(os.path.join(os.getcwd(), candidate))

    seen = set()
    ordered = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    for candidate in ordered:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Checkpoint file does not exist.\n"
        f"Input path: {checkpoint_path}\n"
        f"Current working directory: {os.getcwd()}\n"
        "Attempted paths:\n  - " + "\n  - ".join(ordered)
    )


def resolve_model_path(model_path):
    raw = model_path.strip()
    candidates = []
    if raw:
        candidates.append(raw)

    # Support passing just model name or relative path
    if raw and not os.path.isabs(raw):
        candidates.append(os.path.join(os.getcwd(), raw))

    expanded = list(candidates)
    for candidate in expanded:
        if not os.path.isabs(candidate):
            candidates.append(os.path.join(os.getcwd(), candidate))

    seen = set()
    ordered = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    for candidate in ordered:
        if os.path.isdir(candidate):
            config_path = os.path.join(candidate, "config.json")
            if os.path.exists(config_path):
                return candidate

    raise FileNotFoundError(
        "Available local model directory not found (Offline environment will not access HuggingFace).\n"
        f"Input model_path: {model_path}\n"
        "Please use an absolute local model path, e.g., /private/models/Qwen2.5-coder-32B\n"
        "Attempted paths:\n  - " + "\n  - ".join(ordered)
    )


def build_balanced_device_map(model, device_ids):
    total_layers = len(model.model.layers)
    if not device_ids:
        raise RuntimeError("No available GPU devices for allocation.")

    device_map = {}
    first_device = device_ids[0]
    last_device = device_ids[-1]
    device_map["model.embed_tokens"] = first_device

    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is not None:
        device_map["model.rotary_emb"] = first_device

    layers_per_device = max(1, (total_layers + len(device_ids) - 1) // len(device_ids))
    for layer_idx in range(total_layers):
        device_idx = min(layer_idx // layers_per_device, len(device_ids) - 1)
        device_map[f"model.layers.{layer_idx}"] = device_ids[device_idx]

    device_map["model.norm"] = last_device
    device_map["lm_head"] = last_device
    return device_map


def get_input_device(model):
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def load_model(checkpoint_path, model_path, replace_layers=None,
               device="auto", dtype="auto", max_memory_per_device="20GiB"):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    print("=" * 70)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 70)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    replace_layers = saved_args.get("replace_layers", replace_layers or list(range(48, 64)))

    model = create_hybrid_model(
        model_path=model_path,
        replace_layers=replace_layers,
        u_proj_output_dim=saved_args.get("u_proj_output_dim", 4),
        mlp_hidden_size=saved_args.get("mlp_hidden_size", 128),
        n_reuploads=saved_args.get("n_reuploads", 3),
        block_size=saved_args.get("block_size", 64),
        entangle_rank=saved_args.get("entangle_rank", 64),
        n_deep_layers=saved_args.get("n_deep_layers", 2),
        d_hidden=saved_args.get("d_hidden", 32),
        compute_dtype=saved_args.get("dtype", dtype),
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    runtime_device = resolve_runtime_device(device)
    n_devices = get_device_count(runtime_device)

    if runtime_device == "cuda" and n_devices > 1:
        max_mem = {}
        reserve_bytes = int(1.0 * (1024 ** 3))
        for idx in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            budget = int(max(0, free_bytes - reserve_bytes) * 0.90)
            if budget < int(1.0 * (1024 ** 3)):
                continue
            if isinstance(max_memory_per_device, str) and max_memory_per_device.lower().endswith("gib"):
                requested_gib = float(max_memory_per_device[:-3])
                budget = min(budget, int(requested_gib * (1024 ** 3)))
            elif isinstance(max_memory_per_device, str) and max_memory_per_device.lower().endswith("mib"):
                requested_mib = float(max_memory_per_device[:-3])
                budget = min(budget, int(requested_mib * (1024 ** 2)))
            elif isinstance(max_memory_per_device, (int, float)):
                budget = min(budget, int(max_memory_per_device))
            max_mem[idx] = budget

        if not max_mem:
            raise RuntimeError(
                "Not enough available GPU VRAM detected. Please confirm if GPUs are truly idle or terminate VRAM-consuming processes before retrying."
            )

        print("  CUDA VRAM budget:")
        for idx in sorted(max_mem.keys()):
            print(f"    GPU{idx}: {max_mem[idx] / (1024 ** 3):.2f}GiB")

        device_map = build_balanced_device_map(model, sorted(max_mem.keys()))
        model = dispatch_model(model, device_map=device_map)
        print(f"  Dispatched gracefully across {len(max_mem)} CUDA GPUs")
    else:
        if runtime_device == "cpu":
            raise RuntimeError("This script currently supports only GPU inference. Please run in an environment with sufficient GPUs.")
        target = f"{runtime_device}:0"
        model = model.to(target)
        print(f"  Loaded onto {target}")
    return model


def generate(model, tokenizer, prompt, system="You are a helpful assistant.",
             max_new_tokens=512, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_device = get_input_device(model)
    inputs = tokenizer(text, return_tensors="pt").to(input_device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def save_benchmark_results(output_dir, run_data):
    os.makedirs(output_dir, exist_ok=True)
    run_id = run_data["run_id"]
    json_path = os.path.join(output_dir, f"benchmark_{run_id}.json")
    txt_path = os.path.join(output_dir, f"benchmark_{run_id}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Benchmark Run: {run_id}\n")
        f.write(f"Start Time: {run_data['started_at']}\n")
        f.write(f"End Time: {run_data['finished_at']}\n")
        f.write(f"Duration: {run_data['duration_seconds']:.2f}s\n")
        f.write(f"Success: {run_data['success_count']}, Failure: {run_data['error_count']}\n")
        f.write("=" * 70 + "\n")

        for idx, item in enumerate(run_data["results"], 1):
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"[{idx:02d}] {item['tag']}\n")
            f.write(f"Prompt: {item['prompt']}\n")
            if item["ok"]:
                f.write("Response:\n")
                f.write(item["response"] + "\n")
            else:
                f.write(f"Error: {item['error']}\n")

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/checkpoints_hybrid_v2/epoch_2.pt")
    parser.add_argument("--model_path", type=str,
                        default="PATH_TO_PRETRAINED_MODEL")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "musa", "npu", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max_memory_per_device", type=str, default="20GiB")
    parser.add_argument("--save_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)
    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint)

    print(f"Model Directory: {resolved_model_path}")
    print(f"Checkpoint: {resolved_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(
        resolved_checkpoint,
        resolved_model_path,
        device=args.device,
        dtype=args.dtype,
        max_memory_per_device=args.max_memory_per_device,
    )

    test_cases = [
        # (Category, Prompt, Expected Length)
        ("Code-Simple", "Write a Python function to compute the greatest common divisor of two numbers.", 256),
        ("Code-Complex", "Implement an LRU Cache class in Python with O(1) get and put.", 512),
        ("Math-Simple", "If 3x + 5 = 20, what is x? Please provide the steps.", 256),
        ("Math-WordProblem", "A cage contains chickens and rabbits. There are 35 heads and 94 legs. How many chickens and rabbits are there?", 256),
        ("Commonsense", "What is the boiling point of water at sea level in Celsius?", 128),
        ("Logic", "If all A are B, and all B are C, are all A C? Why?", 256),
        ("Chinese-Open", "Please introduce yourself.", 256),
        ("Chinese-Instruction", "Summarize the application scenarios of machine learning in 3 points, each point under 15 words.", 256),
        ("English-Open", "Explain quantum computing in simple terms.", 256),
        ("Roleplay", "You are a senior Python mentor. A student asks: Why does Python's GIL affect multi-threading performance? Please explain in simple terms.", 512),
        ("Multi-turn", "User: How is the weather in Beijing today?\nAssistant: I cannot access real-time weather data.\nUser: What about Shanghai?", 256),
    ]

    print("\n" + "=" * 70)
    print("Starting comprehensive benchmark")
    print("=" * 70)

    t0 = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_items = []
    success_count = 0
    error_count = 0

    for i, (tag, prompt, max_tokens) in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i:02d}] [{tag}]")
        print(f"Prompt: {prompt}")
        print("Response:")
        try:
            resp = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
            )
            result_items.append({
                "tag": tag,
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "ok": True,
                "response": resp,
            })
            success_count += 1
            # Limit output length to avoid flooding console
            lines = resp.splitlines()
            display = "\n".join(lines[:20])
            if len(lines) > 20:
                display += f"\n... ({len(lines)-20} lines omitted)"
            print(display)
        except Exception as e:
            print(f"[Error] {e}")
            result_items.append({
                "tag": tag,
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "ok": False,
                "error": str(e),
            })
            error_count += 1

    finished_at = datetime.now().isoformat(timespec="seconds")
    duration_seconds = time.time() - t0
    run_data = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "checkpoint": resolved_checkpoint,
        "model_path": resolved_model_path,
        "device": args.device,
        "dtype": args.dtype,
        "decode": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "num_cases": len(test_cases),
        "success_count": success_count,
        "error_count": error_count,
        "results": result_items,
    }
    json_path, txt_path = save_benchmark_results(args.save_dir, run_data)

    print("\n" + "=" * 70)
    print("Benchmark completed")
    print(f"Results saved to: {json_path}")
    print(f"Results saved to: {txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
