#!/usr/bin/env python3
"""Benchmark Whisper large-v3-turbo on CPU, GPU, and NPU with OpenVINO."""

import time
import sys
import os
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import openvino_genai as ogai


def generate_test_audio(duration_sec=10, sample_rate=16000):
    """Generate synthetic test audio (speech-like noise)."""
    samples = int(duration_sec * sample_rate)
    # Generate pink noise (more speech-like than white noise)
    np.random.seed(42)
    audio = np.random.randn(samples).astype(np.float32) * 0.1
    # Add some periodic structure
    t = np.linspace(0, duration_sec, samples)
    audio += 0.05 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


def benchmark_model(model_path, device, audio, num_runs=3, is_npu=False):
    """Benchmark a single model configuration."""
    result = {
        "model": Path(model_path).name,
        "device": device,
        "warmup_ms": None,
        "avg_ms": None,
        "rtf": None,
        "success": False,
        "error": None,
    }

    try:
        # Load model
        load_start = time.time()
        if is_npu:
            pipe = ogai.WhisperPipeline(model_path, device, STATIC_PIPELINE=True)
        else:
            pipe = ogai.WhisperPipeline(model_path, device)
        load_time = (time.time() - load_start) * 1000
        result["load_ms"] = load_time

        # Warmup run
        warmup_start = time.time()
        result_obj = pipe.generate(audio, return_timestamps=True)
        warmup_ms = (time.time() - warmup_start) * 1000
        result["warmup_ms"] = warmup_ms

        # Benchmark runs
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            result_obj = pipe.generate(audio, return_timestamps=True)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        avg_ms = np.mean(latencies)
        std_ms = np.std(latencies)
        rtf = avg_ms / (len(audio) / 16000 * 1000)  # Real-time factor

        result["avg_ms"] = avg_ms
        result["std_ms"] = std_ms
        result["rtf"] = rtf
        result["success"] = True

        # Get transcription sample
        if hasattr(result_obj, "chunks") and result_obj.chunks:
            result["sample_text"] = result_obj.chunks[0].text[:100]
        else:
            result["sample_text"] = str(result_obj)[:100]

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("=" * 70)
    print("Whisper large-v3-turbo Benchmark")
    import openvino

    print("OpenVINO", openvino.__version__)
    print("=" * 70)

    # Model paths
    models = {
        "FP16": "/home/aubrey/Desktop/librevoice/models/whisper-large-v3-turbo-fp16",
        "INT8": "/home/aubrey/Desktop/librevoice/models/whisper-large-v3-turbo-int8",
    }

    # Devices to test
    # NOTE: NPU is NOT compatible with whisper-large-v3-turbo (known issue #1965)
    # The turbo model's 4-layer decoder causes self_attn_nodes error with STATIC_PIPELINE
    # and hangs without it. NPU works with standard whisper models (tiny/base/small/large)
    # but NOT turbo. Using CPU and GPU only.
    devices = ["CPU", "GPU"]

    # Generate test audio (10 seconds)
    print("\nGenerating 10-second test audio...")
    audio = generate_test_audio(duration_sec=10)
    print(f"Audio shape: {audio.shape}, duration: {len(audio) / 16000:.1f}s")

    # Run benchmarks
    results = []
    total_combos = len(models) * len(devices)
    current = 0

    for precision, model_path in models.items():
        for device in devices:
            current += 1
            is_npu = device == "NPU"
            print(
                f"\n[{current}/{total_combos}] Benchmarking {precision} on {device}..."
            )

            result = benchmark_model(
                model_path, device, audio, num_runs=3, is_npu=is_npu
            )
            result["precision"] = precision
            results.append(result)

            if result["success"]:
                print(f"  Load: {result['load_ms']:.0f}ms")
                print(f"  Warmup: {result['warmup_ms']:.0f}ms")
                print(
                    f"  Average: {result['avg_ms']:.0f}ms (+/- {result['std_ms']:.0f}ms)"
                )
                print(
                    f"  RTF: {result['rtf']:.2f}x ({'faster' if result['rtf'] < 1 else 'slower'} than real-time)"
                )
                print(f"  Sample: {result['sample_text'][:80]}...")
            else:
                print(f"  ERROR: {result['error']}")

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'Precision':<10} {'Device':<8} {'Load(ms)':<10} {'Warmup(ms)':<12} {'Avg(ms)':<10} {'RTF':<8} {'Status'}"
    )
    print("-" * 70)

    best_rtf = float("inf")
    best_config = None

    for r in results:
        if r["success"]:
            status = "OK"
            if r["rtf"] < best_rtf:
                best_rtf = r["rtf"]
                best_config = r
            print(
                f"{r['precision']:<10} {r['device']:<8} {r['load_ms']:<10.0f} {r['warmup_ms']:<12.0f} {r['avg_ms']:<10.0f} {r['rtf']:<8.2f} {status}"
            )
        else:
            print(
                f"{r['precision']:<10} {r['device']:<8} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<8} FAILED: {r['error'][:40]}"
            )

    if best_config:
        print("\n" + "=" * 70)
        print(f"RECOMMENDED CONFIGURATION:")
        print(f"  Model: {best_config['precision']}")
        print(f"  Device: {best_config['device']}")
        print(f"  RTF: {best_config['rtf']:.2f}x")
        print(f"  Average latency: {best_config['avg_ms']:.0f}ms for 10s audio")
        print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
