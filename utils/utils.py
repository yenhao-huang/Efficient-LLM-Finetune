import time
import torch
from transformers import TrainerCallback
import os
from functools import wraps

class CUDAMemoryTrackerCallback(TrainerCallback):
    def __init__(self, log_every=100):
        self.log_every = log_every
        self.memory_log = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0:
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            self.memory_log.append(peak_mem)
            print(f"[Step {state.global_step}] Peak CUDA memory: {peak_mem:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, args, state, control, **kwargs):
        if self.memory_log:
            avg_peak = sum(self.memory_log) / len(self.memory_log)
            print(f"\nAverage peak CUDA memory: {avg_peak:.2f} MB over {len(self.memory_log)} samples")

def make_dir_with_timestamp(base_output_dir, base_log_dir):
    timestamp = int(time.time())
    output_dir = os.path.join(base_output_dir, str(timestamp))
    log_dir = os.path.join(base_log_dir, str(timestamp))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return output_dir, log_dir
    
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] took {end - start:.4f} seconds")
        return result
    return wrapper

def get_memory_snapshot(snapshot_path="memory_profile.pkl"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("Starting GPU memory trace...")
            torch.cuda.memory._record_memory_history(max_entries=100000)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                print("Saving memory snapshot to:", snapshot_path)
                torch.cuda.memory._dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                print("Memory tracing done.")
        return wrapper
    return decorator