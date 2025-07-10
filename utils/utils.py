import time
import os
from functools import wraps

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

