import os
import sys

# 專案根目錄 = scripts 的上一層
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from utils import utils, data_utils, model_utils, metric_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--n_rank", type=int, default=8)
    parser.add_argument("--enable_lora", type=int, default=1)
    parser.add_argument("--bit_precision", type=int, default=4)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()

@utils.get_memory_snapshot(snapshot_path="metadata/memory_train.pkl")
def run_training():
    train_agent.train()

if __name__ == "__main__":
    args = parse_args()
    output_dir, log_dir = utils.make_dir_with_timestamp(args.output_dir, args.log_dir)
    raw_data = data_utils.data_loading(args.dataset)
    tonkenized_data, tokenizer = data_utils.data_preprocess(raw_data, args.model, args.text_col, args.label_col, args.seq_len)
    # For accuracy evaluation
    # train_dataset, eval_dataset, _ = data_utils.select_partial_data(tonkenized_data, 10000, 1000)
    # For memory usage testing
    train_dataset, eval_dataset, _ = data_utils.select_partial_data(tonkenized_data, 10, 1)

    train_agent = model_utils.set_train_agent_PEFT(
        args.model, 
        train_dataset, 
        eval_dataset, 
        tokenizer, 
        args.n_labels, 
        metric_utils.compute_metrics,
        output_dir,
        args.label_col,
        args.n_rank,
        args.bit_precision,
        args.enable_lora,
    )

    run_training()