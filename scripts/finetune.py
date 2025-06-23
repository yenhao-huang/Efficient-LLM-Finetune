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
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir, log_dir = utils.make_dir_with_timestamp(args.output_dir, args.log_dir)
    raw_data = data_utils.data_loading(args.dataset)
    tonkenized_data, tokenizer = data_utils.data_preprocess(raw_data, args.model, args.text_col, args.label_col)
    train_dataset, eval_dataset, _ = data_utils.select_partial_data(tonkenized_data, 10000, 1000)
    train_agent = model_utils.set_train_agent_PEFT(
        args.model, 
        train_dataset, 
        eval_dataset, 
        tokenizer, 
        args.n_labels, 
        metric_utils.compute_metrics,
        output_dir,
        args.label_col,
    )
    import torch
    print(torch.cuda.memory_summary(device=0))
    try:
        train_agent.train()
    except torch.cuda.OutOfMemoryError as e:
        print("❌ CUDA OOM 發生！")
        print("=== CUDA Memory Summary ===")
        print(torch.cuda.memory_summary())
        raise e