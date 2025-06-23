from datasets import load_dataset
from transformers import AutoTokenizer

def data_loading(dataset_name):
    raw_data = load_dataset(dataset_name)
    return raw_data

# Output: DatasetDir: {"train":tokenized_train_data,"test":tokenized_test_data}
def data_preprocess(raw_data, model_name, text_col, label_col):
    tokenizer = AutoTokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)

    def tokenize(batch):
        texts = [x if isinstance(x, str) else "" for x in batch[text_col]]
        return tokenizer(texts, padding="max_length", truncation=True)

    tokenized_data = raw_data.map(tokenize, batched=True)
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", label_col])
    return tokenized_data, tokenizer

def select_partial_data(tonkenized_data, train_size=10000, eval_size=2000, seed=42):
    train_dataset = tonkenized_data["train"].shuffle(seed=seed).select(range(train_size))
    eval_dataset = tonkenized_data["test"].select(range(eval_size))
    test_dataset = tonkenized_data["test"].select(range(1000, 1000 + eval_size))
    return train_dataset, eval_dataset, test_dataset