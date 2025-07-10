import csv
import os

def save_predictions_to_csv(eval_agent, test_dataset, tokenizer, output_dir="predicts"):
    predictions = eval_agent.predict(test_dataset)
    preds = predictions.predictions.argmax(axis=-1)
    labels = predictions.label_ids

    input_ids = test_dataset["input_ids"]
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions.csv")

    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction_label", "ground_truth_label", "text"])  # ✅ 調整欄位順序
        for text, pred, label in zip(texts, preds, labels):
            writer.writerow([pred, label, text]) 

    print(f"Saved predictions to: {csv_path}")