import torch
from transformers import AutoTokenizer
from model import MultiTaskModel

def test_model(sentences):
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    num_classes_task_a = 3
    num_classes_task_b = 2
    model = MultiTaskModel(model_name, num_classes_task_a, num_classes_task_b)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

    model.eval()
    with torch.no_grad():
        logits_task_a, logits_task_b = model(input_ids, attention_mask)
    predictions_task_a = torch.argmax(logits_task_a, dim=1)
    predictions_task_b = torch.argmax(logits_task_b, dim=1)

    label_mapping_task_a = {0: "Positive", 1: "Negative", 2: "Neutral"}
    label_mapping_task_b = {0: "Negative", 1: "Positive"}

    for i, sentence in enumerate(sentences):
        print(f"Sentence: {sentence}")
        print(f"Prediction Task A (Classification): {label_mapping_task_a[predictions_task_a[i].item()]}")
        print(f"Prediction Task B (Sentiment Analysis): {label_mapping_task_b[predictions_task_b[i].item()]}\n")

if __name__ == "__main__":
    test_sentences = ["This movie was great!", "I am unhappy with the service.", "The book was boring."]
    test_model(test_sentences)
