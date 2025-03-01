import warnings

warnings.simplefilter("ignore", category=FutureWarning)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
from datasets import Dataset
from transformers import (XLMRobertaTokenizer,
                          XLMRobertaForSequenceClassification,
                          Trainer, TrainingArguments,
                          DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=',', header=0, dtype={'text': str, 'label': int})
    return df


# Load tokenizer
# tokenizer = XLMRobertaTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

# Load dataset
file_path = "data.csv"  # Change to your dataset path
df = load_dataset(file_path)

# Ensure stratified split
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)


# Load model from disk before inference
def load_model():
    return XLMRobertaForSequenceClassification.from_pretrained("./xlm_roberta_model")

#
model = load_model()
tokenizer = XLMRobertaTokenizer.from_pretrained("./xlm_roberta_model")

model.eval()

# Tokenization
def f_tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)


train_dataset = train_dataset.map(f_tokenize_function, batched=True)
test_dataset = test_dataset.map(f_tokenize_function, batched=True)

# Load model
num_labels = len(set(df['label']))
model = XLMRobertaForSequenceClassification.from_pretrained(
    'FacebookAI/xlm-roberta-base',
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)
model.config.return_dict = True

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=1,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# Train model
# trainer.train()

# # Save trained model
# model.save_pretrained("./xlm_roberta_model")
# tokenizer.save_pretrained("./xlm_roberta_model")

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def zero_mask_head(model, head_idx):
    hidden_dim = model.config.hidden_size  # e.g., 768
    num_heads = model.config.num_attention_heads  # e.g., 12
    head_dim = hidden_dim // num_heads

    # Loop through each layer in roberta encoder
    for layer in model.roberta.encoder.layer:
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim

        layer.attention.self.query.weight.data[start:end, :] = 0.0
        if layer.attention.self.query.bias is not None:
            layer.attention.self.query.bias.data[start:end] = 0.0
        layer.attention.self.key.weight.data[start:end, :] = 0.0
        if layer.attention.self.key.bias is not None:
            layer.attention.self.key.bias.data[start:end] = 0.0
        layer.attention.self.value.weight.data[start:end, :] = 0.0
        if layer.attention.self.value.bias is not None:
            layer.attention.self.value.bias.data[start:end] = 0.0
    return model

# Evaluation function
def predict_and_evaluate(trainer, test_dataset, head_list=None):
    if head_list:
        for head_idx in head_list:
            print(f"Evaluating with head {head_idx} masked...")
            fresh_model = load_model()  # Reload the original model for each head
            masked_model = zero_mask_head(fresh_model, head_idx)
            trainer.model = masked_model
            trainer.model.to(device)  # Ensure model is on the correct device

            with torch.no_grad():
                predictions = trainer.predict(test_dataset)
                logits = predictions.predictions
                predicted_labels = logits.argmax(axis=-1)
                true_labels = test_dataset["label"]

                cm = confusion_matrix(true_labels, predicted_labels, labels=range(20))
                total_samples = np.sum(cm)

                for cls in range(20):
                    TP = cm[cls, cls]
                    FP = cm[:, cls].sum() - TP
                    FN = cm[cls, :].sum() - TP
                    TN = total_samples - TP - FP - FN

                    # Accuracy for class `cls`
                    acc = (TP + TN) / total_samples if total_samples else 0.0

                    # Precision for class `cls`
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

                    # Recall for class `cls`
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                    # F1 score for class `cls`
                    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

                    print(
                        f"Class {cls}: Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

                # Metrics
                logits_tensor = torch.from_numpy(logits).float().to(device)
                true_labels_tensor = torch.from_numpy(np.array(true_labels)).long().to(device)
                loss = F.cross_entropy(logits_tensor, true_labels_tensor)
                print(f"Cross-Entropy Loss: {loss.item():.4f}")

                accuracy = accuracy_score(true_labels, predicted_labels)
                f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
                f1_micro = f1_score(true_labels, predicted_labels, average='micro')
                f1_macro = f1_score(true_labels, predicted_labels, average='macro')

                print(f"Head {head_idx} Masked - "
                      f"Accuracy: {accuracy:.4f}, "
                      f"Weighted F1: {f1_weighted:.4f}, " 
                      f"Micro F1: {f1_micro:.4f}, "
                      f"Macro F1: {f1_macro:.4f}\n")

    else:
        # Evaluate the trained model without any masking
        print("Evaluating the trained model without masking...")
        trainer.model = load_model()
        trainer.model.to(device)

        with torch.no_grad():
            predictions = trainer.predict(test_dataset)
            logits = predictions.predictions
            predicted_labels = logits.argmax(axis=-1)
            true_labels = test_dataset["label"]

            # Metrics
            logits_tensor = torch.from_numpy(logits).float().to(device)
            true_labels_tensor = torch.from_numpy(np.array(true_labels)).long().to(device)
            loss = F.cross_entropy(logits_tensor, true_labels_tensor)
            print(f"Cross-Entropy Loss: {loss.item():.4f}")

            accuracy = accuracy_score(true_labels, predicted_labels)
            f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
            f1_micro = f1_score(true_labels, predicted_labels, average='micro')
            f1_macro = f1_score(true_labels, predicted_labels, average='macro')

            print(
                f"No Head Masking - "
                f"Accuracy: {accuracy:.4f}, "
                f"Weighted F1: {f1_weighted:.4f}, "
                f"Micro F1: {f1_micro:.4f}, "
                f"Macro F1: {f1_macro:.4f}\n"
            )

predict_and_evaluate(trainer, test_dataset, head_list=None)

# Example: Apply masking one head at a time in all layers during evaluation
head_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Masking one head at a time from the list

predict_and_evaluate(trainer, test_dataset, head_list=head_list)
