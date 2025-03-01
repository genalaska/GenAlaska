import pandas as pd

file = "original_data.csv"  # Change to your dataset path
df = pd.read_csv(file, sep=',', header=0)
labels = list(set(df['label']))

# convert label to id
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['label'].map(label2id)

# save to new file
df.to_csv("data.csv", index=False)

# Save label2id as a csv file

label2id = pd.DataFrame(label2id.items(), columns=['label', 'id'])
label2id.to_csv("label2id.csv", index=False)
