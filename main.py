from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from keras.optimizers import Adam


def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["sentence"])


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]
dataset = dataset.map(tokenize_dataset)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
model.compile(optimizer=Adam(3e-5))

tf_dataset = model.prepare_tf_dataset(dataset, batch_size=16, shuffle=True, tokenizer=tokenizer)
model.fit(tf_dataset)
x = 5
