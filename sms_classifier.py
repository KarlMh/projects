# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
  !pip install --upgrade tensorflow
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    return df

train_df = load_data(train_file_path)
test_df = load_data(test_file_path)


# Prepare data
def preprocess_data(df):
    texts = df['message'].values
    labels = df['label'].apply(lambda x: 1 if x == 'spam' else 0).values
    return texts, labels

train_texts, train_labels = preprocess_data(train_df)
test_texts, test_labels = preprocess_data(test_df)

# Tokenize and pad sequences
max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=max_len)
test_padded = pad_sequences(test_sequences, maxlen=max_len)


# Build model
model = keras.Sequential([
    keras.layers.Embedding(max_words, 128, input_length=max_len),
    keras.layers.LSTM(64, return_sequences=False),  # Changed return_sequences to False
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_padded, train_labels,
                    epochs=15,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    batch_size=32)

# Prediction on a new message
new_message = ["You have won a free ticket to Bahamas! Click here to claim."]
sequence = tokenizer.texts_to_sequences(new_message)
padded = pad_sequences(sequence, maxlen=max_len)
print(model.predict(padded))


# Predict function
def predict_message(pred_text):
    seq = tokenizer.texts_to_sequences([pred_text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    return [prediction, label]

# Test prediction
pred_text = "You have won"
prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
