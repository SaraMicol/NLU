import numpy as np
import requests
from matplotlib import pyplot as plt 

import tensorflow as tf
from tensorflow.keras.layers import Dropout,TextVectorization, InputLayer, Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Bidirectional,  Masking
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential


class Token:
    def __init__(self, ID, form, lemma, upos, xpos="_", feats="_", head=0, deprel="_", deps="_", misc="_"):
        """
        Initializes a CoNLL-U token with the following fields:
        :param ID: The token's index in the sentence (int or str for multi-word tokens).
        :param form: The surface form of the word (string).
        :param lemma: The lemma (base form) of the word (string).
        :param upos: Universal part-of-speech tag (string).
        :param xpos: Language-specific part-of-speech tag (string, optional, default "_").
        :param feats: Morphological features (string, optional, default "_").
        :param head: The index of the head token in the dependency relation (int, optional, default 0).
        :param deprel: Dependency relation to the head (string, optional, default "_").
        :param deps: Enhanced dependency relations (string, optional, default "_").
        :param misc: Additional information (string, optional, default "_").
        """
        self.ID = ID
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def to_conllu(self):
        """Returns the token as a CoNLL-U formatted string."""
        return f"{self.ID}\t{self.form}\t{self.lemma}\t{self.upos}\t{self.xpos}\t{self.feats}\t{self.head}\t{self.deprel}\t{self.deps}\t{self.misc}"

    def __repr__(self):
        """Returns a readable representation of the token."""
        return f"Token(ID={self.ID}, form={self.form}, lemma={self.lemma}, upos={self.upos}, xpos={self.xpos}, feats={self.feats}, head={self.head}, deprel={self.deprel}, deps={self.deps}, misc={self.misc})"


class MyTagger:

  def __init__(self):
    self.model = None
    self.history = None

  def build_model(self, max_length, vocab_size, embedding_dim, lstm_units, num_pos_tags):
    # Build the model
    model = tf.keras.models.Sequential()

    # 1. Input layer (implicit)
    # The input layer takes the input already vectorized
    model.add(InputLayer(input_shape=(max_length,)))

    # 2. Masking layer to ignore padding (0s)
    model.add(Masking(mask_value=0.0))  # Ignora gli zeri, cambia il valore se il tuo padding è diverso

    # 2. Embedding layer
    # This layer converts integer-encoded words into dense word embeddings.
    # input_dim = vocab_size specifies the size of the vocabulary.
    # output_dim = embedding_dim specifies the dimensionality of the embedding vectors.
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    
    model.add(Dropout(0.2))

    # 3. LSTM layer
    # This LSTM layer processes the sequence of embeddings word by word.
    # return_sequences=True ensures that the LSTM returns an output for each word in the sequence.
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True, input_length= max_length)))

    # 4. TimeDistributed wrapper for the Dense layer
    # The TimeDistributed layer applies the Dense layer to each timestep (word) of the LSTM output.
    # Dense layer output has num_pos_tags units with a softmax activation to predict the PoS tag for each word.
    model.add(TimeDistributed(Dense(units=num_pos_tags, activation='softmax')))

    # Compile the model
    # 'adam' optimizer is used for training.
    # 'categorical_crossentropy' is the loss function because we are performing multi-class classification.
    # The accuracy metric is used to evaluate the model performance.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    self.model = model

  def train_model(self, train_inputs, train_outputs, val_inputs, val_outputs, epochs, batch_size):
    if self.model is not None:
      if val_inputs is None and val_outputs is None:
        self.history = self.model.fit(train_inputs, train_outputs, epochs=epochs, batch_size=batch_size, validation_split = 0.2)
      else:
        self.history = self.model.fit(train_inputs, train_outputs, epochs=epochs, batch_size=batch_size, validation_data=(val_inputs, val_outputs))
    else:
      print("You must create the model first. See build_model() function.")

  def predict(self, test_input):
    if self.model is not None:
      return self.model.predict(test_input)
  
  def model_evaluate(self, test_input, test_output, batch_size):
    if self.model is not None:
      return self.model.evaluate(test_input, test_output, batch_size)

  def plot_history(self):
    if self.history is not None:
      plt.plot(self.history.history['accuracy'])
      plt.plot(self.history.history['val_accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()

