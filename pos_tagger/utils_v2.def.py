import tensorflow
import numpy as np
from tensorflow.keras.layers import TextVectorization, InputLayer, Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Bidirectional,  Masking
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.utils import to_categorical
import requests

from Tokenino import Token

def download_file(url, file_path):
  """
  Download a file from a given URL and save it to a specified file path.

  :param url: The URL of the file to download.
  :param file_path: The local file path where the file will be saved (including the file name).
  """
  try:
      # Send a GET request to the URL
      response = requests.get(url, stream=True)

      # Check if the request was successful (status code 200)
      if response.status_code == 200:
          # Open the file in write-binary mode and write the content in chunks
          with open(file_path, 'wb') as file:
              for chunk in response.iter_content(chunk_size=8192):
                  if chunk:  # Filter out keep-alive new chunks
                      file.write(chunk)
          print(f"File downloaded successfully and saved to {file_path}")
      else:
          print(f"Failed to download the file. Status code: {response.status_code}")
  except Exception as e:
      print(f"An error occurred: {e}")


def parse_conllu(file_path):
  """
  Parse a CoNLL-U formatted file into sentences and their respective tokens
  represented as Token objects.d.

  :param file_path: Path to the CoNLL-U file
  :return: A list of sentences where each sentence is a list of Token objects
  """
  sentences = []
  tokens = []

  with open(file_path, 'r', encoding='utf-8') as f:
      for line in f:
          line = line.strip()

          if line.startswith('#') or not line:  # Skip comment lines
              if tokens:  # If we have tokens, finish the current sentence
                  sentences.append(tokens)
                  tokens = []
              continue

          # Split each line of the dataset with the \t separator
          parts = line.split('\t')

          # Skip lines with incorrect format or non-numeric ID (e.g. '1-2', '2.1')
          if len(parts) != 10 or not parts[0].isdigit():
              continue

          # Extract token information
          ID = int(parts[0])  # Token ID
          form = parts[1]  # Surface form
          lemma = parts[2]  # Lemma
          upos = parts[3]  # Universal POS tag
          xpos = parts[4]  # Language-specific POS tag
          feats = parts[5]  # Morphological features
          head = int(parts[6]) if parts[6].isdigit() else 0  # Head of the token
          deprel = parts[7]  # Dependency relation
          deps = parts[8]  # Enhanced dependency relations
          misc = parts[9]  # Miscellaneous info

          # Create a Token object
          token = Token(ID, form, lemma, upos, xpos, feats, head, deprel, deps, misc)

          # Add the token to the current sentence
          tokens.append(token)

  if tokens:  # Ensure the last sentence is added
      sentences.append(tokens)

  return sentences

def upos_vectorizer(upos_tags, max_length):
  """
  Convert a list of UPOS tags to a padded sequence of IDs.
  Args:
    upos_tags: A list of UPOS tags.
    max_length: The maximum length of the output sequence.
  Returns:
    A padded sequence of IDs.
  """

  # Define UPOS to ID mapping
  upos_to_id = {
      "ADJ": 0,    # adjective
      "ADP": 1,    # adposition
      "ADV": 2,    # adverb
      "AUX": 3,    # auxiliary
      "CCONJ": 4,  # coordinating conjunction
      "DET": 5,    # determiner
      "INTJ": 6,   # interjection
      "NOUN": 7,   # noun
      "NUM": 8,    # numeral
      "PART": 9,  # particle
      "PRON": 10,  # pronoun
      "PROPN": 11, # proper noun
      "PUNCT": 12, # punctuation
      "SCONJ": 13, # subordinating conjunction
      "SYM": 14,   # symbol
      "VERB": 15,  # verb
      "X": 16      # other
  }

  # Convert UPOS tags to IDs
  ids = []
  for upos_list in upos_tags:
    ids_aux = [upos_to_id.get(upos) for upos in upos_list]  # 0 is used for unknown UPOS tags

    # Padding with 0's on the right to the max_length
    padded_ids = pad_sequences([ids_aux], maxlen=max_length, padding='post', value=0)[0]
    ids.append(padded_ids)

  return ids

def build_word_vectorizer(sentences, max_length):
  """
  Builds the TextVectorizer layer for the given sentences.

  Args:
    sentences: A list of sentences.
    max_length: The maximum length of the sentences.

  Returns:
    A TextVectorizer layer.
  """

  # list of lists containing all the sentences so we can build the vocabulary
  forms = []  # To store the 'form' attributes for each sentence

  # Iterate over each sentence in 'sentences'
  for sentence in sentences:
      # Extract 'form' (input)
      input_sentence = [token.form for token in sentence]

      # Add them to the respective lists
      forms.append(input_sentence)


  # Define TextVectorization layer for words
  text_vectorizer = TextVectorization(
      output_mode='int',
      output_sequence_length=max_length,
      standardize=None
  )

  # Adapt the vectorization layer for words
  all_forms = [token for sentence in forms for token in sentence]
  text_vectorizer.adapt(all_forms)

  return text_vectorizer


def prepare_data(sentences, text_vectorizer, max_length):
  """Prepares the input and output data for the POS tagging model.

  This function takes a list of sentences, a text vectorizer, and the maximum
  sequence length as input. It performs the following steps:

  1. Extracts the forms (words) and upos (part-of-speech tags) from the sentences.
  2. Vectorizes the forms using the provided text vectorizer.
  3. Vectorizes the upos and creates a unique mapping for each tag.
  4. Creates one-hot encoding for the output upos tags.

  Args:
    sentences: A list of sentences, where each sentence is a list of Token objects.
    text_vectorizer: A TextVectorization layer used to vectorize the input forms.
    max_length: The maximum sequence length for padding.

  Returns:
    A tuple containing the vectorized input and the corresponding one-hot encoded output.
  """
  # Join the tokens.forms of a sentences back into a string separated by spaces.
  forms = [' '.join([token.form for token in sentence]) for sentence in sentences]

  # Extract the upos for each token in each sentence.
  upos = [[token.upos for token in sentence] for sentence in sentences]

  # Vectorize forms and upos
  vectorized_input = text_vectorizer(forms)
  #vectorized_output, unique_upos_ids = upos_vectorizer(upos, max_length)
  vectorized_output  = upos_vectorizer(upos, max_length)
  
  # Create one hot encoding for the output
  categorical_output = to_categorical(np.array(vectorized_output))

  return vectorized_input, categorical_output
  





