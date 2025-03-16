import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input ,TextVectorization, Dense, Embedding,Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TextVectorization
from data_utils import create_inputs
from data_utils import extract_dependency
from data_utils import create_dataset_testing
from model_utils import map_transitions_to_int
from model_utils import select_valid_transition
from algorithm import Sample
from algorithm import Transition


class ParserMLP:
    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64,
                 epochs: int = 1, batch_size: int = 64,
                 vocab_size: int = 10000, num_labels_transitions: int = 5,
                 num_labels_dependencies: int = 44, max_sequence_length: int = 10):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.
       
        Parameters:
            max_sequence_length (int): Maximum length of input sequences
        """
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_labels_transitions = num_labels_transitions
        self.num_labels_dependencies = num_labels_dependencies
        self.max_sequence_length = max_sequence_length
       
        # Build the model
        self.model = self.build_model()

    def build_model(self):
        word_input = Input(shape=(self.max_sequence_length,), dtype=tf.int32, name="word_input")
       
        # Embedding layer
        word_embedding = Embedding(
            self.vocab_size,
            self.word_emb_dim,
            input_length=self.max_sequence_length,
            mask_zero=True
        )(word_input)
       
        # Flatten layer
        flattened = Flatten()(word_embedding)
       
        # Hidden layers 
        hidden1 = Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(0.001)
        )(flattened)
       
        hidden2 = Dense(
            64,
            activation='relu',
            kernel_regularizer=l2(0.001)
        )(hidden1)
       
        # Dropout 
        dropout = Dropout(0.3)(hidden2)
       
        # Output layers
        transition_output = Dense(
            self.num_labels_transitions,
            activation="softmax",
            name="transition_output"
        )(dropout)
       
        dependency_output = Dense(
            self.num_labels_dependencies,
            activation="softmax",
            name="dependency_output"
        )(dropout)
       
        model = Model(inputs=word_input, outputs=[transition_output, dependency_output])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'transition_output': 'categorical_crossentropy',
                'dependency_output': 'categorical_crossentropy'
            },
            loss_weights={
                'transition_output': 1.0,
                'dependency_output': 1.0
            },
            metrics={
                'transition_output': 'accuracy',
                'dependency_output': 'accuracy'
            }
        )
        return model

    def preprocess_samples(self, train_samples, dev_samples, arc_eager_instance, max_tokens=5000, max_len=8):
        """
        Preprocess training and development samples for ArcEager parser.

        Args:
            self: Instance of the class (ParserMLP).
            train_samples (list): Training samples.
            dev_samples (list): Development samples.
            arc_eager_instance: An instance of the ArcEager parser.
            max_tokens (int): Maximum number of tokens for vectorization.
            max_len (int): Maximum sequence length for vectorization.

        Returns:
            dict: A dictionary containing preprocessed data.
        """
        # Extract features and transitions from samples
        train_all_words, train_all_postags, train_all_transitions, train_all_dependencies, train_all_features = create_inputs(train_samples, extract_dependency)
        dev_all_words, dev_all_postags, dev_all_transitions, dev_all_dependencies, dev_all_features = create_inputs(dev_samples, extract_dependency)

        # Vectorize words
        word_vectorize_layer = TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            output_sequence_length=max_len
        )
        self.word_vectorize_layer = word_vectorize_layer

        grouped_train_features = [' '.join(features) for features in train_all_features]
        grouped_dev_features = [' '.join(features) for features in dev_all_features]

        word_vectorize_layer.adapt(grouped_train_features)

        vectorized_train_all_features = word_vectorize_layer(grouped_train_features)
        vectorized_dev_all_features = word_vectorize_layer(grouped_dev_features)
        self.vectorized_dev_all_features=vectorized_dev_all_features

        # Map transitions to integers
        train_transition_ids, train_transition_to_id, train_id_to_transitions = map_transitions_to_int(train_all_transitions)
        dev_transition_ids, dev_transition_to_id, dev_id_to_transitions = map_transitions_to_int(dev_all_transitions)
        self.train_id_to_transitions=train_id_to_transitions
        train_categorical_transitions = to_categorical(train_transition_ids, num_classes=len(train_transition_to_id))
        dev_categorical_transitions = to_categorical(dev_transition_ids, num_classes=len(dev_transition_to_id))
        self.dev_categorical_transitions=dev_categorical_transitions
        # Combine all dependencies from train and dev
        all_dependencies = set(train_all_dependencies + dev_all_dependencies)
        global_dependency_to_id = {dep: idx for idx, dep in enumerate(all_dependencies)}
        self.global_dependency_to_id=global_dependency_to_id 
        global_dependency_to_id

        train_dep_ids = [global_dependency_to_id[dep] for dep in train_all_dependencies]
        dev_dep_ids = [global_dependency_to_id[dep] for dep in dev_all_dependencies]

        train_categorical_dependencies = to_categorical(train_dep_ids, num_classes=len(global_dependency_to_id))
        dev_categorical_dependencies = to_categorical(dev_dep_ids, num_classes=len(global_dependency_to_id))
        self.dev_categorical_dependencies=dev_categorical_dependencies
        # Return processed data
        return {
            'vectorized_train_features': vectorized_train_all_features,
            'vectorized_dev_features': vectorized_dev_all_features,
            'train_categorical_transitions': train_categorical_transitions,
            'dev_categorical_transitions': dev_categorical_transitions,
            'train_categorical_dependencies': train_categorical_dependencies,
            'dev_categorical_dependencies': dev_categorical_dependencies,
            'word_vectorize_layer': word_vectorize_layer,
            'train_transition_to_id': train_transition_to_id,
            'dev_transition_to_id': dev_transition_to_id,
            'global_dependency_to_id': global_dependency_to_id
            }

    
    def train(self,train_samples,dev_samples,arc_eager):
            preprocessed_data =self.preprocess_samples(train_samples, dev_samples, arc_eager)

            # Extract Data for training
            vectorized_train_all_features = preprocessed_data['vectorized_train_features']
            train_categorical_transitions = preprocessed_data['train_categorical_transitions']
            train_categorical_dependencies = preprocessed_data['train_categorical_dependencies']

            # Extract Data for validation
            vectorized_dev_all_features = preprocessed_data['vectorized_dev_features']
            dev_categorical_transitions = preprocessed_data['dev_categorical_transitions']
            dev_categorical_dependencies = preprocessed_data['dev_categorical_dependencies']
            
            self.model.fit(
            vectorized_train_all_features,  
            [train_categorical_transitions, train_categorical_dependencies],  
            epochs=10,
            batch_size=64,
            validation_data=(
                vectorized_dev_all_features,  
                [dev_categorical_transitions, dev_categorical_dependencies]  
            )
        )
    def evaluate(self):
           
            results = self.model.evaluate(self.vectorized_dev_all_features, [self.dev_categorical_transitions, self.dev_categorical_dependencies])

            print("Evaluation Results:")
            print(f"Total Loss: {results[0]}")
            print(f"Transition Loss: {results[1]}")
            print(f"Dependency Loss: {results[2]}")
            print(f"Transition Accuracy: {results[3]}")
            print(f"Dependency Accuracy: {results[4]}")

            return results

    
    def run(self,sentences,arc_eager):
        final_states = []

        for idx, sent in enumerate(sentences):
            print(f"\n=== Parsing of the sentence {idx + 1} ===")
            
            # Create the initial state for the sentence
            current_state = arc_eager.create_initial_state(sent)
            
            # Lists for samples and transitions
            samples = []
            transition = None
            
            # Add the initial sample
            samples.append(Sample(current_state, transition))
            
            all_test_input_features = []
            all_test_transitions = []
            all_test_dependency = []
            
            # Iterazione per ogni passo
            step = 0
            while current_state.B:
                print(f"\n--- Step {step + 1} ---")
                print(f"--- State BEFORE Transition (sentence {idx + 1}) ---")
                print(current_state)
                
                # Crea il campione corrente
                current_sample = samples[-1]
                
                # Genera features di test
                test_input_features = create_dataset_testing(current_sample, arc_eager)
                print('CURRENT FEATURES:', test_input_features)
                all_test_input_features.append(test_input_features)
                
                # Prepara le features per la vettorizzazione
                grouped_features = [' '.join(map(str, features)) for features in test_input_features]
                vectorized_features = self.word_vectorize_layer(grouped_features)
                
                # Prediction
                batch_actions, batch_deprels = self.model.predict(vectorized_features)
                
                # Get transition and dependency indeces
                transition_position = batch_actions.argmax(axis=1)[0]
                dependency_position = batch_deprels.argmax(axis=1)[0]
                
                transizione = self.train_id_to_transitions.get(transition_position)
               
                # Check if the transitionn is valid
                selected_transition = select_valid_transition(
                    batch_actions,  
                    arc_eager, 
                    current_state, 
                    transizione,
                    transition_position,
                    self.train_id_to_transitions
                )
                all_test_transitions.append(selected_transition)
                
                # Get the dependency
                print("dependency_position", dependency_position)
                # Invert the global map
                id_to_global_dependency = {idx: dep for dep, idx in self.global_dependency_to_id.items()}
                dependency = id_to_global_dependency[dependency_position]
                
                all_test_dependency.append(dependency)
                
                # Apply the transition
                transition_obj = Transition(selected_transition, dependency)
                arc_eager.apply_transition(current_state, transition_obj)
                
                #update samples and state
                transition = selected_transition
                samples.append(Sample(current_state, transition))
                
                #step increment
                step += 1
            
            # Save the final state
            final_states.append(current_state)
            # Print all the final states
            print("\n=== ALL FINAL STATE ===")
            for idx, state in enumerate(final_states):
                print(f"\n Final state sentence {idx + 1}:")
                print(state)
        return final_states