from algorithm import Sample
from conllu_token import Token
print("Sample imported successfully:", Sample)


def read_file(reader, path, inference):
    """
    Reads a file using the specified reader.

    Args:
        reader: The reader object to use for parsing.
        path: The path to the file to read.
        inference: Whether the file is for inference.

    Returns:
        trees: A list of dependency trees read from the file.
    """
    trees = reader.read_conllu_file(path, inference)
    return trees


def create_inputs(samples, extract_dependency):
    """
    Prepares input features and labels for the parser from a list of samples.

    Args:
        samples: List of samples, where each sample contains state and transition information.
        extract_dependency: Function to extract dependency labels from transitions.

    Returns:
        tuple: Contains the words, POS tags, transitions, and dependencies.
    """
    all_words = []
    all_postags = []
    all_transitions = []
    all_dependencies = []
    all_input_features=[]
   
    for sample in samples:
        extractor = Sample(sample.state, sample.transition)

        # Extract features
        features = extractor.state_to_feats(nbuffer_feats=2, nstack_feats=2)
        n_half = len(features) // 2
        words = features[:n_half]
        pos_tags = features[n_half:]

        all_words.append(words)
        all_postags.append(pos_tags)
        input_features = words + pos_tags
        all_input_features.append(input_features)

        # Extract transitions and dependencies
        output_transition = sample.transition
        all_transitions.append(output_transition)

        dependencies = extract_dependency(str(output_transition))
        all_dependencies.append(dependencies)

    return all_words, all_postags, all_transitions, all_dependencies,all_input_features



def extract_dependency(transition_str):
    """
    Extracts the dependency label from a transition string.

    Args:
        transition_str: The transition string in the format "ACTION-SUBTYPE-LABEL".

    Returns:
        str: The dependency label (third part of the string) or None if not found.
    """
    parts = transition_str.split('-')
    if len(parts) >= 3:
        return parts[2]  # Return the third part (dependency label)
    return None  # Return None if the format is unexpected

def create_dataset_testing(sample, arc_eager_instance, nbuffer_feats=2, nstack_feats=2):
    """
    Creates a testing dataset using only words and universal POS tags.

    Args:
        trees: List of dependency trees.
        arc_eager_instance: An instance of the ArcEager parser.
        nbuffer_feats: Number of features to consider from the buffer.
        nstack_feats: Number of features to consider from the stack.

    Returns:
        tuple: Contains the dataset, words, POS tags, and input features.
    """
    dataset = []
    all_words = []
    all_upos_tags = []
    all_input_features = []
    
    extractor = Sample(sample.state, sample.transition)
    features = extractor.state_to_feats(nbuffer_feats=nbuffer_feats, nstack_feats=nstack_feats)
    
    n_half = len(features) // 2
    words = features[:n_half]
    pos_tags = features[n_half:]
    input_features = words + pos_tags
    all_input_features.append(input_features)

    return all_input_features

def update_trees(trees, final_states):
    updated_trees = []
    
    for idx, (tree, final_state) in enumerate(zip(trees, final_states)):
        #Takes the arcs for the final state
        all_arcs = final_state.A
        
        updated_tokens = []
        
        for token in tree:
            matching_arc = None
            for arc in all_arcs:
                if arc[2] == token.id:
                    matching_arc = arc
                    break
            if matching_arc:
                head = matching_arc[0]
                deprel = matching_arc[1]
                
                # Update the Token
                updated_token = Token(
                    token.id, 
                    token.form, 
                    token.lemma, 
                    token.upos, 
                    token.cpos, 
                    token.feats, 
                    head, 
                    deprel  
                )
                updated_tokens.append(updated_token)
            else:
                updated_tokens.append(token)
        
        updated_trees.append(updated_tokens)
    
    return updated_trees
