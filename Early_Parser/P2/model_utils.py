from algorithm import ArcEager

def map_transitions_to_int(transitions):
    # Extract only the first two parts (ignoring the dependency)
    transition_prefixes = ['-'.join(str(transition).split('-')[:2]) for transition in transitions]
   
    # Get unique transitions
    unique_transitions = sorted(set(transition_prefixes))
   
    # Create mapping from transition to ID
    transition_to_id = {transition: idx for idx, transition in enumerate(unique_transitions)}
    
    # Create inverse mapping from ID to transition
    id_to_transition = {idx: transition for transition, idx in transition_to_id.items()}
   
    # Convert transitions to their corresponding integer representation
    transition_ids = [transition_to_id['-'.join(str(transition).split('-')[:2])] for transition in transitions]
   
    return transition_ids, transition_to_id, id_to_transition


def is_transition_valid(arc_eager,state, transition):
    
    print("initial transition",transition)
    if transition == 'LEFT-ARC':  # Left-Arc
        return arc_eager.LA_is_valid(state)
    
    if transition == 'RIGHT-ARC':  # Right-Arc
        return arc_eager.RA_is_valid(state)
    
    if transition == 'REDUCE':  # Reduce
        return arc_eager.REDUCE_is_valid(state)
    
    if transition == 'SHIFT':  # Shift
        return bool(state.B)  
    
    return False  


def select_valid_transition(batch_actions, arc_eager, initial_state, transition, position,train_id_to_transitions):
    while True:
        if is_transition_valid(arc_eager, initial_state, transition):
            break
        else:
            print("Transition is not valid, select the next")
            batch_actions[0, position] = -float('inf')  
            position = batch_actions.argmax(axis=1)[0]  
            transition = train_id_to_transitions.get(position, None) 
            
            if transition is None:
                raise ValueError("No valid transition find")
    
    return transition