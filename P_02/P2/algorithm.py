from state import State
from conllu_token import Token

class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.
    
    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.

    Attributes:
        state (State): An instance of the State class, representing the current parsing 
                       state at a given timestep in the parsing process.
        transition (Transition): An instance of the Transition class, representing the 
                                 parser action to be taken in the given state.

    Methods:
        state_to_feats(nbuffer_feats: int = 2, nstack_feats: int = 2): Extracts features from the parsing state.
    """

    def __init__(self, state: State, transition: Transition):
        """
        Initializes a new instance of the Sample class.

        Parameters:
            state (State): The current parsing state.
            transition (Transition): The transition action corresponding to the state.
        """
        self._state = state
        self._transition = transition

    @property
    def state(self):
        """
        Retrieves the current parsing state of the sample.

        Returns:
            State: The current parsing state in this sample.
        """
        return self._state


    @property
    def transition(self):
        """
        Retrieves the transition action of the sample.

        Returns:
            Transition: The transition action representing the parser's decision at this sample's state.
        """
        return self._transition
    

    
    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        features = []
        
        # Extract stack and buffer from self._state
        stack_items = self._state.S[-nstack_feats:] if len(self._state.S) > 0 else []
        buffer_items = self._state.B[:nbuffer_feats] if len(self._state.B) > 0 else []
        
        # Padding for stack
        while len(stack_items) < nstack_feats:
            stack_items.insert(0, Token(0, "<PAD>", "<PAD>", "_", "_", "_", "_", "_"))
        
        # Padding for buffer
        while len(buffer_items) < nbuffer_feats:
            buffer_items.append(Token(0, "<PAD>", "<PAD>", "_", "_", "_", "_", "_"))
        
        # Add words (form)
        for token in stack_items:
            features.append(token.form)
        for token in buffer_items:
            features.append(token.form)
        
        # Add POS tags
        for token in stack_items:
            features.append(token.upos)
        for token in buffer_items:
            features.append(token.upos)
        
        return features

def __str__(self):
        """
        Returns a string representation of the sample, including its state and transition.

        Returns:
            str: A string representing the state and transition of the sample.
        """
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"



class ArcEager:
    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        return State([sent[0]], sent[1:], set([]))

    def final_state(self, state: State) -> bool:
        return len(state.B) == 0

    def LA_is_valid(self, state: State) -> bool:
        top = state.S[-1]
        if top.id == 0:
            return False
        if any(arc[2] == top.id for arc in state.A):
            return False
        return True

    def LA_is_correct(self, state: State) -> bool:
        return (state.B[0].id == state.S[-1].head)
    
    def RA_is_valid(self, state: State) -> bool:
        if not state.B:
            return False
        buffer_first = state.B[0]
        if any(arc[2] == buffer_first.id for arc in state.A):
            return False
        return True
    def RA_is_correct(self, state: State) -> bool:
        return (state.S[-1].id == state.B[0].head)

    #we can remove it if we already have assigned an head to it
    
    def REDUCE_is_valid(self, state: State) -> bool:
        top = state.S[-1]
        return any(arc[2] == top.id for arc in state.A)

    def REDUCE_is_correct(self, state: State) -> bool:
        top = state.S[-1]  # Elemento in cima alla stack
        # Verifica che l'id di 'top' non sia l'head di NESSUN token nel buffer
        return all(token.head != top.id for token in state.B)

    def oracle(self, sent: list['Token'], sentence_index: int = None) -> list['Sample']:
        state = self.create_initial_state(sent)
        samples = []
        while not self.final_state(state):
            if self.LA_is_valid(state) and self.LA_is_correct(state):
                dependency = next(dep for dep in sent if dep.id == state.S[-1].id).dep
                transition = Transition(self.LA, dependency)
                samples.append(Sample(state.copy(), transition))
                self.apply_transition(state, transition)
            elif self.RA_is_valid(state) and self.RA_is_correct(state):
                dependency = next(dep for dep in sent if dep.id == state.B[0].id).dep
                transition = Transition(self.RA, dependency)
                samples.append(Sample(state.copy(), transition))
                self.apply_transition(state, transition)
            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state):
                transition = Transition(self.REDUCE)
                samples.append(Sample(state.copy(), transition))
                self.apply_transition(state, transition)
            else:
                transition = Transition(self.SHIFT)
                samples.append(Sample(state.copy(), transition))
                self.apply_transition(state, transition)
        
        # Aggiungi l'ultimo stato
        samples.append(Sample(state.copy(), None))  
        try:
            assert self.gold_arcs(sent) == state.A
        except AssertionError:
            sentence_text = " ".join([token.form for token in sent])
            error_message = f"Assertion failed for sentence {sentence_index}: {sentence_text}. "
            error_message += f"Expected gold arcs: {self.gold_arcs(sent)}, "
            error_message += f"but got: {state.A}."
            raise AssertionError(error_message)
        return samples
    

    def apply_transition(self, state: State, transition: Transition):
            t = transition.action
            dep = transition.dependency
            s = state.S[-1] if state.S else None
            b = state.B[0] if state.B else None

            if t == self.LA and self.LA_is_valid(state):
                state.A.add((b.id, dep, s.id))
                state.S.pop()
            elif t == self.RA and self.RA_is_valid(state):
                state.A.add((s.id, dep, b.id))
                state.S.append(b)
                del state.B[:1]
            elif t == self.REDUCE and self.REDUCE_is_valid(state):
                state.S.pop()
            elif t == self.SHIFT and state.B:
                state.S.append(b)
                del state.B[:1]

    def gold_arcs(self, sent: list['Token']) -> set:
            gold_arcs = set([])
            for token in sent[1:]:
                gold_arcs.add((token.head, token.dep, token.id))
            return gold_arcs

   


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    # Applying a SHIFT transition
    transition1 = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition1)
    print("State after applying the SHIFT transition:")
    print(state, "\n")

    #### QUESTO LO HO SCRITTO IO
    # Applying a Right arc transition
    transition2 = Transition(arc_eager.RA)
    arc_eager.apply_transition(state, transition2)
    print("State after applying the RA transition:")
    print(state, "\n")
    #####

    #Obtaining the gold_arcs of the sentence with the function gold_arcs
    gold_arcs = arc_eager.gold_arcs(tree)
    print (f"Set of gold arcs: {gold_arcs}\n\n")


    print("**************************************************")
    print("*  Creating instances of the class Transition    *")
    print("**************************************************")

    # Creating a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)
    # Printing the created transition
    print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # Creating a LEFT-ARC transition with a specific dependency type
    left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # Printing the created transition
    print(f"Created Transition: {left_arc_transition}")

    # Creating a RIGHT-ARC transition with a specific dependency type
    right_arc_transition = Transition(ArcEager.RA, "amod")
    # Printing the created transition
    print(f"Created Transition: {right_arc_transition}")

    # Creating a REDUCE transition
    reduce_transition = Transition(ArcEager.REDUCE)
    # Printing the created transition
    print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    print()
    print("**************************************************")
    print("*     Creating instances of the class  Sample    *")
    print("**************************************************")

    # For demonstration, let's create a dummy State instance
    state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # Create a Transition instance. For example, a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)

    # Now, create a Sample instance using the state and transition
    sample_instance = Sample(state, shift_transition)

    # To display the created Sample instance
    print("Sample:\n", sample_instance)





