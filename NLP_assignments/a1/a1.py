# coding: utf-8
"""CS585: Assignment 1

This assignment has two parts:

(1) Finite-State Automata

- Implement a function run_fsa that takes in a *deterministic* FSA and
  a list of symbols and returns True if the FSA accepts the string.

- Design a deterministic FSA to represent a toy example of person names.

(2) Parsing
- Read a context-free grammar from a string.

- Implement a function that takes in a parse tree, a sentence, and a
  grammar, and returns True if the parse tree is a valid parse of the
  sentence.

See the TODO comments below to complete your assignment. I strongly
recommend working one function at a time, using a1_test to check that
your implementation is correct.

For example, after implementing the run_fsa function, run `python
a1_test.py TestA1.test_baa_fsa` to make sure your FSA implementation
works for the sheep example. Then, move on to the next function.

"""


#######
# FSA #
#######

def run_fsa(states, initial_state, accept_states, transition, input_symbols):
    """
    Implement a deterministic finite-state automata.
    See test_baa_fsa.

    Params:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
      input_symbols...list of strings representing the input string
    Returns:
      True if this FSA accepts the input string; False otherwise.
    """
    cur_state = initial_state
    try:
        for char in input_symbols:
            cur_state = transition[cur_state][char]
    except:
        return False

    if cur_state in accept_states:
        return True
    else:
        return False


def get_name_fsa():
    """
    Define a deterministic finite-state machine to recognize a small set of person names, such as:
      Mr. Frank Michael Lewis
      Ms. Flo Lutz
      Frank Micael Lewis
      Flo Lutz

    See test_name_fsa for examples.

    Names have the following:
    - an optional prefix in the set {'Mr.', 'Ms.'}
    - a required first name in the set {'Frank', 'Flo'}
    - an optional middle name in the set {'Michael', 'Maggie'}
    - a required last name in the set {'Lewis', 'Lutz'}

    Returns:
      A 4-tuple of variables, in this order:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
    """
    initial_state = 0
    states = [0,1,2,3,4]
    accept_states = [4]
    transition = {
        0:{'Mr.':1, 'Ms.': 1,'Flo':2,'Frank':2},
        1:{'Frank': 2, 'Flo': 2},
        2:{'Michael':3, 'Maggie': 3,'Lewis':4,'Lutz':4},
        3:{'Lewis':4,'Lutz':4}
    }
    return (states,initial_state,accept_states,transition)


###########
# PARSING #
###########

def read_grammar(lines):
    """Read a list of strings representing CFG rules. E.g., the string
    'S :- NP VP'
    should be parsed into a tuple
    ('S', ['NP', 'VP'])

    Note that the first element of the tuple is a string ('S') for the
    left-hand-side of the rule, and the second element is a list of
    strings (['NP', 'VP']) for the right-hand-side of the rule.

    See test_read_grammar.

    Params:
      lines...A list of strings, one per rule
    Returns:
      A list of (LHS, RHS) tuples, one per rule.
    """
    rule_list = []
    for line in lines:
        Left, Right = line.split(':-')
        LHS = str(Left.strip())
        RHS = [str(x) for x in Right.split()]
        rule_list.append((LHS, RHS))
    return rule_list


class Tree:
    """A partial implementation of a Tree class to represent a parse tree.
    Each node in the Tree is also a Tree.
    Each Tree has two attributes:
      - label......a string representing the node (e.g., 'S', 'NP', 'dog')
      - children...a (possibly empty) list of children of this
                   node. Each element of this list is another Tree.

    A leaf node is a Tree with an empty list of children.
    """

    def __init__(self, label, children=[]):
        """The constructor.
        Params:
          label......A string representing this node
          children...An optional list of Tree nodes, representing the
                     children of this node.
        This is done for you and should not be modified.
        """
        self.label = label
        self.children = children

    def __str__(self):
        """
        Print a string representation of the tree, for debugging.
        This is done for you and should not be modified.
        """
        s = self.label
        for c in self.children:
            s += ' ( ' + str(c) + ' ) '
        return s

    def get_leaves(self):
        """
        Returns:
          A list of strings representing the leaves of this tree.
        See test_get_leaves.
        """
        leaves = []
        def collect_leaves(n):
            nodes = n.children
            for n in nodes:
                ch = n.children
                if len(ch)==0:
                    leaves.append(n.label)
                else:
                    collect_leaves(n)

        collect_leaves(self)
        return leaves


    def get_productions(self):
        """Returns:
          A list of tuples representing a depth-first traversal of
          this tree.  Each tuple is of the form (LHS, RHS), where LHS
          is a string representing the left-hand-side of the
          production, and RHS is a list of strings representing the
          right-hand-side of the production.

        See test_get_productions.
        """
        def collect_prods(n):
            nodes = n.children
            ch_labels = [c.label for c in nodes]
            if len(nodes) > 0:
                prods.append((n.label, ch_labels))
            for n in nodes:
                collect_prods(n)

        prods = []
        collect_prods(self)
        return prods


def is_pos(rule, rules):
    """
    Returns:
      True if this rule is a part-of-speech rule, which is true if none of the
      RHS symbols appear as LHS symbols in other rules.
    E.g., if the grammar is:
    S :- NP VP
    NP :- N
    VP :- V
    N :- dog cat
    V :- run likes

    Then the final two rules are POS rules.
    See test_is_pos.

    This function should be used by the is_valid_production function
    below.
    """
    all_rules =[]
    LHS =[]
    RHS = []
    for r in rules:
        L = r[0]
        R = r[1]
        for s in R:
            all_rules.append((L, [s]))

    for r in all_rules:
        LHS.append(r[0])
        RHS.extend(r[1])

    children = rule[1]

    for c in children:
        if c in LHS:
            return False
    return True


def is_valid_production(production, rules):
    """
    Params:
      production...A (LHS, RHS) tuple representing one production,
                   where LHS is a string and RHS is a list of strings.
      rules........A list of tuples representing the rules of the grammar.

    Returns:
      True if this production is valid according to the rules of the
      grammar; False otherwise.

    See test_is_valid_production.

    This function should be used in the is_valid_tree method below.
    """

    all_prods = rules[:]
    for r in rules:
        L = r[0]
        R = r[1]
        for rc in R:
            new_prod = (L, [rc])
            if is_pos(new_prod, rules) == True:
                all_prods.append(new_prod)

    if production in all_prods:
        return True
    else:
        return False


def is_valid_tree(tree, rules, words):
    """
    Params:
      tree....A Tree object representing a parse tree.
      rules...The list of rules in the grammar.
      words...A list of strings representing the sentence to be parsed.

    Returns:
      True if the tree is a valid parse of this sentence. This requires:
        - every production in the tree is valid (present in the list of rules).
        - the leaf nodes in the tree match the words in the sentence, in order.

    See test_is_valid_tree.
    """
    valid_prods = 0
    all_prods = tree.get_productions()

    for p in all_prods:
        if is_valid_production(p,rules) == False:
            return False
        else:
            valid_prods += 1

    if words == tree.get_leaves():
        l_stat = True
    else:
        return False

    if valid_prods == len(all_prods) and l_stat == True:
        return True
    else:
        return False
