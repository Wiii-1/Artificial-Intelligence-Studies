class RegexNode:
    def __init__(self, symbol=None, left=None, right=None, node_type=None):
        self.symbol = symbol       # character for leaves
        self.left = left           # left child
        self.right = right         # right child
        self.node_type = node_type # 'leaf', 'concat', 'union', 'star'
        self.nullable = False
        self.firstpos = set()
        self.lastpos = set()

def build_syntax_tree(regex):
    # For illustration, only supports single chars, concatenation, union, star
    # Assumes regex is fully parenthesized and simple (e.g. (a|b)*c)
    import re
    # include '#' in tokens and allow lowercase letters; keep operators and parentheses
    tokens = list(filter(lambda x: x, re.findall(r'[()*|#]|[a-z]', regex)))

    pos = 1
    leaf_nodes = {}

    def parse_expr(tokens):
        def parse_factor():
            nonlocal pos
            token = tokens.pop(0)
            if token == '(':
                node = parse_expr(tokens)
                assert tokens.pop(0) == ')'
            elif token.isalpha() or token == '#':
                leaf = RegexNode(symbol=token, node_type='leaf')
                leaf_nodes[pos] = leaf
                leaf.firstpos.add(pos)
                leaf.lastpos.add(pos)
                leaf.nullable = False
                leaf.pos = pos
                pos += 1
                node = leaf
            else:
                raise ValueError(f"Unexpected token {token}")

            # handle postfix star operator(s)
            while tokens and tokens[0] == '*':
                tokens.pop(0)
                node = RegexNode(left=node, node_type='star')
            return node

        def parse_term():
            node = parse_factor()
            while tokens and tokens[0] not in ['|', ')']:
                right = parse_factor()
                node = RegexNode(left=node, right=right, node_type='concat')
            return node

        node = parse_term()
        while tokens and tokens[0] == '|':
            tokens.pop(0)
            right = parse_term()
            node = RegexNode(left=node, right=right, node_type='union')
        return node

    root = parse_expr(tokens)

    return root, leaf_nodes

def compute_nullable_first_last(node):
    if node.node_type == 'leaf':
        return
    if node.node_type == 'concat':
        compute_nullable_first_last(node.left)
        compute_nullable_first_last(node.right)
        node.nullable = node.left.nullable and node.right.nullable
        node.firstpos = node.left.firstpos.copy()
        if node.left.nullable:
            node.firstpos.update(node.right.firstpos)
        node.lastpos = node.right.lastpos.copy()
        if node.right.nullable:
            node.lastpos.update(node.left.lastpos)
    elif node.node_type == 'union':
        compute_nullable_first_last(node.left)
        compute_nullable_first_last(node.right)
        node.nullable = node.left.nullable or node.right.nullable
        node.firstpos = node.left.firstpos | node.right.firstpos
        node.lastpos = node.left.lastpos | node.right.lastpos
    elif node.node_type == 'star':
        compute_nullable_first_last(node.left)
        node.nullable = True
        node.firstpos = node.left.firstpos.copy()
        node.lastpos = node.left.lastpos.copy()

def compute_followpos(node, followpos):
    if node.node_type == 'concat':
        for i in node.left.lastpos:
            followpos[i].update(node.right.firstpos)
        compute_followpos(node.left, followpos)
        compute_followpos(node.right, followpos)
    elif node.node_type == 'star':
        for i in node.lastpos:
            followpos[i].update(node.firstpos)
        compute_followpos(node.left, followpos)
    elif node.node_type in ['union', 'leaf']:
        if node.left:
            compute_followpos(node.left, followpos)
        if node.right:
            compute_followpos(node.right, followpos)

class DFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def matches(self, s):
        current = self.start_state
        for c in s:
            if c not in self.alphabet or (current, c) not in self.transitions:
                return False
            current = self.transitions[(current, c)]
        return current in self.accept_states

def regex_to_dfa(regex):
    # Append a unique symbol '#' to mark string end
    regex = f"({regex})#"
    root, leaf_nodes = build_syntax_tree(regex)

    compute_nullable_first_last(root)

    followpos = {pos: set() for pos in leaf_nodes}
    compute_followpos(root, followpos)

    states = []
    alphabet = set(c.symbol for c in leaf_nodes.values() if c.symbol != '#')
    start_state = frozenset(root.firstpos)
    states.append(start_state)
    unmarked = [start_state]
    transitions = {}
    accept_pos = next(pos for pos, node in leaf_nodes.items() if node.symbol == '#')
    accept_states = set()

    while unmarked:
        state = unmarked.pop()
        if accept_pos in state:
            accept_states.add(state)
        for symbol in alphabet:
            next_state = set()
            for p in state:
                if leaf_nodes[p].symbol == symbol:
                    next_state.update(followpos[p])
            next_state = frozenset(next_state)
            if next_state and next_state not in states:
                states.append(next_state)
                unmarked.append(next_state)
            if next_state:
                transitions[(state, symbol)] = next_state

    return DFA(states, alphabet, transitions, start_state, accept_states)

# Testing the DFA with regex and strings
regex = "a|(bc)*" 
dfa = regex_to_dfa(regex)

test_strings = ["", "a", "bc", "bcbc", "b", "c", "abc"]
for s in test_strings:
    print(f"Matches '{s}': {dfa.matches(s)}")
