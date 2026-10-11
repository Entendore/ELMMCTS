import numpy as np
import math

# ==========================================
# 1. Code Environment (Simulated Compiler)
# ==========================================
class CodeGenEnv:
    def __init__(self):
        # Target code to generate: 'if x : y <EOS>'
        self.target_seq = ['if', 'x', ':', 'y', '<EOS>']
        self.vocab = ['if', 'x', 'y', 'z', ':', '<EOS>']
        self.vocab_size = len(self.vocab)
        self.max_len = 5
        
    def reset(self):
        self.sequence = []
        
    def get_state(self):
        """One-hot encode the current sequence, padded to max_len."""
        state = np.zeros((self.max_len, self.vocab_size))
        for i, token in enumerate(self.sequence):
            state[i, self.vocab.index(token)] = 1
        return state.flatten()
    
    def get_valid_actions(self):
        """Grammar rules: enforce valid syntax."""
        if len(self.sequence) == 0: return [self.vocab.index('if')]
        if len(self.sequence) == 1: return [self.vocab.index('x'), self.vocab.index('z')]
        if len(self.sequence) == 2: return [self.vocab.index(':')]
        if len(self.sequence) == 3: return [self.vocab.index('y'), self.vocab.index('z')]
        if len(self.sequence) == 4: return [self.vocab.index('<EOS>')]
        return []
        
    def step(self, action_idx):
        self.sequence.append(self.vocab[action_idx])
        
    def is_terminal(self):
        return len(self.sequence) == self.max_len
        
    def get_reward(self):
        """Simulated Compiler Reward: +1 for exact match, -0.5 for valid syntax but wrong var."""
        if not self.is_terminal(): return 0
        if self.sequence == self.target_seq: return 1.0
        # Valid syntax structure but used wrong variables
        if self.sequence[0] == 'if' and self.sequence[2] == ':' and self.sequence[4] == '<EOS>':
            return -0.5
        return -1.0 # Syntax error (should not happen due to valid_actions mask)

# ==========================================
# 2. Feed-Forward Network (RELM)
# ==========================================
class RELM_CodeGen_Net:
    def __init__(self, input_size, hidden_size, action_size, l2_reg=0.01):
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.l2_reg = l2_reg
        
        # Frozen random input-to-hidden weights (feature mapping)
        self.W_in = np.random.randn(input_size, hidden_size) * 0.1
        self.bias = np.random.randn(1, hidden_size) * 0.1
        
        # Learnable hidden-to-output weights
        self.W_out = None

    def _get_hidden(self, state):
        if state.ndim == 1: state = state.reshape(1, -1)
        return np.maximum(0, np.dot(state, self.W_in) + self.bias) # ReLU

    def predict(self, state):
        if state.ndim == 1: state = state.reshape(1, -1)
        H = self._get_hidden(state)
        
        if self.W_out is None:
            policy_logits = np.zeros(self.action_size)
            value = 0.0
        else:
            out = np.dot(H, self.W_out)[0]
            policy_logits = out[:self.action_size]
            value = out[self.action_size]
            
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        policy = exp_logits / np.sum(exp_logits)
        value = np.tanh(value)
        return policy, value

    def fit(self, states, target_policies, target_values):
        """One-shot learning via Regularized Pseudoinverse (No Backprop)"""
        H = self._get_hidden(states)
        
        # Convert target policies to logits
        target_log_policies = np.log(target_policies + 1e-8) 
        Y = np.hstack([target_log_policies, target_values.reshape(-1, 1)])
        
        # RELM closed-form solution: W_out = (H^T * H + λI)^-1 * H^T * Y
        HtH = np.dot(H.T, H)
        reg_matrix = self.l2_reg * np.eye(HtH.shape[0])
        inv_term = np.linalg.inv(HtH + reg_matrix)
        self.W_out = np.dot(np.dot(inv_term, H.T), Y)

# ==========================================
# 3. PUCT MCTS for Sequence Generation
# ==========================================
class MCTSNode:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        
    @property
    def q_value(self):
        if self.visit_count == 0: return 0
        return self.total_value / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

def expand_and_evaluate(node, env, network):
    """ Expands node using network's policy. Returns value from sequence perspective. """
    if env.is_terminal():
        return env.get_reward()
        
    policy, value = network.predict(node.state)
    valid_actions = env.get_valid_actions()
    
    masked_policy = np.zeros(network.action_size)
    for a in valid_actions:
        masked_policy[a] = policy[a]
    masked_policy /= np.sum(masked_policy)
    
    for a in valid_actions:
        # Simulate appending token to create child state
        temp_env = CodeGenEnv()
        temp_env.sequence = env.sequence.copy()
        temp_env.step(a)
        child_state = temp_env.get_state()
        node.children[a] = MCTSNode(state=child_state, parent=node, prior=masked_policy[a])
        
    return value

def mcts_search(root_state, root_seq, network, num_simulations=50, c_puct=1.0):
    root = MCTSNode(state=root_state)
    
    # Initial expansion of root
    temp_env = CodeGenEnv()
    temp_env.sequence = root_seq.copy()
    value = expand_and_evaluate(root, temp_env, network)
    root.visit_count = 1
    root.total_value = value
    
    # Add Dirichlet noise to root for exploration
    dirichlet_noise = np.random.dirichlet([0.5] * len(root.children))
    for i, child in enumerate(root.children.values()):
        child.prior = 0.75 * child.prior + 0.25 * dirichlet_noise[i]

    for _ in range(num_simulations):
        node = root
        env = CodeGenEnv()
        env.sequence = root_seq.copy()
        
        # Selection
        while node.is_expanded() and node.children:
            action = max(node.children, key=lambda a: node.children[a].q_value + c_puct * node.children[a].prior * math.sqrt(node.visit_count) / (1 + node.children[a].visit_count))
            node = node.children[action]
            env.step(action)
            
        # Expansion & Evaluation
        leaf_value = expand_and_evaluate(node, env, network)
        
        # Backpropagation
        while node is not None:
            node.visit_count += 1
            node.total_value += leaf_value
            node = node.parent
            
    return root

# ==========================================
# 4. Code Generation & Training Loop
# ==========================================
def generate_and_train():
    print("Initializing RELM Network for Code Gen...")
    env = CodeGenEnv()
    net = RELM_CodeGen_Net(input_size=30, hidden_size=128, action_size=6, l2_reg=0.01)
    
    num_iterations = 15
    games_per_iter = 10
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        all_states = []
        all_policies = []
        all_values = []
        
        for g in range(games_per_iter):
            env.reset()
            sequence_data = []
            
            while not env.is_terminal():
                root = mcts_search(env.get_state(), env.sequence.copy(), net, num_simulations=30)
                
                # Select action (Temperature = 0.5 for moderate exploration)
                visits = np.array([child.visit_count for child in root.children.values()])
                actions = list(root.children.keys())
                visits = visits ** (1.0 / 0.5)
                probs = visits / np.sum(visits)
                action = np.random.choice(actions, p=probs)
                
                # Store training data
                all_states.append(env.get_state())
                improved_policy = np.zeros(env.vocab_size)
                for a, child in root.children.items():
                    improved_policy[a] = child.visit_count / root.visit_count
                all_policies.append(improved_policy)
                
                env.step(action)
            
            # Assign values to all steps in this generated sequence
            final_reward = env.get_reward()
            # Simple credit assignment: all steps get the final reward (can be improved)
            for _ in range(len(env.sequence)):
                all_values.append(final_reward)
                
        print(f"Generated {games_per_iter} code snippets. Training RELM (No Backprop)...")
        X = np.array(all_states)
        Y_p = np.array(all_policies)
        Y_v = np.array(all_values)
        
        net.fit(X, Y_p, Y_v)

    # Final Evaluation
    print("\n=== Final Code Generation Test ===")
    env.reset()
    print("Target: 'if x : y <EOS>'")
    print("Generated: ", end="")
    
    while not env.is_terminal():
        root = mcts_search(env.get_state(), env.sequence.copy(), net, num_simulations=50, add_dirichlet=False)
        action = max(root.children, key=lambda a: root.children[a].visit_count) # Greedy
        env.step(action)
        print(f"'{env.sequence[-1]}' ", end="")
    print(f"\nFinal Reward: {env.get_reward()}")

if __name__ == "__main__":
    generate_and_train()