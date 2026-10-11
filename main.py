import numpy as np

# ==========================================
# 1. The ELM Ensemble (No Backpropagation)
# ==========================================
class ELM_Net:
    def __init__(self, input_size, hidden_size, output_size, l2_lambda=1e-3):
        self.hidden_size = hidden_size
        self.l2_lambda = l2_lambda
        
        # Frozen random features
        self.W_in = np.random.randn(input_size, hidden_size) * 0.1
        self.bias = np.random.randn(1, hidden_size) * 0.1
        
        # Learnable output weights (solved analytically)
        self.W_out = None

    def _get_hidden(self, X):
        return np.maximum(0, np.dot(X, self.W_in) + self.bias) # ReLU

    def fit(self, X, Y):
        H = self._get_hidden(X)
        # Ridge Regression (Regularized Pseudoinverse) to prevent overfitting
        # W = (H^T * H + λI)^-1 * H^T * Y
        HtH = np.dot(H.T, H) + self.l2_lambda * np.eye(self.hidden_size)
        self.W_out = np.dot(np.linalg.inv(HtH), np.dot(H.T, Y))

    def predict_linear(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        if self.W_out is None: # Untrained, return zeros
            return np.zeros((X.shape[0], self.W_out_shape))
        H = self._get_hidden(X)
        return np.dot(H, self.W_out)

class ELM_Ensemble:
    """
    Combines multiple ELMs to create a more robust AlphaZero-like network.
    Averages the outputs of all ELMs for both Policy and Value.
    """
    def __init__(self, input_size, action_size, num_elms=3, hidden_size=64):
        self.action_size = action_size
        # Output = action_size (policy logits) + 1 (value)
        self.nets = [ELM_Net(input_size, hidden_size, action_size + 1) for _ in range(num_elms)]

    def predict(self, state):
        if state.ndim == 1: state = state.reshape(1, -1)
        
        # Average predictions across all ELMs
        outputs = [net.predict_linear(state) for net in self.nets]
        avg_out = np.mean(outputs, axis=0)[0] # Extract single sample
        
        # Split into policy and value
        policy_logits = avg_out[:self.action_size]
        value = avg_out[self.action_size]
        
        # Softmax for policy
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        policy = exp_logits / np.sum(exp_logits)
        
        # Tanh for value
        value = np.tanh(value)
        return policy, value

    def fit(self, states, target_policies, target_values):
        # Convert targets to linear space
        target_log_policies = np.log(target_policies + 1e-8)
        Y = np.hstack([target_log_policies, target_values.reshape(-1, 1)])
        
        # Train each ELM in the ensemble on the same data
        for net in self.nets:
            net.fit(states, Y)


# ==========================================
# 2. The "Learning to Code" Environment
# ==========================================
class CodeEnv:
    def __init__(self, target_code):
        self.tokens = ["def", " ", "add", "(", "a", ")", ":", " ", "return", " ", "a"] # Target: "def add(a): return a"
        self.target = target_code
        self.max_len = len(self.target)
        self.vocab = sorted(list(set(self.target)))
        self.vocab_size = len(self.vocab)
        self.action_size = self.vocab_size

    def reset(self):
        return [] # Empty code string

    def step(self, state, action_idx):
        token = self.vocab[action_idx]
        new_state = state + [token]
        return new_state

    def is_terminal(self, state):
        return len(state) >= self.max_len

    def get_reward(self, state):
        # Partial reward: fraction of correct tokens in the right position
        correct = sum(1 for i, t in enumerate(state) if i < len(self.target) and t == self.target[i])
        return correct / len(self.target)

    def state_to_vector(self, state):
        # One-hot encoding of the current code sequence
        vec = np.zeros(self.max_len * self.vocab_size)
        for i, token in enumerate(state):
            vec[i * self.vocab_size + self.vocab.index(token)] = 1.0
        return vec

# ==========================================
# 3. Corrected PUCT MCTS
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
    """Expands node using network's policy, returns network's value."""
    state_vec = env.state_to_vector(node.state)
    policy, value = network.predict(state_vec)
    
    # Create children for all valid actions (appending any vocab token)
    for a in range(env.action_size):
        next_state = env.step(node.state, a)
        node.children[a] = MCTSNode(state=next_state, parent=node, prior=policy[a])
        
    return value

def puct_select_child(node, c_puct=1.0):
    best_score = -float('inf')
    best_action = None
    for action, child in node.children.items():
        u_value = c_puct * child.prior * (np.sqrt(node.visit_count) / (1 + child.visit_count))
        score = child.q_value + u_value
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

def mcts_search(root_state, env, network, num_simulations=50):
    root = MCTSNode(state=root_state)
    root.visit_count = 1 # Initialize root visits to avoid N=0 division
    
    # 1. Expand root first
    value = expand_and_evaluate(root, env, network)
    root.total_value += value
    
    for _ in range(num_simulations):
        node = root
        state = root_state
        
        # 2. Selection: Traverse tree until a leaf is found
        while node.is_expanded() and not env.is_terminal(node.state):
            action = puct_select_child(node)
            node = node.children[action]
            
        # 3. Expansion & Evaluation
        if env.is_terminal(node.state):
            # TRUE terminal value (no network guess needed)
            value = env.get_reward(node.state)
        else:
            # Network evaluates leaf
            value = expand_and_evaluate(node, env, network)
            
        # 4. Backpropagation
        temp_node = node
        while temp_node is not None:
            temp_node.visit_count += 1
            temp_node.total_value += value
            temp_node = temp_node.parent
            
    # Return improved policy based on visit counts
    improved_policy = np.zeros(env.action_size)
    for action, child in root.children.items():
        improved_policy[action] = child.visit_count / root.visit_count
        
    return improved_policy


# ==========================================
# 4. Self-Play & Training Loop
# ==========================================
if __name__ == "__main__":
    env = CodeEnv(target_code=["def", " ", "add", "(", "a", ")", ":", " ", "return", " ", "a"])
    
    # Ensemble of 3 ELMs, mapping 44-dim state to 6 actions
    network = ELM_Ensemble(input_size=env.max_len * env.vocab_size, action_size=env.action_size, num_elms=3)
    
    print("--- Simulating Self-Play & Training (Learning to Code) ---")
    
    replay_states = []
    replay_policies = []
    replay_values = []
    
    # Play 10 "games" (write 10 code attempts)
    for game in range(10):
        state = env.reset()
        game_states = []
        game_policies = []
        
        while not env.is_terminal(state):
            state_vec = env.state_to_vector(state)
            
            # MCTS guides the next move
            improved_policy = mcts_search(state, env, network, num_simulations=30)
            
            # Action selection (we can sample or pick argmax. Let's pick argmax for exploitation)
            action = np.argmax(improved_policy)
            
            game_states.append(state_vec)
            game_policies.append(improved_policy)
            
            state = env.step(state, action)
            
        # Game over, get true reward
        reward = env.get_reward(state)
        
        # Store in replay buffer
        # Value is the final reward, seen from the perspective of the state
        replay_states.extend(game_states)
        replay_policies.extend(game_policies)
        replay_values.extend([reward] * len(game_states))
        
        generated_code = "".join(state)
        print(f"Game {game+1} Code: {generated_code} | Reward: {reward:.2f}")
        
    # Train Ensemble WITHOUT BACKPROPAGATION
    X = np.array(replay_states)
    P = np.array(replay_policies)
    V = np.array(replay_values)
    
    print("\nTraining ELM Ensemble analytically...")
    network.fit(X, P, V)
    print("Training complete.")
    
    # Test the trained network
    print("\n--- Testing Trained Network ---")
    test_state = env.reset()
    while not env.is_terminal(test_state):
        state_vec = env.state_to_vector(test_state)
        # Use pure network prediction (no MCTS) to see if it learned the code
        policy, value = network.predict(state_vec)
        action = np.argmax(policy)
        test_state = env.step(test_state, action)
        
    print(f"Network generated code: {test_state}")