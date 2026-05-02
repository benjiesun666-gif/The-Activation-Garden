"""
Trainable EML Trees for Symbolic Regression
Based on Section 4.3 of arXiv:2603.21852

Uses PyTorch to implement differentiable EML trees where leaf selection
is parameterized via softmax over α, β, γ parameters.
"""
import torch
import torch.nn as nn
import math

class TrainableEMLNode(nn.Module):
    """
    A single EML node that selects its inputs via differentiable parameters.
    Each input can be: 1, x, or output from previous layer.
    Parameterized as: α*1 + β*x + γ*f_prev
    """
    def __init__(self, x_available: bool = True, prev_available: bool = True):
        super().__init__()
        self.x_available = x_available
        self.prev_available = prev_available
        # α (constant 1 weight), β (input x weight), γ (previous result weight)
        n_params = 1  # always have α (for constant 1)
        if x_available:
            n_params += 1
        if prev_available:
            n_params += 1
        self.logits = nn.Parameter(torch.randn(n_params) * 0.1)

    def forward(self, x: torch.Tensor, prev: torch.Tensor = None) -> torch.Tensor:
        """Compute the weighted sum of available inputs."""
        parts = []
        idx = 0
        # Constant 1 term
        parts.append(torch.ones_like(x) * self.logits[idx])
        idx += 1
        # Input x term
        if self.x_available:
            parts.append(x * self.logits[idx])
            idx += 1
        # Previous result term
        if self.prev_available and prev is not None:
            parts.append(prev * self.logits[idx])
            idx += 1
        return sum(parts)


class TrainableEMLTree(nn.Module):
    """
    Trainable EML tree of a given depth.
    
    Architecture:
    - Depth 0: just input selection (leaves)
    - Depth d: binary tree with 2^d leaves and 2^d - 1 EML nodes
    
    Each EML node has two child sub-trees (left and right).
    """
    def __init__(self, depth: int, use_softmax: bool = True):
        super().__init__()
        self.depth = depth
        self.use_softmax = use_softmax
        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1
        
        # Build tree structure recursively
        self._build_leaves(depth)
        self._build_nodes(depth)
    
    def _build_leaves(self, depth: int):
        """Build leaf parameters - each leaf selects: constant 1 or input x."""
        n_leaves = 2 ** depth
        # Each leaf has logits for [1, x] selection
        self.leaf_params = nn.Parameter(torch.randn(n_leaves, 2) * 0.01)
    
    def _build_nodes(self, depth: int):
        """Build internal EML node parameters."""
        # Each internal node produces: eml(left_input, right_input)
        # Where left_input = α_L*1 + β_L*x + γ_L*prev_L
        #       right_input = α_R*1 + β_R*x + γ_R*prev_R
        # But for simplicity, each node selects from [1, x, lchild_output, rchild_output]
        n_nodes = 2 ** depth - 1
        self.node_params = nn.Parameter(torch.randn(n_nodes, 4) * 0.01)  # weights for 1, x, L, R
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: evaluate the EML tree on input x."""
        batch_size = x.shape[0]
        x = x.view(-1, 1)
        
        # Compute leaf values: soft selection between 1 and x
        logits = self.leaf_params
        if self.use_softmax:
            weights = torch.softmax(logits, dim=1)
        else:
            weights = torch.sigmoid(logits) / torch.sigmoid(logits).sum(dim=1, keepdim=True)
        ones = torch.ones(batch_size, self.n_leaves, device=x.device)
        xs = x.unsqueeze(1).expand(-1, self.n_leaves)
        leaf_values = weights[:, 0:1] * ones + weights[:, 1:2] * xs  # [B, n_leaves]
        
        # Compute internal EML nodes bottom-up
        current_values = leaf_values  # [B, n_leaves]
        
        for level in range(self.depth - 1, -1, -1):
            n_at_level = 2 ** level
            start = 0
            new_values = []
            for i in range(n_at_level):
                # Each node takes two children
                left = current_values[:, 2*i]
                right = current_values[:, 2*i + 1]
                
                # EML operation
                result = torch.exp(left) - torch.log(torch.clamp(right, min=1e-10))
                new_values.append(result)
            
            current_values = torch.stack(new_values, dim=1)  # [B, n_at_level]
        
        return current_values[:, 0]  # Root value


class TrainableEMLTreeV2(nn.Module):
    """
    Improved version with proper parameterized input selection
    following the paper's master formula approach (Section 4.3).
    
    Each node has 3 parameters α, β, γ controlling:
    input = α*1 + β*x + γ*(intermediate_value)
    """
    def __init__(self, depth: int, temperature: float = 1.0):
        super().__init__()
        self.depth = depth
        self.temperature = temperature
        self.n_leaves = 2 ** depth
        self.n_internal = 2 ** depth - 1
        total_nodes = self.n_leaves + self.n_internal
        
        # For each node in the binary tree (in breadth-first order),
        # we have 3 parameters: alpha (constant), beta (x), gamma (parent/previous)
        # Leaves: only alpha and beta (no gamma since no previous)
        # Internal: alpha, beta, gamma (gamma points to left or right child result)
        
        # Use a simpler approach: flat representation
        # Each of the 2^depth - 1 internal EML nodes gets its own 
        # left-selector (αl, βl, γl) and right-selector (αr, βr, γr)
        
        # For leaves: softmax over [constant 1, variable x] -> 2 params each
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, 2) * 0.1)
        
        # For internal EML nodes: left/right select from [1, x, child_output]
        self.node_logits_L = nn.Parameter(torch.randn(self.n_internal, 3) * 0.1)
        self.node_logits_R = nn.Parameter(torch.randn(self.n_internal, 3) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.view(-1, 1)
        
        # Step 1: Compute all leaf values as soft combination of 1 and x
        leaf_w = torch.softmax(self.leaf_logits / self.temperature, dim=1)  # [L, 2]
        ones = torch.ones(batch, self.n_leaves, device=x.device)
        xs = x.expand(-1, self.n_leaves)
        leaf_vals = leaf_w[:, 0] * ones + leaf_w[:, 1] * xs  # [B, L]
        
        # Step 2: Build up tree bottom-up
        # Tree in heap-like indexing: internal node i has children 2i+1 and 2i+2
        # Leaves are at indices [n_internal, n_internal + n_leaves)
        
        total = self.n_internal + self.n_leaves
        values = [None] * total
        
        # Fill leaves
        for i in range(self.n_leaves):
            values[self.n_internal + i] = leaf_vals[:, i]
        
        # Compute internal nodes bottom-up (reverse order)
        node_w_L = torch.softmax(self.node_logits_L / self.temperature, dim=1)
        node_w_R = torch.softmax(self.node_logits_R / self.temperature, dim=1)
        
        for i in range(self.n_internal - 1, -1, -1):
            left_child_idx = 2 * i + 1
            right_child_idx = 2 * i + 2
            
            # Children could be internal nodes or leaves
            if left_child_idx >= self.n_internal:
                left_child_val = values[left_child_idx]  # it's a leaf
            else:
                left_child_val = values[left_child_idx]  # it's an internal node
            
            if right_child_idx >= self.n_internal:
                right_child_val = values[right_child_idx]
            else:
                right_child_val = values[right_child_idx]
            
            # Select left input: mix of 1, x, left_child_val
            wl = node_w_L[i]  # [3]
            left_input = wl[0] * torch.ones(batch, device=x.device) + \
                         wl[1] * x.squeeze(-1) + \
                         wl[2] * left_child_val
            
            # Select right input: mix of 1, x, right_child_val
            wr = node_w_R[i]  # [3]
            right_input = wr[0] * torch.ones(batch, device=x.device) + \
                          wr[1] * x.squeeze(-1) + \
                          wr[2] * right_child_val
            
            # EML operation
            result = torch.exp(left_input) - torch.log(torch.clamp(right_input, min=1e-10))
            values[i] = result
        
        return values[0]


class EMLLoss(nn.Module):
    """MSE loss for training EML trees."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


def train_eml_tree(
    tree: TrainableEMLTreeV2,
    target_fn,
    x_range=(-3.0, 3.0),
    n_points=200,
    n_epochs=5000,
    lr=0.01,
    device='cuda'
):
    """Train an EML tree to approximate a target function."""
    tree = tree.to(device)
    optimizer = torch.optim.Adam(tree.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=300, factor=0.5)
    loss_fn = nn.MSELoss()
    
    # Generate training data
    x_train = torch.linspace(x_range[0], x_range[1], n_points, device=device)
    y_train = target_fn(x_train)
    
    tree.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = tree(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 10.0)
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}: loss = {loss.item():.6e}")
    
    return tree


def analyze_leaf_weights(tree: TrainableEMLTreeV2) -> dict:
    """Analyze which leaves selected '1' vs 'x'."""
    with torch.no_grad():
        leaf_w = torch.softmax(tree.leaf_logits, dim=1)
        results = {
            'leaf_weights': leaf_w.cpu().numpy(),
            'n_constant_leaves': (leaf_w[:, 0] > 0.5).sum().item(),
            'n_variable_leaves': (leaf_w[:, 1] > 0.5).sum().item(),
        }
    return results


def analyze_node_weights(tree: TrainableEMLTreeV2) -> dict:
    """Analyze what each EML node selected for left/right inputs."""
    with torch.no_grad():
        wL = torch.softmax(tree.node_logits_L, dim=1).cpu().numpy()
        wR = torch.softmax(tree.node_logits_R, dim=1).cpu().numpy()
        results = {
            'node_weights_L': wL,
            'node_weights_R': wR,
            'dominant_L': wL.argmax(axis=1),  # 0=constant, 1=x, 2=child
            'dominant_R': wR.argmax(axis=1),
        }
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test: fit exp(x) with depth-2 EML tree
    print("\n=== Test: Fit exp(x) with depth-2 EML tree ===")
    tree = TrainableEMLTreeV2(depth=2)
    
    def target_exp(x):
        return torch.exp(x)
    
    trained = train_eml_tree(
        tree, target_exp, 
        x_range=(-2.0, 2.0), 
        n_points=100, 
        n_epochs=3000, 
        lr=0.01,
        device=device
    )
    
    # Analyze weights
    leaf_info = analyze_leaf_weights(trained)
    node_info = analyze_node_weights(trained)
    print(f"Leaf weights:\n{leaf_info['leaf_weights']}")
    print(f"Node L dominant: {node_info['dominant_L']}")
    print(f"Node R dominant: {node_info['dominant_R']}")
    
    # Verify
    trained.eval()
    with torch.no_grad():
        x_test = torch.tensor([0.0, 0.5, 1.0, 2.0], device=device)
        y_pred = trained(x_test)
        y_true = torch.exp(x_test)
        print(f"\nx:      {x_test.cpu().numpy()}")
        print(f"pred:   {y_pred.cpu().numpy()}")
        print(f"true:   {y_true.cpu().numpy()}")
