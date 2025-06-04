//! Reinforcement learning for tree generation

use std::sync::Arc;
use ndarray::{Array1, Array2};
use crate::algebra::{Tree, Forest, HopfAlgebra, TreeBuilder};

/// Actions for tree construction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeAction {
    /// Add a child to node i
    AddChild(usize),
    /// Terminate construction
    Stop,
}

/// State representation for tree construction
#[derive(Debug, Clone)]
pub struct TreeState {
    /// Current tree being constructed
    pub tree: Tree,
    /// Maximum allowed size
    pub max_size: usize,
    /// Current step
    pub step: usize,
    /// History of actions taken
    pub action_history: Vec<TreeAction>,
}

impl TreeState {
    /// Create initial state
    pub fn new(max_size: usize) -> Self {
        TreeState {
            tree: Tree::new(),
            max_size,
            step: 0,
            action_history: Vec::new(),
        }
    }
    
    /// Check if state is terminal
    pub fn is_terminal(&self) -> bool {
        self.tree.size() >= self.max_size || 
        self.action_history.last() == Some(&TreeAction::Stop)
    }
    
    /// Get valid actions from current state
    pub fn valid_actions(&self) -> Vec<TreeAction> {
        let mut actions = vec![TreeAction::Stop];
        
        if self.tree.size() < self.max_size {
            // Can add child to any existing node
            for node in 0..self.tree.size() {
                actions.push(TreeAction::AddChild(node));
            }
        }
        
        actions
    }
    
    /// Apply action to get next state
    pub fn apply_action(&self, action: TreeAction) -> TreeState {
        let mut new_state = self.clone();
        new_state.step += 1;
        new_state.action_history.push(action);
        
        match action {
            TreeAction::AddChild(parent) => {
                // Add a new node as child of parent
                let new_node = new_state.tree.size();
                let mut children = vec![Vec::new(); new_state.tree.size() + 1];
                
                // Copy existing structure
                for i in 0..new_state.tree.size() {
                    children[i] = new_state.tree.children(i).to_vec();
                }
                
                // Add new child
                children[parent].push(new_node);
                
                // Rebuild tree
                new_state.tree = Tree::from_adjacency(children)
                    .unwrap_or_else(|_| self.tree.clone());
            }
            TreeAction::Stop => {
                // No change to tree
            }
        }
        
        new_state
    }
    
    /// Extract features from state
    pub fn features(&self) -> Array1<f32> {
        let mut features = Array1::zeros(20);
        
        // Basic tree features
        features[0] = self.tree.size() as f32;
        features[1] = self.tree.max_depth() as f32;
        features[2] = self.tree.leaf_count() as f32;
        features[3] = (self.tree.size() as f32) / (self.max_size as f32);
        features[4] = self.step as f32;
        
        // Structural features
        if self.tree.size() > 0 {
            let degrees: Vec<usize> = (0..self.tree.size())
                .map(|n| self.tree.node_degree(n))
                .collect();
            features[5] = degrees.iter().sum::<usize>() as f32 / degrees.len() as f32;
            features[6] = *degrees.iter().max().unwrap_or(&0) as f32;
        }
        
        // Recent action features
        if let Some(last_action) = self.action_history.last() {
            match last_action {
                TreeAction::AddChild(node) => {
                    features[7] = 1.0;
                    features[8] = *node as f32;
                    features[9] = self.tree.node_depth(*node) as f32;
                }
                TreeAction::Stop => {
                    features[10] = 1.0;
                }
            }
        }
        
        features
    }
}

/// Environment for tree generation
pub struct TreeEnvironment {
    /// Hopf algebra for rewards
    hopf: Arc<HopfAlgebra>,
    /// Target properties
    target_size: Option<usize>,
    target_depth: Option<usize>,
    /// Reward shaping parameters
    balance_weight: f32,
    hopf_weight: f32,
    diversity_weight: f32,
}

impl TreeEnvironment {
    /// Create new environment
    pub fn new(hopf: Arc<HopfAlgebra>) -> Self {
        TreeEnvironment {
            hopf,
            target_size: None,
            target_depth: None,
            balance_weight: 1.0,
            hopf_weight: 2.0,
            diversity_weight: 0.5,
        }
    }
    
    /// Set target properties
    pub fn with_targets(mut self, size: Option<usize>, depth: Option<usize>) -> Self {
        self.target_size = size;
        self.target_depth = depth;
        self
    }
    
    /// Compute reward for a state transition
    pub fn reward(&self, state: &TreeState, action: &TreeAction, next_state: &TreeState) -> f32 {
        let mut reward = 0.0;
        
        // Step penalty to encourage efficiency
        reward -= 0.01;
        
        // Action-specific rewards
        match action {
            TreeAction::AddChild(parent) => {
                // Reward for growth
                reward += 0.1;
                
                // Reward for maintaining balance
                let balance_score = self.compute_balance(&next_state.tree);
                reward += self.balance_weight * balance_score;
                
                // Penalty for making tree too deep
                if let Some(target_depth) = self.target_depth {
                    let depth_diff = (next_state.tree.max_depth() as f32 - target_depth as f32).abs();
                    reward -= depth_diff * 0.1;
                }
            }
            TreeAction::Stop => {
                // Terminal reward based on final tree quality
                reward += self.evaluate_tree(&next_state.tree);
            }
        }
        
        reward
    }
    
    /// Evaluate final tree quality
    fn evaluate_tree(&self, tree: &Tree) -> f32 {
        let mut score = 0.0;
        
        // Size closeness to target
        if let Some(target_size) = self.target_size {
            let size_diff = (tree.size() as f32 - target_size as f32).abs();
            score -= size_diff * 0.5;
        }
        
        // Depth closeness to target
        if let Some(target_depth) = self.target_depth {
            let depth_diff = (tree.max_depth() as f32 - target_depth as f32).abs();
            score -= depth_diff * 0.3;
        }
        
        // Hopf-algebraic properties
        let hopf_score = self.compute_hopf_score(tree);
        score += self.hopf_weight * hopf_score;
        
        // Structural diversity
        let diversity_score = self.compute_diversity(tree);
        score += self.diversity_weight * diversity_score;
        
        score
    }
    
    /// Compute balance score
    fn compute_balance(&self, tree: &Tree) -> f32 {
        if tree.size() <= 1 {
            return 1.0;
        }
        
        let mut balance_scores = Vec::new();
        
        for node in 0..tree.size() {
            let children = tree.children(node);
            if children.len() >= 2 {
                let sizes: Vec<f32> = children.iter()
                    .map(|&c| tree.subtree_size(c) as f32)
                    .collect();
                let mean = sizes.iter().sum::<f32>() / sizes.len() as f32;
                let variance = sizes.iter()
                    .map(|&s| (s - mean).powi(2))
                    .sum::<f32>() / sizes.len() as f32;
                balance_scores.push(1.0 / (1.0 + variance.sqrt()));
            }
        }
        
        if balance_scores.is_empty() {
            0.5
        } else {
            balance_scores.iter().sum::<f32>() / balance_scores.len() as f32
        }
    }
    
    /// Compute Hopf-algebraic score
    fn compute_hopf_score(&self, tree: &Tree) -> f32 {
        let mut score = 0.0;
        
        // Number of admissible cuts
        let cuts = self.hopf.admissible_cuts(tree);
        score += (cuts.len() as f32).ln();
        
        // Antipode complexity
        let antipode = self.hopf.antipode(tree);
        score += (antipode.size() as f32) / (tree.size() as f32);
        
        // Coproduct richness
        let coproduct = self.hopf.coproduct(tree);
        score += coproduct.terms().count() as f32 * 0.1;
        
        score
    }
    
    /// Compute structural diversity
    fn compute_diversity(&self, tree: &Tree) -> f32 {
        if tree.size() <= 1 {
            return 0.0;
        }
        
        // Variety in node degrees
        let degrees: Vec<usize> = (0..tree.size())
            .map(|n| tree.node_degree(n))
            .collect();
        let unique_degrees = degrees.iter().collect::<std::collections::HashSet<_>>().len();
        
        // Variety in subtree sizes
        let subtree_sizes: Vec<usize> = (0..tree.size())
            .map(|n| tree.subtree_size(n))
            .collect();
        let unique_sizes = subtree_sizes.iter().collect::<std::collections::HashSet<_>>().len();
        
        let diversity = (unique_degrees as f32 + unique_sizes as f32) / (2.0 * tree.size() as f32);
        diversity.min(1.0)
    }
}

/// Simple policy network for tree generation
pub struct TreePolicy {
    /// Hidden dimension
    hidden_dim: usize,
    /// Learning rate
    learning_rate: f32,
}

impl TreePolicy {
    /// Create new policy
    pub fn new(hidden_dim: usize, learning_rate: f32) -> Self {
        TreePolicy {
            hidden_dim,
            learning_rate,
        }
    }
    
    /// Select action given state
    pub fn select_action(&self, state: &TreeState) -> TreeAction {
        let valid_actions = state.valid_actions();
        let features = state.features();
        
        // Simple heuristic policy (would use neural network in practice)
        if state.tree.size() >= state.max_size * 8 / 10 {
            // Close to max size, consider stopping
            if rand::random::<f32>() < 0.3 {
                return TreeAction::Stop;
            }
        }
        
        // Prefer adding to nodes with fewer children
        let mut best_action = TreeAction::Stop;
        let mut best_score = f32::NEG_INFINITY;
        
        for action in valid_actions {
            let score = match action {
                TreeAction::AddChild(node) => {
                    let degree = state.tree.node_degree(node) as f32;
                    let depth = state.tree.node_depth(node) as f32;
                    // Prefer nodes with fewer children and moderate depth
                    10.0 - degree - (depth - 2.0).abs()
                }
                TreeAction::Stop => {
                    // Score for stopping
                    if state.tree.size() >= 3 {
                        5.0
                    } else {
                        -10.0
                    }
                }
            };
            
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
        
        best_action
    }
}

/// Episode data for training
#[derive(Debug, Clone)]
pub struct Episode {
    /// States visited
    pub states: Vec<TreeState>,
    /// Actions taken
    pub actions: Vec<TreeAction>,
    /// Rewards received
    pub rewards: Vec<f32>,
}

impl Episode {
    /// Total return
    pub fn total_return(&self) -> f32 {
        self.rewards.iter().sum()
    }
    
    /// Discounted return from step t
    pub fn discounted_return(&self, t: usize, gamma: f32) -> f32 {
        self.rewards[t..]
            .iter()
            .enumerate()
            .map(|(i, &r)| r * gamma.powi(i as i32))
            .sum()
    }
}

/// Generate episode using policy
pub fn generate_episode(
    policy: &TreePolicy,
    env: &TreeEnvironment,
    max_size: usize,
) -> Episode {
    let mut state = TreeState::new(max_size);
    let mut states = vec![state.clone()];
    let mut actions = Vec::new();
    let mut rewards = Vec::new();
    
    while !state.is_terminal() {
        let action = policy.select_action(&state);
        let next_state = state.apply_action(action);
        let reward = env.reward(&state, &action, &next_state);
        
        actions.push(action);
        rewards.push(reward);
        states.push(next_state.clone());
        
        state = next_state;
    }
    
    Episode { states, actions, rewards }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tree_state() {
        let state = TreeState::new(5);
        assert_eq!(state.tree.size(), 1);
        assert!(!state.is_terminal());
        
        let actions = state.valid_actions();
        assert!(actions.contains(&TreeAction::Stop));
        assert!(actions.contains(&TreeAction::AddChild(0)));
    }
    
    #[test]
    fn test_action_application() {
        let state = TreeState::new(5);
        let next_state = state.apply_action(TreeAction::AddChild(0));
        
        assert_eq!(next_state.tree.size(), 2);
        assert_eq!(next_state.tree.children(0).len(), 1);
    }
    
    #[test]
    fn test_episode_generation() {
        let hopf = Arc::new(HopfAlgebra::new(100));
        let env = TreeEnvironment::new(hopf).with_targets(Some(5), Some(3));
        let policy = TreePolicy::new(128, 0.01);
        
        let episode = generate_episode(&policy, &env, 10);
        assert!(!episode.states.is_empty());
        assert!(!episode.actions.is_empty());
        assert_eq!(episode.states.len(), episode.actions.len() + 1);
    }
} 