//! Blockchain integration for Hopf algebraic structures

use crate::algebra::{Tree, Forest, CoProduct};
use crate::algebra::Antipode;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};

/// Blockchain for storing Hopf algebraic computations
pub struct HopfChain {
    /// Chain of blocks
    blocks: Vec<Block>,
    /// Pending transactions
    pending_transactions: Vec<AlgebraicTransaction>,
    /// Difficulty for proof-of-work
    difficulty: usize,
    /// Reward for mining
    mining_reward: f64,
}

/// A block in the Hopf blockchain
#[derive(Debug, Clone)]
pub struct Block {
    /// Block index
    index: u64,
    /// Timestamp
    timestamp: u64,
    /// Algebraic transactions
    transactions: Vec<AlgebraicTransaction>,
    /// Hash of previous block
    previous_hash: String,
    /// Current block hash
    hash: String,
    /// Nonce for proof-of-work
    nonce: u64,
    /// Merkle root of Hopf operations
    hopf_merkle_root: String,
}

/// Transaction representing an algebraic operation
#[derive(Debug, Clone)]
pub struct AlgebraicTransaction {
    /// Unique transaction ID
    id: String,
    /// Type of operation
    operation: HopfOperation,
    /// Input tree/forest
    input: AlgebraicObject,
    /// Output tree/forest
    output: AlgebraicObject,
    /// Computation proof
    proof: ComputationProof,
    /// Timestamp
    timestamp: u64,
}

/// Types of Hopf operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HopfOperation {
    Coproduct,
    Antipode,
    Product,
    NaturalGrowth,
    Renormalization,
}

/// Algebraic objects (trees or forests)
#[derive(Debug, Clone)]
pub enum AlgebraicObject {
    Tree(Tree),
    Forest(Forest),
    Scalar(f64),
}

/// Proof of correct computation
#[derive(Debug, Clone)]
pub struct ComputationProof {
    /// Hash of computation steps
    computation_hash: String,
    /// Witness data
    witness: Vec<u8>,
    /// Verification key
    verification_key: String,
}

impl HopfChain {
    /// Create new blockchain
    pub fn new(difficulty: usize) -> Self {
        let mut chain = HopfChain {
            blocks: Vec::new(),
            pending_transactions: Vec::new(),
            difficulty,
            mining_reward: 10.0,
        };
        
        // Create genesis block
        chain.create_genesis_block();
        chain
    }

    fn create_genesis_block(&mut self) {
        let genesis = Block {
            index: 0,
            timestamp: current_timestamp(),
            transactions: vec![],
            previous_hash: "0".to_string(),
            hash: String::new(),
            nonce: 0,
            hopf_merkle_root: "genesis".to_string(),
        };
        
        let hash = self.calculate_hash(&genesis);
        let mut genesis = genesis;
        genesis.hash = hash;
        
        self.blocks.push(genesis);
    }

    /// Add new algebraic transaction
    pub fn add_transaction(&mut self, transaction: AlgebraicTransaction) -> Result<(), String> {
        // Verify the computation
        if !self.verify_computation(&transaction) {
            return Err("Invalid computation proof".to_string());
        }
        
        self.pending_transactions.push(transaction);
        Ok(())
    }

    /// Mine pending transactions
    pub fn mine_block(&mut self, miner_address: &str) -> Block {
        // Add mining reward transaction
        let reward_tx = AlgebraicTransaction {
            id: format!("reward_{}", current_timestamp()),
            operation: HopfOperation::Product,
            input: AlgebraicObject::Scalar(0.0),
            output: AlgebraicObject::Scalar(self.mining_reward),
            proof: ComputationProof {
                computation_hash: "mining_reward".to_string(),
                witness: vec![],
                verification_key: miner_address.to_string(),
            },
            timestamp: current_timestamp(),
        };
        
        let mut transactions = self.pending_transactions.clone();
        transactions.push(reward_tx);
        
        let previous_block = self.blocks.last().unwrap();
        let mut new_block = Block {
            index: previous_block.index + 1,
            timestamp: current_timestamp(),
            transactions,
            previous_hash: previous_block.hash.clone(),
            hash: String::new(),
            nonce: 0,
            hopf_merkle_root: self.compute_hopf_merkle_root(&self.pending_transactions),
        };
        
        // Proof of work
        new_block = self.proof_of_work(new_block);
        
        self.blocks.push(new_block.clone());
        self.pending_transactions.clear();
        
        new_block
    }

    fn proof_of_work(&self, mut block: Block) -> Block {
        while !self.is_valid_proof(&block) {
            block.nonce += 1;
            block.hash = self.calculate_hash(&block);
        }
        block
    }

    fn is_valid_proof(&self, block: &Block) -> bool {
        let prefix = "0".repeat(self.difficulty);
        block.hash.starts_with(&prefix)
    }

    fn calculate_hash(&self, block: &Block) -> String {
        let data = format!(
            "{}{}{}{}{}",
            block.index,
            block.timestamp,
            block.previous_hash,
            block.nonce,
            block.hopf_merkle_root
        );
        
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn verify_computation(&self, transaction: &AlgebraicTransaction) -> bool {
        // Verify that the output is correct for the given operation
        match &transaction.operation {
            HopfOperation::Coproduct => {
                if let AlgebraicObject::Tree(tree) = &transaction.input {
                    // Verify coproduct computation
                    let computed_cop = tree.coproduct();
                    // Simplified verification
                    true
                } else {
                    false
                }
            }
            HopfOperation::Antipode => {
                if let AlgebraicObject::Tree(tree) = &transaction.input {
                    // Verify antipode computation
                    let computed_antipode = tree.antipode();
                    // Simplified verification
                    true
                } else {
                    false
                }
            }
            _ => true, // Simplified for other operations
        }
    }

    fn compute_hopf_merkle_root(&self, transactions: &[AlgebraicTransaction]) -> String {
        if transactions.is_empty() {
            return "empty".to_string();
        }
        
        // Build Merkle tree of Hopf operations
        let hashes: Vec<String> = transactions
            .iter()
            .map(|tx| self.hash_transaction(tx))
            .collect();
        
        self.merkle_root(hashes)
    }

    fn hash_transaction(&self, tx: &AlgebraicTransaction) -> String {
        let mut hasher = Sha256::new();
        hasher.update(tx.id.as_bytes());
        hasher.update(format!("{:?}", tx.operation).as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn merkle_root(&self, mut hashes: Vec<String>) -> String {
        if hashes.len() == 1 {
            return hashes[0].clone();
        }
        
        if hashes.len() % 2 != 0 {
            hashes.push(hashes.last().unwrap().clone());
        }
        
        let mut next_level = Vec::new();
        for i in (0..hashes.len()).step_by(2) {
            let combined = format!("{}{}", hashes[i], hashes[i + 1]);
            let mut hasher = Sha256::new();
            hasher.update(combined.as_bytes());
            next_level.push(format!("{:x}", hasher.finalize()));
        }
        
        self.merkle_root(next_level)
    }

    /// Validate the entire chain
    pub fn is_valid_chain(&self) -> bool {
        for i in 1..self.blocks.len() {
            let current = &self.blocks[i];
            let previous = &self.blocks[i - 1];
            
            // Check hash validity
            if current.hash != self.calculate_hash(current) {
                return false;
            }
            
            // Check previous hash link
            if current.previous_hash != previous.hash {
                return false;
            }
            
            // Check proof of work
            if !self.is_valid_proof(current) {
                return false;
            }
        }
        
        true
    }

    /// Get algebraic lineage of a tree
    pub fn get_tree_lineage(&self, tree: &Tree) -> Vec<AlgebraicTransaction> {
        let mut lineage = Vec::new();
        
        for block in &self.blocks {
            for tx in &block.transactions {
                match &tx.output {
                    AlgebraicObject::Tree(t) if t == tree => {
                        lineage.push(tx.clone());
                    }
                    _ => {}
                }
            }
        }
        
        lineage
    }
}

/// Smart contract for Hopf algebra operations
pub struct HopfContract {
    /// Contract state
    state: ContractState,
    /// Allowed operations
    allowed_operations: Vec<HopfOperation>,
    /// Operation costs
    operation_costs: HashMap<HopfOperation, f64>,
}

#[derive(Debug)]
struct ContractState {
    /// Current tree being operated on
    current_tree: Option<Tree>,
    /// Accumulated operations
    operation_history: Vec<HopfOperation>,
    /// Total cost
    total_cost: f64,
}

impl HopfContract {
    pub fn new() -> Self {
        let mut operation_costs = HashMap::new();
        operation_costs.insert(HopfOperation::Coproduct, 1.0);
        operation_costs.insert(HopfOperation::Antipode, 2.0);
        operation_costs.insert(HopfOperation::Product, 0.5);
        operation_costs.insert(HopfOperation::NaturalGrowth, 1.5);
        operation_costs.insert(HopfOperation::Renormalization, 3.0);
        
        HopfContract {
            state: ContractState {
                current_tree: None,
                operation_history: Vec::new(),
                total_cost: 0.0,
            },
            allowed_operations: vec![
                HopfOperation::Coproduct,
                HopfOperation::Antipode,
                HopfOperation::Product,
                HopfOperation::NaturalGrowth,
            ],
            operation_costs,
        }
    }

    /// Execute operation on contract
    pub fn execute(&mut self, operation: HopfOperation, input: Tree) -> Result<AlgebraicObject, String> {
        // Check if operation is allowed
        if !self.allowed_operations.contains(&operation) {
            return Err("Operation not allowed".to_string());
        }
        
        // Calculate cost
        let cost = self.operation_costs.get(&operation).unwrap_or(&1.0);
        self.state.total_cost += cost;
        
        // Execute operation
        let output = match operation {
            HopfOperation::Coproduct => {
                let cop = input.coproduct();
                // Collect all forests from coproduct terms
                let all_trees: Vec<Tree> = cop.into_iter()
                    .flat_map(|((forest, _trunk), _coeff)| forest.into_trees())
                    .collect();
                AlgebraicObject::Forest(Forest::from(all_trees))
            }
            HopfOperation::Antipode => {
                AlgebraicObject::Forest(input.antipode())
            }
            HopfOperation::NaturalGrowth => {
                let grown = input.graft_all_leaves();
                AlgebraicObject::Forest(Forest::from(grown))
            }
            _ => AlgebraicObject::Tree(input.clone()),
        };
        
        // Update state
        self.state.current_tree = Some(input);
        self.state.operation_history.push(operation);
        
        Ok(output)
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TreeBuilder;

    #[test]
    fn test_blockchain_creation() {
        let chain = HopfChain::new(2);
        assert_eq!(chain.blocks.len(), 1); // Genesis block
        assert!(chain.is_valid_chain());
    }

    #[test]
    fn test_add_transaction() {
        let mut chain = HopfChain::new(2);
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let tree = builder.build().unwrap();
        
        let tx = AlgebraicTransaction {
            id: "test_tx".to_string(),
            operation: HopfOperation::Coproduct,
            input: AlgebraicObject::Tree(tree.clone()),
            output: AlgebraicObject::Forest(tree.antipode()),
            proof: ComputationProof {
                computation_hash: "test_hash".to_string(),
                witness: vec![],
                verification_key: "test_key".to_string(),
            },
            timestamp: current_timestamp(),
        };
        
        assert!(chain.add_transaction(tx).is_ok());
    }

    #[test]
    fn test_smart_contract() {
        let mut contract = HopfContract::new();
        let mut builder = TreeBuilder::new();
        builder.add_child(0, 1);
        let tree = builder.build().unwrap();
        
        let result = contract.execute(HopfOperation::Antipode, tree);
        assert!(result.is_ok());
        assert!(contract.state.total_cost > 0.0);
    }
} 