//! Molecular structure conversion to trees

use crate::algebra::{Tree, TreeBuilder};
use crate::Result;
use std::collections::{HashMap, HashSet};

/// Atom types in molecules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomType {
    Carbon,
    Hydrogen,
    Oxygen,
    Nitrogen,
    Sulfur,
    Phosphorus,
    Halogen(HalogenType),
    Other(u8), // Atomic number
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HalogenType {
    Fluorine,
    Chlorine,
    Bromine,
    Iodine,
}

/// Bond types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

/// Molecular atom
#[derive(Debug, Clone)]
pub struct Atom {
    pub atom_type: AtomType,
    pub index: usize,
    pub charge: i8,
    pub hybridization: Hybridization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hybridization {
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    Unknown,
}

/// Molecular bond
#[derive(Debug, Clone)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: BondType,
}

/// Molecule representation
#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub name: Option<String>,
}

impl Molecule {
    /// Create new molecule
    pub fn new() -> Self {
        Molecule {
            atoms: Vec::new(),
            bonds: Vec::new(),
            name: None,
        }
    }
    
    /// Add atom
    pub fn add_atom(&mut self, atom_type: AtomType) -> usize {
        let index = self.atoms.len();
        self.atoms.push(Atom {
            atom_type,
            index,
            charge: 0,
            hybridization: Hybridization::Unknown,
        });
        index
    }
    
    /// Add bond
    pub fn add_bond(&mut self, atom1: usize, atom2: usize, bond_type: BondType) {
        self.bonds.push(Bond {
            atom1,
            atom2,
            bond_type,
        });
    }
    
    /// Convert to tree using BFS from a root atom
    pub fn to_tree(&self, root_atom: usize) -> Result<Tree> {
        if root_atom >= self.atoms.len() {
            return Err(crate::HopfMLError::AlgebraError(
                "Root atom index out of bounds".to_string()
            ));
        }
        
        // Build adjacency list
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for bond in &self.bonds {
            adjacency.entry(bond.atom1).or_insert_with(Vec::new).push(bond.atom2);
            adjacency.entry(bond.atom2).or_insert_with(Vec::new).push(bond.atom1);
        }
        
        // BFS to build tree
        let mut builder = TreeBuilder::new();
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut atom_to_node = HashMap::new();
        
        // Map atoms to tree nodes
        atom_to_node.insert(root_atom, 0);
        visited.insert(root_atom);
        queue.push_back((root_atom, 0));
        
        let mut node_counter = 1;
        
        while let Some((atom_idx, parent_node)) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&atom_idx) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        atom_to_node.insert(neighbor, node_counter);
                        builder.add_child(parent_node, node_counter);
                        queue.push_back((neighbor, node_counter));
                        node_counter += 1;
                    }
                }
            }
        }
        
        builder.build()
    }
    
    /// Convert to tree using chemical principles (functional groups as subtrees)
    pub fn to_functional_tree(&self) -> Result<Tree> {
        // Identify functional groups
        let groups = self.identify_functional_groups();
        
        // Build tree with functional groups as subtrees
        let mut builder = TreeBuilder::new();
        let mut node_counter = 1;
        
        // Create nodes for each functional group
        for (group_idx, group) in groups.iter().enumerate() {
            if group_idx > 0 {
                builder.add_child(0, node_counter);
                
                // Add atoms in the group as children
                for &atom_idx in &group.atoms {
                    node_counter += 1;
                    builder.add_child(node_counter - 1, node_counter);
                }
            }
        }
        
        builder.build()
    }
    
    /// Identify functional groups
    fn identify_functional_groups(&self) -> Vec<FunctionalGroup> {
        let mut groups = Vec::new();
        let mut assigned = vec![false; self.atoms.len()];
        
        // Look for common functional groups
        
        // Carbonyl groups (C=O)
        for bond in &self.bonds {
            if bond.bond_type == BondType::Double {
                let atom1 = &self.atoms[bond.atom1];
                let atom2 = &self.atoms[bond.atom2];
                
                if atom1.atom_type == AtomType::Carbon && atom2.atom_type == AtomType::Oxygen {
                    if !assigned[bond.atom1] && !assigned[bond.atom2] {
                        groups.push(FunctionalGroup {
                            group_type: FunctionalGroupType::Carbonyl,
                            atoms: vec![bond.atom1, bond.atom2],
                        });
                        assigned[bond.atom1] = true;
                        assigned[bond.atom2] = true;
                    }
                }
            }
        }
        
        // Hydroxyl groups (O-H)
        for (idx, atom) in self.atoms.iter().enumerate() {
            if atom.atom_type == AtomType::Oxygen && !assigned[idx] {
                // Check if bonded to hydrogen
                for bond in &self.bonds {
                    if bond.atom1 == idx || bond.atom2 == idx {
                        let other = if bond.atom1 == idx { bond.atom2 } else { bond.atom1 };
                        if self.atoms[other].atom_type == AtomType::Hydrogen {
                            groups.push(FunctionalGroup {
                                group_type: FunctionalGroupType::Hydroxyl,
                                atoms: vec![idx, other],
                            });
                            assigned[idx] = true;
                            assigned[other] = true;
                            break;
                        }
                    }
                }
            }
        }
        
        // Remaining atoms as alkyl groups
        let mut alkyl_atoms = Vec::new();
        for (idx, atom) in self.atoms.iter().enumerate() {
            if !assigned[idx] && atom.atom_type == AtomType::Carbon {
                alkyl_atoms.push(idx);
            }
        }
        
        if !alkyl_atoms.is_empty() {
            groups.push(FunctionalGroup {
                group_type: FunctionalGroupType::Alkyl,
                atoms: alkyl_atoms,
            });
        }
        
        groups
    }
}

/// Functional group types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionalGroupType {
    Alkyl,
    Carbonyl,
    Carboxyl,
    Hydroxyl,
    Amine,
    Aromatic,
    Other,
}

/// Functional group
#[derive(Debug, Clone)]
pub struct FunctionalGroup {
    pub group_type: FunctionalGroupType,
    pub atoms: Vec<usize>,
}

/// Create example molecules
pub mod examples {
    use super::*;
    
    /// Create methane (CH4)
    pub fn methane() -> Molecule {
        let mut mol = Molecule::new();
        let c = mol.add_atom(AtomType::Carbon);
        let h1 = mol.add_atom(AtomType::Hydrogen);
        let h2 = mol.add_atom(AtomType::Hydrogen);
        let h3 = mol.add_atom(AtomType::Hydrogen);
        let h4 = mol.add_atom(AtomType::Hydrogen);
        
        mol.add_bond(c, h1, BondType::Single);
        mol.add_bond(c, h2, BondType::Single);
        mol.add_bond(c, h3, BondType::Single);
        mol.add_bond(c, h4, BondType::Single);
        
        mol.name = Some("Methane".to_string());
        mol
    }
    
    /// Create ethanol (C2H5OH)
    pub fn ethanol() -> Molecule {
        let mut mol = Molecule::new();
        let c1 = mol.add_atom(AtomType::Carbon);
        let c2 = mol.add_atom(AtomType::Carbon);
        let o = mol.add_atom(AtomType::Oxygen);
        let h1 = mol.add_atom(AtomType::Hydrogen);
        let h2 = mol.add_atom(AtomType::Hydrogen);
        let h3 = mol.add_atom(AtomType::Hydrogen);
        let h4 = mol.add_atom(AtomType::Hydrogen);
        let h5 = mol.add_atom(AtomType::Hydrogen);
        let h6 = mol.add_atom(AtomType::Hydrogen);
        
        mol.add_bond(c1, c2, BondType::Single);
        mol.add_bond(c2, o, BondType::Single);
        mol.add_bond(o, h6, BondType::Single);
        mol.add_bond(c1, h1, BondType::Single);
        mol.add_bond(c1, h2, BondType::Single);
        mol.add_bond(c1, h3, BondType::Single);
        mol.add_bond(c2, h4, BondType::Single);
        mol.add_bond(c2, h5, BondType::Single);
        
        mol.name = Some("Ethanol".to_string());
        mol
    }
    
    /// Create benzene (C6H6)
    pub fn benzene() -> Molecule {
        let mut mol = Molecule::new();
        
        // Carbon atoms
        let carbons: Vec<usize> = (0..6)
            .map(|_| mol.add_atom(AtomType::Carbon))
            .collect();
            
        // Hydrogen atoms
        let hydrogens: Vec<usize> = (0..6)
            .map(|_| mol.add_atom(AtomType::Hydrogen))
            .collect();
        
        // Ring bonds (aromatic)
        for i in 0..6 {
            let next = (i + 1) % 6;
            mol.add_bond(carbons[i], carbons[next], BondType::Aromatic);
        }
        
        // C-H bonds
        for i in 0..6 {
            mol.add_bond(carbons[i], hydrogens[i], BondType::Single);
        }
        
        mol.name = Some("Benzene".to_string());
        mol
    }
}

/// SMILES parser (simplified)
pub fn parse_smiles(smiles: &str) -> Result<Molecule> {
    let mut mol = Molecule::new();
    let mut atom_stack = Vec::new();
    let mut chars = smiles.chars().peekable();
    
    while let Some(ch) = chars.next() {
        match ch {
            'C' => {
                let idx = mol.add_atom(AtomType::Carbon);
                if let Some(&last_atom) = atom_stack.last() {
                    mol.add_bond(last_atom, idx, BondType::Single);
                }
                atom_stack.push(idx);
            }
            'O' => {
                let idx = mol.add_atom(AtomType::Oxygen);
                if let Some(&last_atom) = atom_stack.last() {
                    mol.add_bond(last_atom, idx, BondType::Single);
                }
                atom_stack.push(idx);
            }
            'N' => {
                let idx = mol.add_atom(AtomType::Nitrogen);
                if let Some(&last_atom) = atom_stack.last() {
                    mol.add_bond(last_atom, idx, BondType::Single);
                }
                atom_stack.push(idx);
            }
            '(' => {
                // Branch start - continue with current atom
            }
            ')' => {
                // Branch end - pop back
                atom_stack.pop();
            }
            '=' => {
                // Double bond marker (simplified - modify last bond)
            }
            _ => {
                // Ignore other characters for now
            }
        }
    }
    
    Ok(mol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::examples::*;
    
    #[test]
    fn test_methane_to_tree() {
        let mol = methane();
        let tree = mol.to_tree(0).unwrap(); // Start from carbon
        assert_eq!(tree.size(), 5); // 1 carbon + 4 hydrogens
        assert_eq!(tree.children(0).len(), 4); // Carbon has 4 children
    }
    
    #[test]
    fn test_ethanol_to_tree() {
        let mol = ethanol();
        let tree = mol.to_tree(0).unwrap(); // Start from first carbon
        assert_eq!(tree.size(), 9); // All atoms
    }
    
    #[test]
    fn test_benzene_to_tree() {
        let mol = benzene();
        let tree = mol.to_tree(0).unwrap(); // Start from first carbon
        assert_eq!(tree.size(), 12); // 6 carbons + 6 hydrogens
    }
    
    #[test]
    fn test_simple_smiles() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol
        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.bonds.len(), 2);
    }
} 