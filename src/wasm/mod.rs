//! WebAssembly interface for interactive Hopf algebra visualization

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
use crate::algebra::{Tree, Forest, CoProduct};
use crate::graph::GraphData;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct HopfVisualizer {
    /// Current tree being visualized
    current_tree: Option<Tree>,
    /// Graph representation
    graph_data: Option<GraphData>,
    /// Animation state
    animation_state: AnimationState,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
struct AnimationState {
    /// Current frame
    frame: u32,
    /// Total frames
    total_frames: u32,
    /// Animation type
    animation_type: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl HopfVisualizer {
    /// Create new visualizer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        HopfVisualizer {
            current_tree: None,
            graph_data: None,
            animation_state: AnimationState {
                frame: 0,
                total_frames: 60,
                animation_type: "none".to_string(),
            },
        }
    }

    /// Load tree from JSON
    #[wasm_bindgen]
    pub fn load_tree(&mut self, json: &str) -> Result<(), JsValue> {
        match serde_json::from_str::<Tree>(json) {
            Ok(tree) => {
                self.current_tree = Some(tree.clone());
                self.graph_data = Some(tree.to_graph());
                Ok(())
            }
            Err(e) => Err(JsValue::from_str(&format!("Failed to parse tree: {}", e))),
        }
    }

    /// Get tree as JSON
    #[wasm_bindgen]
    pub fn get_tree_json(&self) -> Result<String, JsValue> {
        match &self.current_tree {
            Some(tree) => {
                serde_json::to_string(tree)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            None => Err(JsValue::from_str("No tree loaded")),
        }
    }

    /// Get graph data for visualization
    #[wasm_bindgen]
    pub fn get_graph_data(&self) -> Result<String, JsValue> {
        match &self.graph_data {
            Some(graph) => {
                // Convert to D3.js compatible format
                let nodes: Vec<NodeData> = (0..graph.node_count())
                    .map(|i| NodeData {
                        id: i,
                        label: format!("Node {}", i),
                        size: graph.node_features(i).map(|f| f[0] as f32).unwrap_or(1.0),
                    })
                    .collect();

                let edges: Vec<EdgeData> = graph.edges()
                    .map(|(src, tgt, weight)| EdgeData {
                        source: src,
                        target: tgt,
                        weight: *weight,
                    })
                    .collect();

                let viz_data = VisualizationData { nodes, edges };
                
                serde_json::to_string(&viz_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            None => Err(JsValue::from_str("No graph data available")),
        }
    }

    /// Compute and visualize coproduct
    #[wasm_bindgen]
    pub fn visualize_coproduct(&mut self) -> Result<String, JsValue> {
        match &self.current_tree {
            Some(tree) => {
                let cop = tree.coproduct();
                
                // Convert coproduct to visualization format
                let mut coproduct_data = Vec::new();
                
                for (i, (forest, trunk)) in cop.iter().enumerate() {
                    coproduct_data.push(CoproductTerm {
                        index: i,
                        forest_size: forest.trees().len(),
                        trunk_size: trunk.size(),
                        coefficient: 1, // Simplified
                    });
                }
                
                self.animation_state.animation_type = "coproduct".to_string();
                self.animation_state.frame = 0;
                
                serde_json::to_string(&coproduct_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            None => Err(JsValue::from_str("No tree loaded")),
        }
    }

    /// Compute and visualize antipode
    #[wasm_bindgen]
    pub fn visualize_antipode(&mut self) -> Result<String, JsValue> {
        match &self.current_tree {
            Some(tree) => {
                let antipode = tree.antipode();
                
                // Convert antipode to visualization format
                let antipode_data = AntipodeData {
                    input_size: tree.size(),
                    output_trees: antipode.trees().len(),
                    total_nodes: antipode.trees().iter().map(|t| t.size()).sum(),
                };
                
                self.animation_state.animation_type = "antipode".to_string();
                self.animation_state.frame = 0;
                
                serde_json::to_string(&antipode_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            None => Err(JsValue::from_str("No tree loaded")),
        }
    }

    /// Update animation frame
    #[wasm_bindgen]
    pub fn update_animation(&mut self) -> u32 {
        self.animation_state.frame = (self.animation_state.frame + 1) % self.animation_state.total_frames;
        self.animation_state.frame
    }

    /// Get current animation progress
    #[wasm_bindgen]
    pub fn get_animation_progress(&self) -> f32 {
        self.animation_state.frame as f32 / self.animation_state.total_frames as f32
    }
}

// Data structures for visualization
#[derive(serde::Serialize)]
struct NodeData {
    id: usize,
    label: String,
    size: f32,
}

#[derive(serde::Serialize)]
struct EdgeData {
    source: usize,
    target: usize,
    weight: f32,
}

#[derive(serde::Serialize)]
struct VisualizationData {
    nodes: Vec<NodeData>,
    edges: Vec<EdgeData>,
}

#[derive(serde::Serialize)]
struct CoproductTerm {
    index: usize,
    forest_size: usize,
    trunk_size: usize,
    coefficient: i32,
}

#[derive(serde::Serialize)]
struct AntipodeData {
    input_size: usize,
    output_trees: usize,
    total_nodes: usize,
}

/// Interactive tree builder for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct InteractiveTreeBuilder {
    builder: crate::algebra::TreeBuilder,
    history: Vec<TreeSnapshot>,
}

#[derive(Clone)]
struct TreeSnapshot {
    tree: Tree,
    operation: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl InteractiveTreeBuilder {
    /// Create new interactive builder
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        InteractiveTreeBuilder {
            builder: crate::algebra::TreeBuilder::new(),
            history: vec![],
        }
    }

    /// Add a child node
    #[wasm_bindgen]
    pub fn add_child(&mut self, parent: usize, child: usize) -> Result<(), JsValue> {
        self.builder.add_child(parent, child);
        
        // Try to build and save snapshot
        match self.builder.clone().build() {
            Ok(tree) => {
                self.history.push(TreeSnapshot {
                    tree,
                    operation: format!("Add child {} to parent {}", child, parent),
                });
                Ok(())
            }
            Err(e) => Err(JsValue::from_str(&format!("Invalid tree: {}", e))),
        }
    }

    /// Get current tree
    #[wasm_bindgen]
    pub fn get_current_tree(&self) -> Result<String, JsValue> {
        match self.builder.clone().build() {
            Ok(tree) => {
                serde_json::to_string(&tree)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            Err(e) => Err(JsValue::from_str(&format!("Invalid tree: {}", e))),
        }
    }

    /// Undo last operation
    #[wasm_bindgen]
    pub fn undo(&mut self) -> Result<(), JsValue> {
        if self.history.len() > 1 {
            self.history.pop();
            // Rebuild from history
            self.rebuild_from_history();
            Ok(())
        } else {
            Err(JsValue::from_str("Nothing to undo"))
        }
    }

    /// Get operation history
    #[wasm_bindgen]
    pub fn get_history(&self) -> Result<String, JsValue> {
        let history_data: Vec<String> = self.history
            .iter()
            .map(|snapshot| snapshot.operation.clone())
            .collect();
        
        serde_json::to_string(&history_data)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    fn rebuild_from_history(&mut self) {
        // This is a simplified rebuild - in practice would reconstruct exact builder state
        if let Some(last) = self.history.last() {
            // For now, we can't perfectly reconstruct the builder
            // This is a limitation of the current design
        }
    }
}

/// Create JavaScript bindings for the library
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn init() {
    // Set panic hook for better error messages in WASM
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wasm_types() {
        // Basic compile test for WASM types
        let _node = NodeData {
            id: 0,
            label: "test".to_string(),
            size: 1.0,
        };
        
        let _edge = EdgeData {
            source: 0,
            target: 1,
            weight: 1.0,
        };
    }
} 