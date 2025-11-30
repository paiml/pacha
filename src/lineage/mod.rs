//! Model lineage tracking.
//!
//! Tracks how models are derived from other models through various operations.

use crate::model::ModelId;
use crate::recipe::RecipeId;
use serde::{Deserialize, Serialize};

/// Types of model lineage relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelLineageEdge {
    /// Model was fine-tuned from parent.
    FineTuned {
        /// Parent model ID.
        parent: ModelId,
        /// Recipe used for fine-tuning.
        recipe: RecipeId,
    },
    /// Model was distilled from teacher.
    Distilled {
        /// Teacher model ID.
        teacher: ModelId,
        /// Distillation temperature.
        temperature: f32,
    },
    /// Model was merged from multiple sources.
    Merged {
        /// Source model IDs.
        sources: Vec<ModelId>,
        /// Merge weights.
        weights: Vec<f32>,
    },
    /// Model was quantized from source.
    Quantized {
        /// Source model ID.
        source: ModelId,
        /// Quantization type.
        quantization: QuantizationType,
    },
    /// Model was pruned from source.
    Pruned {
        /// Source model ID.
        source: ModelId,
        /// Target sparsity (0.0 to 1.0).
        sparsity: f32,
    },
}

/// Types of quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization.
    Int4,
    /// 16-bit floating point.
    Fp16,
    /// Brain floating point 16.
    Bf16,
    /// Dynamic quantization.
    Dynamic,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Dynamic => "dynamic",
        };
        write!(f, "{s}")
    }
}

/// A node in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    /// Model ID.
    pub model_id: ModelId,
    /// Model name.
    pub model_name: String,
    /// Model version string.
    pub model_version: String,
}

/// A lineage graph showing model derivation history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineageGraph {
    /// Nodes in the graph.
    pub nodes: Vec<LineageNode>,
    /// Edges representing derivation relationships.
    pub edges: Vec<LineageEdgeRecord>,
}

/// A recorded edge in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdgeRecord {
    /// Source node index.
    pub from_idx: usize,
    /// Target node index.
    pub to_idx: usize,
    /// Edge type and metadata.
    pub edge: ModelLineageEdge,
}

impl LineageGraph {
    /// Create an empty lineage graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: LineageNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, from_idx: usize, to_idx: usize, edge: ModelLineageEdge) {
        self.edges.push(LineageEdgeRecord {
            from_idx,
            to_idx,
            edge,
        });
    }

    /// Get the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get ancestors of a node (models it was derived from).
    #[must_use]
    pub fn ancestors(&self, node_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.to_idx == node_idx)
            .map(|e| e.from_idx)
            .collect()
    }

    /// Get descendants of a node (models derived from it).
    #[must_use]
    pub fn descendants(&self, node_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.from_idx == node_idx)
            .map(|e| e.to_idx)
            .collect()
    }

    /// Find node index by model ID.
    #[must_use]
    pub fn find_node(&self, model_id: &ModelId) -> Option<usize> {
        self.nodes.iter().position(|n| &n.model_id == model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_display() {
        assert_eq!(QuantizationType::Int8.to_string(), "int8");
        assert_eq!(QuantizationType::Fp16.to_string(), "fp16");
    }

    #[test]
    fn test_lineage_graph_basic() {
        let mut graph = LineageGraph::new();

        let base_id = ModelId::new();
        let finetuned_id = ModelId::new();

        let base_idx = graph.add_node(LineageNode {
            model_id: base_id.clone(),
            model_name: "base-model".to_string(),
            model_version: "1.0.0".to_string(),
        });

        let finetuned_idx = graph.add_node(LineageNode {
            model_id: finetuned_id.clone(),
            model_name: "finetuned-model".to_string(),
            model_version: "1.0.0".to_string(),
        });

        graph.add_edge(
            base_idx,
            finetuned_idx,
            ModelLineageEdge::FineTuned {
                parent: base_id.clone(),
                recipe: RecipeId::new(),
            },
        );

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.ancestors(finetuned_idx), vec![base_idx]);
        assert_eq!(graph.descendants(base_idx), vec![finetuned_idx]);
    }

    #[test]
    fn test_lineage_graph_find_node() {
        let mut graph = LineageGraph::new();
        let model_id = ModelId::new();

        graph.add_node(LineageNode {
            model_id: model_id.clone(),
            model_name: "test-model".to_string(),
            model_version: "1.0.0".to_string(),
        });

        assert_eq!(graph.find_node(&model_id), Some(0));
        assert_eq!(graph.find_node(&ModelId::new()), None);
    }

    #[test]
    fn test_lineage_edge_serialization() {
        let edge = ModelLineageEdge::Quantized {
            source: ModelId::new(),
            quantization: QuantizationType::Int8,
        };

        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("quantized"));
        assert!(json.contains("int8"));

        let deserialized: ModelLineageEdge = serde_json::from_str(&json).unwrap();
        if let ModelLineageEdge::Quantized { quantization, .. } = deserialized {
            assert_eq!(quantization, QuantizationType::Int8);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_merged_lineage() {
        let sources = vec![ModelId::new(), ModelId::new(), ModelId::new()];
        let weights = vec![0.5, 0.3, 0.2];

        let edge = ModelLineageEdge::Merged {
            sources: sources.clone(),
            weights: weights.clone(),
        };

        let json = serde_json::to_string(&edge).unwrap();
        let deserialized: ModelLineageEdge = serde_json::from_str(&json).unwrap();

        if let ModelLineageEdge::Merged {
            sources: s,
            weights: w,
        } = deserialized
        {
            assert_eq!(s.len(), 3);
            assert_eq!(w.len(), 3);
        } else {
            panic!("Wrong variant");
        }
    }
}
