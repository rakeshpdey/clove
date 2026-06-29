/*
 * src/export.rs
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Rakesh Pradip Dey
 *
 * Licensed under the MIT License <LICENSE-MIT or http://opensource.org/licenses/MIT>.
 * 
 * Note: Portions of this software are adapted from existing open-source frameworks.
 * This file may not be copied, modified, or distributed except according to the terms
 * of the MIT license.
 */

//! ONNX Exporter for the Clove Engine.
//!
//! This module provides the capability to serialize the internal computation
//! graph into the ONNX (Open Neural Network Exchange) binary format. This
//! enables deployment of trained models to diverse inference environments
//! such as TensorRT, CoreML, and standard ONNX Runtimes.

use prost::Message;
use std::fs::File;
use std::io::Write;
use crate::backend::{ComputeGraph, Opcode};

// ONNX PROTOBUF DEFINITIONS 
// These structs provide a compact binary representation for ONNX compatibility.
// They are manually defined to remain lightweight and avoid complex dependencies.

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag="1")]
    pub ir_version: i64,
    #[prost(string, tag="2")]
    pub producer_name: String,
    #[prost(message, optional, tag="7")]
    pub graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag="1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag="2")]
    pub name: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag="1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag="2")]
    pub output: Vec<String>,
    #[prost(string, tag="3")]
    pub op_type: String,
}

// THE EXPORTER ENGINE

/// Handles the translation of internal `ComputeGraph` operations to standard ONNX 
/// operator definitions.
pub struct ONNXExporter;

impl ONNXExporter {
    /// Serializes the graph to an `.onnx` file.
    ///
    /// # Arguments
    /// * `graph` - The internal compute graph to be translated.
    /// * `filepath` - Destination file path for the exported model.
    pub fn export(graph: &ComputeGraph, filepath: &str) -> std::io::Result<()> {
        let mut nodes = Vec::new();

        for node in &graph.nodes {
            // Translate Clove Opcodes to official ONNX standard operators.
            // Note: Some custom kernels (like PagedAttention) do not map directly 
            // to ONNX and will trigger warnings upon export.
            let op_type = match node.op {
                Opcode::MatMul => "MatMul",
                Opcode::Add => "Add",
                Opcode::Sub => "Sub",
                Opcode::Mul => "Mul",
                Opcode::ReLU => "Relu",
                Opcode::Softmax => "Softmax",
                Opcode::Transpose => "Transpose",
                Opcode::Flatten => "Flatten",
                Opcode::Concat => "Concat",
                Opcode::Conv2d => "Conv",
                Opcode::MaxPool2d(_) => "MaxPool",
                Opcode::AvgPool2d(_) => "AveragePool",
                Opcode::LayerNorm => "LayerNormalization",
                Opcode::BatchNorm => "BatchNormalization",
                Opcode::Sigmoid => "Sigmoid",
                Opcode::Tanh => "Tanh",
                Opcode::TopK(_) => "TopK",
                Opcode::PagedAttention => {
                    println!("WARNING: PagedAttention is a hardware-specific virtual memory kernel and cannot be exported to ONNX. This node will be exported as 'UnknownOp'.");
                    "UnknownOp"
                },
                // Handled implicitly as graph inputs
                Opcode::Input => continue,
                _ => "UnknownOp", // fallback
            };

            let inputs: Vec<String> = node.dependencies.iter().map(|id| format!("tensor_{}", id)).collect();
            let outputs = vec![format!("tensor_{}", node.id)];

            nodes.push(NodeProto {
                input: inputs,
                output: outputs,
                op_type: op_type.to_string(),
            });
        }

        let onnx_graph = GraphProto {
            node: nodes,
            name: "Organon_Exported_Model".to_string(),
        };

        let model = ModelProto {
            // Standard ONNX IR version
            ir_version: 8,
            producer_name: "Clove Framework".to_string(),
            graph: Some(onnx_graph),
        };

        // Serialize to binary Protobuf
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();

        let mut file = File::create(filepath)?;
        file.write_all(&buf)?;
        
        println!("Model successfully exported to ONNX format at: {}", filepath);
        Ok(())
    }
}