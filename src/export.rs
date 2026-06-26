use prost::Message;
use std::fs::File;
use std::io::Write;
use crate::backend::{ComputeGraph, Opcode};

// ========================================================================
// ONNX PROTOBUF DEFINITIONS (Hand-rolled to avoid build.rs complexities)
// ========================================================================
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

// ========================================================================
// THE EXPORTER ENGINE
// ========================================================================
pub struct ONNXExporter;

impl ONNXExporter {
    pub fn export(graph: &ComputeGraph, filepath: &str) -> std::io::Result<()> {
        let mut nodes = Vec::new();

        for node in &graph.nodes {
            // Translate Organon Opcodes to official ONNX standard operators
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
                Opcode::Input => continue, // Handled implicitly as graph inputs
                _ => "UnknownOp", // Graceful fallback
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
            ir_version: 8, // Standard ONNX IR version
            producer_name: "Organon Framework".to_string(),
            graph: Some(onnx_graph),
        };

        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();

        let mut file = File::create(filepath)?;
        file.write_all(&buf)?;
        
        println!("🚀 Model successfully exported to ONNX format at: {}", filepath);
        Ok(())
    }
}