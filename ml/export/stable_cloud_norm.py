"""Stable normalization arithmetic for the COC legacy ONNX export.

The graph retains FP32 weights/I/O/convolutions and uses FP64 centered moments
inside normalization. This precision choice is explicit in its bundle contract.
It preserves y = scale * (x - mean) / sqrt(variance + epsilon) + bias.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def stabilize_normalization(path, expected_count):
    graph = onnx.load(path)
    if next(x.version for x in graph.opset_import if x.domain == "") != 17:
        raise ValueError("Stable COC normalization rewrite expects opset 17")
    inferred = onnx.shape_inference.infer_shapes(graph)
    ranks = {value.name: len(value.type.tensor_type.shape.dim)
             for value in list(inferred.graph.value_info) + list(inferred.graph.input)}
    nodes, count = [], 0
    for index, original in enumerate(graph.graph.node):
        if original.op_type != "InstanceNormalization":
            nodes.append(original)
            continue
        if ranks.get(original.input[0]) != 3:
            raise ValueError("Expected reshaped [batch, groups, flattened_group] normalization")
        count += 1
        prefix = f"coc_centered_norm64_{index}"
        def node(op, inputs, suffix, **attributes):
            output = prefix + "_" + suffix
            nodes.append(helper.make_node(op, inputs, [output], name=output, **attributes))
            return output
        x = node("Cast", [original.input[0]], "input", to=TensorProto.DOUBLE)
        mean = node("ReduceMean", [x], "mean", axes=[2], keepdims=1)
        centered = node("Sub", [x, mean], "centered")
        squared = node("Mul", [centered, centered], "square")
        variance = node("ReduceMean", [squared], "variance", axes=[2], keepdims=1)
        epsilon = next((a.f for a in original.attribute if a.name == "epsilon"), 1e-5)
        eps = node("Constant", [], "epsilon", value=numpy_helper.from_array(np.array(epsilon, dtype=np.float64)))
        denominator = node("Sqrt", [node("Add", [variance, eps], "variance_epsilon")], "denominator")
        normalized = node("Div", [centered, denominator], "normalized")
        shape = node("Constant", [], "weight_shape", value=numpy_helper.from_array(np.array([1, -1, 1], dtype=np.int64)))
        scale = node("Reshape", [node("Cast", [original.input[1]], "scale64", to=TensorProto.DOUBLE), shape], "scale")
        bias = node("Reshape", [node("Cast", [original.input[2]], "bias64", to=TensorProto.DOUBLE), shape], "bias")
        affine = node("Add", [node("Mul", [normalized, scale], "scaled"), bias], "affine")
        nodes.append(helper.make_node("Cast", [affine], list(original.output), to=TensorProto.FLOAT, name=prefix + "_output"))
    if count != expected_count:
        raise ValueError("Exported normalization count differs from the trained architecture")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    return {"rewritten_nodes": count, "normalization_arithmetic": "FP64 centered moments", "weights_and_io": "FP32"}
