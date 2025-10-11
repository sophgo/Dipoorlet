import numpy as np
import torch
from onnx import numpy_helper

from ..quantize import get_qnode_by_param

LEARNABLE_LAYER_TYPES = ['Conv', 'Gemm', 'ConvTranspose', 'MatMul']
__all__ = ['LEARNABLE_LAYER_TYPES', 'follow_relu', 'following_relu',
           'update_weight', 'get_quant_tensor', 'get_block_from_first',
           'follow_nolinear', 'following_nolinear', 'follow_bias_after_matmul']


def follow_bias_after_matmul(graph, node):
    node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    return len(nxt_node) == 1 and not isinstance(nxt_node[0], str) and nxt_node[0].op_type == 'Add' and \
            (nxt_node[0].input[1] in graph.initializer or nxt_node[0].input[0] in graph.initializer)

def follow_relu(graph, node):
    node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    return len(nxt_node) == 1 and not isinstance(nxt_node[0], str) and nxt_node[0].op_type == 'Relu'

def follow_silu(graph, node):
    node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    if len(nxt_node) == 1 and not isinstance(nxt_node[0], str) and nxt_node[0].op_type == 'SiLU':
        return True
    if len(nxt_node) == 2:
        mul_node = None
        sigmoid_node = None
        if nxt_node[0].op_type == 'Mul' and nxt_node[1].op_type == 'Sigmoid':
            mul_node = nxt_node[0]
            sigmoid_node = nxt_node[1]
        if nxt_node[1].op_type == 'Mul' and nxt_node[0].op_type == 'Sigmoid':
            mul_node = nxt_node[1]
            sigmoid_node = nxt_node[0]
        if mul_node is not None and sigmoid_node is not None:
            if sigmoid_node.output[0] in mul_node.input:
                return True
    return False

def follow_gelu(graph, node):
    if node.op_type == 'MatMul' and follow_bias_after_matmul(graph, node):
        node_out = graph.get_tensor_consumer(node.output[0])[0].output[0]
    else:
        node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    return len(nxt_node) == 1 and not isinstance(nxt_node[0], str) and nxt_node[0].op_type == 'Gelu'

def follow_nolinear(graph, node):
    nolinear_type = None
    if follow_relu(graph, node):
        nolinear_type = 'relu'
    elif follow_silu(graph, node):
        nolinear_type = 'silu'
    elif follow_gelu(graph, node):
        nolinear_type = 'gelu'
    return nolinear_type

def following_relu(graph, node):
    node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    assert nxt_node[0].op_type == 'Relu'
    return nxt_node[0]

def following_silu(graph, node):
    node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    if len(nxt_node) == 1 and not isinstance(nxt_node[0], str) and nxt_node[0].op_type == 'SiLU':
        return nxt_node[0]
    if len(nxt_node) == 2:
        if nxt_node[0].op_type == 'Mul' and nxt_node[1].op_type == 'Sigmoid':
            return nxt_node[0]
        if nxt_node[1].op_type == 'Mul' and nxt_node[0].op_type == 'Sigmoid':
            return nxt_node[1]
    raise ValueError('No following silu node found.')

def following_gelu(graph, node):
    if node.op_type == 'MatMul' and follow_bias_after_matmul(graph, node):
        node_out = graph.get_tensor_consumer(node.output[0])[0].output[0]
    else:
        node_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(node_out)
    assert nxt_node[0].op_type == 'Gelu'
    return nxt_node[0]

def following_nolinear(graph, node, nolinear_type):
    following_node = node
    if nolinear_type == 'relu':
        following_node = following_relu(graph, node)
    elif nolinear_type == 'silu':
        following_node = following_silu(graph, node)
    elif nolinear_type == 'gelu':
        following_node = following_gelu(graph, node)
    return following_node

def update_weight(graph, weight_tensor, weight_name):
    name = graph.initializer[weight_name][0].name
    graph.set_initializer(name, weight_tensor)


def get_quant_tensor(weight_shape, param, weight_range, signed=True):
    q_nodes, q_min, q_max = get_qnode_by_param(param, 'tmp', weight_shape, weight_range, signed=signed)
    scale = None
    for init in q_nodes.initializer:
        if init.name == 'tmp_scale':
            scale = numpy_helper.to_array(init)

    if 'per_channel' in param and param['per_channel']:
        c_num = weight_shape[0]
        scale = torch.from_numpy(np.array(scale).astype(np.float32)).view(
            [c_num, *[1] * (len(weight_shape) - 1)]).cuda()
        q_min = torch.from_numpy(np.array(q_min).astype(np.float32)).view(
            [c_num, *[1] * (len(weight_shape) - 1)]).cuda()
        q_max = torch.from_numpy(np.array(q_max).astype(np.float32)).view(
            [c_num, *[1] * (len(weight_shape) - 1)]).cuda()
    else:
        scale = torch.from_numpy(np.array(scale).astype(np.float32)).cuda()
        q_min = torch.from_numpy(np.array(q_min).astype(np.float32)).cuda()
        q_max = torch.from_numpy(np.array(q_max).astype(np.float32)).cuda()
    scale.requires_grad = False
    q_min.requires_grad = False
    q_max.requires_grad = False
    return scale, q_min, q_max


def get_block_from_first(graph, node, args):
    res = [node]
    while True:
        if follow_silu(graph, res[-1]):
            node = following_silu(graph, res[-1])
        elif follow_gelu(graph, res[-1]):
            node = following_gelu(graph, res[-1])
        next_node = graph.get_tensor_consumer(node.output[0])
        if len(next_node) != 1 or isinstance(next_node[0], str) or next_node[0].op_type not in LEARNABLE_LAYER_TYPES + ['Relu']:
            return res
        if next_node[0].op_type != 'Relu':
            res.append(next_node[0])
            # We set max len=3.
            if len(res) == 3:
                return res
        node = next_node[0]
