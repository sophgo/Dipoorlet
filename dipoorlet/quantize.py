import copy

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .platform_settings import LAYER_HAS_WEIGHT, platform_setting_table
from .utils import ONNXGraph, logger

QTENSORSUFFIX = '_q'
DQTENSORSUFFIX = '_dq'
QNODESUFFIX = '_fake_quant'
DQNODESUFFIX = '_fake_dequant'
QUANT_NODE_NAME_LIST = ['QuantizeLinear', 'DequantizeLinear']
MERGE_RELU = ['Conv', 'Gemm', 'Eltwise', 'Add']
RELU_TYPE = ['Relu', 'PRelu', 'Mul']
WEIGHT_TRANSPOSE_SUFFIX = '_transpose'
CLIP_SUFFIX = '_clip'


def quant_graph(onnx_graph, clip_val, args):
    graph_q = ONNXGraph()
    graph_q.copy_from(onnx_graph)
    quant_node_list = []
    quant_node_list_w4a4 = []

    for node in graph_q.graph.node:
        if node.name in args.skip_layers:
            continue
        if 'w4a4' in platform_setting_table[args.deploy] and node.name in platform_setting_table[args.deploy]['w4a4']:
            quant_node_list_w4a4.append(node)
            continue
        if node.op_type == "Add": # skip matmul bias
            input_producers = [graph_q.get_tensor_producer(inp) for inp in node.input]
            # one matmul, one in initializer
            if len(input_producers) == 2:
                if any([prod.op_type == "MatMul" for prod in input_producers if not isinstance(prod, str)]) and \
                   any([inp in graph_q.initializer for inp in node.input]):
                    continue
        if node.op_type in platform_setting_table[args.deploy]['quant_nodes']:
            quant_node_list.append(node)

    if args.w8_kurtosis_topk is not None:
        # further filter quant_node_list by kurtosis
        node_kurtosis = {}
        for node in quant_node_list:
            if node.name in args.w8_layers or node.name in args.skip_layers:
                continue
            if node.op_type == 'Conv':
                group = [attr for attr in node.attribute if attr.name == 'group'][0]
                if group.i != 1 and args.deploy == 'sophgo': continue
            if node.op_type in LAYER_HAS_WEIGHT and node.input[1] in graph_q.initializer:
                weight = numpy_helper.to_array(graph_q.initializer[node.input[1]][0])
                weight_mean = np.mean(weight)
                weight_std = np.std(weight)
                if weight_std == 0:
                    weight_std = 1e-5
                kurtosis = np.mean(((weight - weight_mean) / weight_std) ** 4)
                node_kurtosis[node.name] = kurtosis
        # get topk nodes by kurtosis
        sorted_nodes = sorted(node_kurtosis.items(), key=lambda x: x[1], reverse=True)
        topk_nodes = [name for name, _ in sorted_nodes[:args.w8_kurtosis_topk]]
        args.w8_layers.extend([name for name in topk_nodes if name not in args.w8_layers])
        logger.info(f"Top-{args.w8_kurtosis_topk} layers by kurtosis for w8 quantization:")
        for name in topk_nodes:
            logger.info(f"  {name}: kurtosis={node_kurtosis[name]:.4f}")

    act_quantized = []
    for node in quant_node_list:
        insert_fake_quant_node(graph_q, node, act_quantized, clip_val, args)
    if platform_setting_table[args.deploy]['quantize_network_output']:
        insert_fake_quant_node_output(graph_q, clip_val, args)
    for node in quant_node_list_w4a4:
        insert_fake_quant_node_w4a4(graph_q, node, act_quantized, clip_val, args)
    graph_q.update_model()
    return graph_q, quant_node_list


def insert_fake_quant_node(graph, node, act_quantized, data_range_list, args):
    param = copy.deepcopy(platform_setting_table[args.deploy])
    # We now quant input and weight tp INT8 but left output fp32.
    find_weight = False
    trt_merge_add = False
    for idx, in_tensor in enumerate(node.input):
        # ConvTranspose need transpose for weight.
        need_transpose = False
        shape = graph.get_tensor_shape(in_tensor)
        # Merge ReLU
        if node.op_type in RELU_TYPE:
            if isinstance(graph.get_tensor_producer(node.input[0]), str):
                continue
            _prev = graph.get_tensor_producer(node.input[0])
            if len(node.input) == 1 and not isinstance(_prev, str) and _prev.op_type in MERGE_RELU:
                continue

        q_nodes = None
        # Quantize weight.
        if in_tensor in graph.initializer and node.op_type in LAYER_HAS_WEIGHT:
            if not find_weight:
                # We find Weight here.
                find_weight = True
                if node.op_type == 'ConvTranspose' or node.op_type == 'MatMul':
                    need_transpose = True
                if node.op_type == 'Conv':
                    group = [attr for attr in node.attribute if attr.name == 'group'][0]
                    if group.i != 1 and args.deploy == 'sophgo':
                        # sophgo does not support w4a8 depthwise conv
                        param['qw_params']['bit_width'] = 8
                if param['qw_params']['bit_width'] != 8 and node.name in args.w8_layers:
                    param['qw_params']['bit_width'] = 8
                if param['qw_params']['bit_width'] != 8 and args.w8_max_threshold is not None:
                    min_weight = np.min(data_range_list[in_tensor])
                    max_weight = np.max(data_range_list[in_tensor])
                    if (min_weight < -args.w8_max_threshold) or (max_weight > args.w8_max_threshold):
                        param['qw_params']['bit_width'] = 8
                if param['qw_params']['bit_width'] != 8 and args.w8_p99_threshold is not None:
                    weight = numpy_helper.to_array(graph.initializer[in_tensor][0])
                    max_value = np.max(np.abs(weight))
                    p99_value = np.percentile(np.abs(weight), 99)
                    if max_value > args.w8_p99_threshold * p99_value:
                        param['qw_params']['bit_width'] = 8
                if param['qw_params']['bit_width'] != 8 and args.w8_kurtosis_threshold is not None:
                    weight = numpy_helper.to_array(graph.initializer[in_tensor][0])
                    weight_mean = np.mean(weight)
                    weight_std = np.std(weight)
                    if weight_std == 0:
                        weight_std = 1e-5
                    kurtosis = np.mean(((weight - weight_mean) / weight_std) ** 4)
                    if kurtosis > args.w8_kurtosis_threshold:
                        param['qw_params']['bit_width'] = 8

                q_nodes, _, _ = get_qnode_by_param(param['qw_params'], in_tensor, shape, data_range_list[in_tensor],
                                                    need_transpose)

            elif 'qb_params' in param:
                # We find bias here.
                q_nodes, _, _ = get_qnode_by_param(param['qb_params'], in_tensor, shape, data_range_list[in_tensor],
                                                   need_transpose)

        # Quantize input.
        if in_tensor in graph.network_inputs or in_tensor not in graph.input:
            # Conv   Conv    Conv
            #  |       |       |
            #  skip    Q       Q
            #   \      |      /
            #         Add           means the first add branch with conv should merge in TRT.
            if (args.deploy == 'trt' or args.deploy == 'stpu') and node.op_type == 'Add' and not trt_merge_add:
                _prev = graph.get_tensor_producer(in_tensor)
                if _prev.op_type == 'Conv' or _prev.op_type == 'MatMul':
                    trt_merge_add = True
                    continue
            if (args.deploy == 'sophgo') and node.op_type == 'Mul':
                _prev = graph.get_tensor_producer(in_tensor)
                if _prev.op_type == 'Sigmoid':
                    continue

            if (args.deploy == 'sophgo') and node.op_type == 'Add' and args.optim_transformer:
                _prev = graph.get_tensor_producer(in_tensor)
                if _prev.op_type == 'Add':
                    input_producers = [graph.get_tensor_producer(inp) for inp in _prev.input]
                    if len(input_producers) != 2:
                        pass
                    elif any([prod.op_type == "MatMul" for prod in input_producers if not isinstance(prod, str)]) and \
                        any([inp in graph.initializer for inp in _prev.input]):
                        pass
                    else:
                        continue
            signed = data_range_list[in_tensor][0] < 0
            q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor, shape, data_range_list[in_tensor], signed=signed)

        if q_nodes is not None:
            node.input[idx] = q_nodes.output[0].name
            # Output already quantized.
            if in_tensor in act_quantized:
                continue
            graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
            act_quantized.append(in_tensor)

    graph.topologize_graph()

def insert_fake_quant_node_w4a4(graph, node, act_quantized, data_range_list, args):
    param = platform_setting_table[args.deploy]
    # We now quant input and weight tp INT8 but left output fp32.
    find_weight = False
    trt_merge_add = False
    for idx, in_tensor in enumerate(node.input):
        # ConvTranspose need transpose for weight.
        need_transpose = False
        shape = graph.get_tensor_shape(in_tensor)

        q_nodes = None
        # Quantize weight.
        if in_tensor in graph.initializer and node.op_type in LAYER_HAS_WEIGHT:
            if not find_weight:
                # We find Weight here.
                find_weight = True
                if node.op_type == 'ConvTranspose' or node.op_type == 'MatMul':
                    need_transpose = True
                param['qw_params']['bit_width'] = 4
                q_nodes, _, _ = get_qnode_by_param(param['qw_params'], in_tensor, shape, data_range_list[in_tensor],
                                                need_transpose)

            elif 'qb_params' in param:
                # We find bias here.
                q_nodes, _, _ = get_qnode_by_param(param['qb_params'], in_tensor, shape, data_range_list[in_tensor],
                                                   need_transpose)
            if q_nodes is not None:
                node.input[idx] = q_nodes.output[0].name
                # Output already quantized.
                if in_tensor in act_quantized:
                    continue
                graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                act_quantized.append(in_tensor)
        # Quantize input.
        if in_tensor in graph.network_inputs or in_tensor not in graph.input:
            # Conv   Conv    Conv
            #  |       |       |
            #  skip    Q       Q
            #   \      |      /
            #         Add           means the first add branch with conv should merge in TRT.
            if (args.deploy == 'stpu') and node.op_type in LAYER_HAS_WEIGHT:
                # in_tensor 已经被插入int8伪量化节点，现在仅仅需要在int8伪量化节点之后插入int4伪量化节点就可以
                if in_tensor in act_quantized:
                    param['qi_params']['bit_width'] = 4
                    in_tensor_1 = in_tensor + DQTENSORSUFFIX
                    q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor_1, shape, data_range_list[in_tensor])
                    if q_nodes is not None:
                        node.input[idx] = q_nodes.output[0].name
                        graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                #in_tensor还没有被插入伪量化节点，需要判断前继节点是int4 or int8，如果是int4，只需要插入int4伪量化节点，如果是int8，则需要插入int8+int4伪量化节点
                else:
                    _prev = graph.get_tensor_producer(in_tensor)
                    if _prev.name in platform_setting_table[args.deploy]['w4a4']:
                        param['qi_params']['bit_width'] = 4
                        q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor, shape, data_range_list[in_tensor])
                        if q_nodes is not None:
                            node.input[idx] = q_nodes.output[0].name
                            graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                            act_quantized.append(in_tensor)
                    else:
                        param['qi_params']['bit_width'] = 8
                        q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor, shape, data_range_list[in_tensor])
                        if q_nodes is not None:
                            node.input[idx] = q_nodes.output[0].name
                            graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                            graph.topologize_graph()
                        param['qi_params']['bit_width'] = 4
                        in_tensor_1 = in_tensor + DQTENSORSUFFIX
                        q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor_1, shape, data_range_list[in_tensor])
                        if q_nodes is not None:
                            node.input[idx] = q_nodes.output[0].name
                            graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                        act_quantized.append(in_tensor)
                # _prev = graph.get_tensor_producer(in_tensor)
                # if _prev.name in platform_setting_table[args.deploy]['w4a4']:
                #     param['qi_params']['bit_width'] = 4
                #     q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor, shape, data_range_list[in_tensor])
                #     if q_nodes is not None:
                #         node.input[idx] = q_nodes.output[0].name
                #         # Output already quantized.
                #         if in_tensor in act_quantized:
                #             continue
                #         graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                #         act_quantized.append(in_tensor)
                # else:
                #     param['qi_params']['bit_width'] = 8
                #     q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor, shape, data_range_list[in_tensor])
                #     if q_nodes is not None:
                #         node.input[idx] = q_nodes.output[0].name
                #         # Output already quantized.
                #         if in_tensor in act_quantized:
                #             continue
                #         graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)
                #         graph.topologize_graph()
                #     param['qi_params']['bit_width'] = 4
                #     in_tensor_1 = in_tensor + DQTENSORSUFFIX
                #     q_nodes, _, _ = get_qnode_by_param(param['qi_params'], in_tensor_1, shape, data_range_list[in_tensor])
                #     if q_nodes is not None:
                #         node.input[idx] = q_nodes.output[0].name
                #         graph.insert_qnodes_purely(q_nodes=q_nodes, node=node)   
                #     act_quantized.append(in_tensor)
    graph.topologize_graph()

def insert_fake_quant_node_output(graph, clip_val, args):
    param = platform_setting_table[args.deploy]
    out_tensor_list = copy.deepcopy(graph.network_outputs)
    for out_tensor in out_tensor_list:
        signed = clip_val[out_tensor][0] < 0
        q_nodes, _, _ = get_qnode_by_param(param['qi_params'], out_tensor, graph.get_tensor_shape(out_tensor),
                                           clip_val[out_tensor], signed=signed)
        graph.insert_qnodes_purely(q_nodes=q_nodes, idx=graph.index(graph.get_tensor_producer(out_tensor)) + 1)
        graph.del_network_output(out_tensor)
        graph.add_network_output(q_nodes.output[0])
    graph.topologize_graph()
    return


def get_qnode_by_param(param, in_tensor_name, tensor_shape, range, need_transpose=False, signed=True):
    bit_width = param['bit_width']
    zero_point = [0]
    per_channel = True
    if 'per_channel' not in param or not param['per_channel']:
        per_channel = False

    if param['type'] == "Linear":
        symmetric = param['symmetric']
        if 'per_channel' not in param or not param['per_channel']:
            range[0] = np.min(range[0])
            range[1] = np.max(range[1])
            # dynamic_sym only works on activations which could not
            # be perchannel.
            if 'dynamic_sym' in param and param['dynamic_sym']:
                if np.abs(range[0] - 0.0) < 1e-6:
                    symmetric = False
        if symmetric:
            if not isinstance(range[0], np.ndarray):
                channel_num = 1
            else:
                channel_num = len(range[0])
            # 8bit -128-127 actually identical to -127-127
            if signed:
                q_min = [-2 ** (bit_width - 1) + 1] * channel_num
                q_max = [2 ** (bit_width - 1) - 1] * channel_num
            else:
                q_min = [0] * channel_num
                q_max = [2 ** bit_width - 1] * channel_num
            data_max = np.max(np.abs(range), axis=0)
            scale = np.array(data_max) / q_max
            if np.any(scale == 0):
                # force set scale to 1 for zero weight channel
                # print("Find {} channels all zero in {}, set scale to 1.".
                #       format(len(np.where(scale == 0)[0]), in_tensor_name))
                scale = np.where(scale == 0, 1., scale)
            scale = scale.tolist()

        else:
            if not isinstance(range[0], np.ndarray):
                # Per layer.
                data_min = min(0, range[0])
                data_max = max(0, range[1])
                scale = (data_max - data_min) / (2 ** bit_width - 1)
                # Align zero point.
                if scale == 0.0:
                    scale += 1.
                zero_point = np.round(-data_min / scale)
                # data_min = -zero_point * scale
                # data_max = scale * (2 ** bit_width - 1) + data_min
                # scale = (data_max - data_min) / (2 ** bit_width - 1)
                # q_min = [int(np.round(data_min / scale))]
                # q_max = [int(np.round(data_max / scale))]
                q_min = [int(-zero_point)]
                q_max = [int(2 ** (bit_width) - 1 - zero_point)]
                scale = [float(scale)]
            else:
                # Per channel.
                data_min = range[0]
                data_min[data_min > 0.] = 0.
                data_max = range[1]
                data_max[data_max < 0.] = 0.
                scale = (data_max - data_min) / (2 ** bit_width - 1)
                # Fix all zero channel.
                if np.any(scale == 0):
                    # force set scale to 1 for zero weight channel
                    logger.warning("Find {} channels all zero in {}, set scale to 1.".format(
                                   len(np.where(scale == 0)[0]), in_tensor_name))
                    scale = np.where(scale == 0, 1., scale)
                # Align zero point.
                zero_point = (-data_min / scale).round()
                q_min = -zero_point
                q_max = (2 ** (bit_width) - 1 - zero_point).astype(np.int32).tolist()
                q_min = q_min.astype(np.int32).tolist()
                scale = scale.tolist()
        if 'log_scale' in param and param['log_scale']:
            scale = 2 ** np.round(np.log2(scale))
        scale = np.array(scale, dtype=np.float32)
        zero_point = np.full(scale.shape, zero_point, dtype=np.int8)
        if not signed:
            zero_point -= 128
        q_nodes = make_quant_dequant(in_tensor_name,
                                     tensor_shape,
                                     scale,
                                     zero_point,
                                     need_transpose,
                                     per_channel,
                                     symmetric)

    return q_nodes, q_min, q_max


def make_quant_dequant(tensor_name, tensor_shape, scale_val, zero_point_val, need_transpose=False,
                       per_channel=False, symmetric=True):
    in_tensor = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, tensor_shape)
    if len(scale_val) == 1:
        shape = []
    else:
        shape = list(scale_val.shape)
    scale = helper.make_tensor(tensor_name + '_scale', TensorProto.FLOAT, shape, scale_val)
    zero_point = helper.make_tensor(tensor_name + '_zero_point',
                                    TensorProto.INT8 if symmetric else TensorProto.UINT8, shape, zero_point_val)
    tensor_dequant = helper.make_tensor_value_info(tensor_name + DQTENSORSUFFIX, TensorProto.FLOAT, tensor_shape)
    if per_channel:
        q_node = helper.make_node(
            name=tensor_name + "_QuantizeLinear",
            op_type="QuantizeLinear",
            inputs=[tensor_name, tensor_name + '_scale', tensor_name + '_zero_point'],
            outputs=[tensor_name + QTENSORSUFFIX],
            axis=1 if need_transpose else 0)
        dq_node = helper.make_node(
            name=tensor_name + "_DequantizeLinear",
            op_type="DequantizeLinear",
            inputs=[tensor_name + QTENSORSUFFIX, tensor_name + '_scale', tensor_name + '_zero_point'],
            outputs=[tensor_name + DQTENSORSUFFIX],
            axis=1 if need_transpose else 0)
    else:
        q_node = helper.make_node(
            name=tensor_name + "_QuantizeLinear",
            op_type="QuantizeLinear",
            inputs=[tensor_name, tensor_name + '_scale', tensor_name + '_zero_point'],
            outputs=[tensor_name + QTENSORSUFFIX])
        dq_node = helper.make_node(
            name=tensor_name + "_DequantizeLinear",
            op_type="DequantizeLinear",
            inputs=[tensor_name + QTENSORSUFFIX, tensor_name + '_scale', tensor_name + '_zero_point'],
            outputs=[tensor_name + DQTENSORSUFFIX])
    graph_quant = helper.make_graph(
        [q_node, dq_node],
        'graph_quant',
        [in_tensor],
        [tensor_dequant],
        initializer=[scale, zero_point],
    )
    return graph_quant
