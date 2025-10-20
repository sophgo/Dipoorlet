import copy
import json
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper
from torch.nn.parallel import DistributedDataParallel as DDP

from ..forward_net import ActivationCache
from ..platform_settings import platform_setting_table
from ..quantize import QUANT_NODE_NAME_LIST, quant_graph
from ..utils import logger
from .ada_quant_layer import *
from .utils import *
from .weight_equalization import node_has_equalized


def brecq(graph_ori, graph, act_clip_val, weight_clip_val, args):
    dist.barrier()
    rank = dist.get_rank()
    num_per_rank = args.data_num // dist.get_world_size()
    rank_st = rank * num_per_rank
    rank_ed = rank_st + num_per_rank
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_brecq = copy.deepcopy(graph)
    fp_act_cache = ActivationCache(graph_ori, args, rank_st, rank_ed)
    prev_act_cache = None
    already = []
    w8_nodes = {}
    dw_convs = {}
    _log_head = 'Qdrop' if args.drop is True else 'Brecq'
    for node in graph_ori.graph.node:
        if node.name in args.skip_layers:
            continue
        if node.op_type in LEARNABLE_LAYER_TYPES and node.name not in already:
            if node.op_type == "MatMul" and node.input[1] not in graph_ori.initializer:
                continue
            block_layer_list = get_block_from_first(graph, node, args)
            logger.info("Get block: "+" ".join([_node.name for _node in block_layer_list]))
            # If the last node has weight equalized, it cannot be the last.
            if args.we:
                if node_has_equalized(graph, block_layer_list[-1]):
                    block_layer_list.pop(-1)
            if dist.get_rank() == 0:
                logger.info("{} for: {}".format(_log_head, ' '.join([_node.name for _node in block_layer_list])))
            already.extend([_node.name for _node in block_layer_list])
            # Using graph_brecq and restore act cache for incremental update.
            if not prev_act_cache:
                graph_q, quant_node_list = quant_graph(graph_brecq, clip_val, args)
                rank = dist.get_rank()
                if rank == 0:
                    graph_q.save_onnx_model(name='quant_model')
                q_act_cache = ActivationCache(graph_q, args, rank_st, rank_ed)
            else:
                q_act_cache.update_graph(graph_q)
                q_act_cache.activation_cache = prev_act_cache
            prev_node = graph_q.get_tensor_consumer(block_layer_list[0].input[0])[0]
            prev_node = graph_q.get_tensor_consumer(prev_node.output[0])[0]
            in_tensor_name = block_layer_list[0].input[0]
            if prev_node.op_type == QUANT_NODE_NAME_LIST[-1]:
                in_tensor_name = prev_node.output[0]
            q_in_tensor = np.stack(q_act_cache[in_tensor_name])
            fp_in_tensor = np.stack(fp_act_cache[block_layer_list[0].input[0]])
            if block_layer_list[-1].op_type == 'MatMul' and follow_bias_after_matmul(graph_brecq, block_layer_list[-1]):
                add_node = graph_brecq.get_tensor_consumer(block_layer_list[-1].output[0])[0]
                out_tensor_name = add_node.output[0]
            else:
                out_tensor_name = block_layer_list[-1].output[0]
            fp_out_tensor = np.stack(fp_act_cache[out_tensor_name])
            prev_act_cache = q_act_cache.activation_cache.copy()
            # Use one reg for seq.
            total_iter = args.ada_epoch * len(block_layer_list) * np.ceil(num_per_rank / args.ada_bs)
            reg = adaround_reg(total_iter)
            ada_layer_list = []
            # Get weight and build torch conv.
            for _node in block_layer_list:
                weight = numpy_helper.to_array(graph_brecq.initializer[_node.input[1]][0])
                weight = torch.from_numpy(weight).cuda()
                bias = None
                if len(_node.input) == 3:
                    bias = numpy_helper.to_array(graph_brecq.initializer[_node.input[2]][0])
                if _node.op_type == 'MatMul' and follow_bias_after_matmul(graph_brecq, _node):
                    add_node = graph_brecq.get_tensor_consumer(_node.output[0])[0]
                    if add_node.input[0] == _node.output[0]:
                        bias = numpy_helper.to_array(graph_brecq.initializer[add_node.input[1]][0])
                    else:
                        bias = numpy_helper.to_array(graph_brecq.initializer[add_node.input[0]][0])
                    logger.info("Find bias after matmul for node: {}".format(_node.name))
                # Get quantization param.
                if args.deploy != 'nnie':
                    weight_range = clip_val[_node.input[1]]
                    qw_param = platform_setting_table[args.deploy]['qw_params'].copy()
                    if _node.op_type == 'ConvTranspose' or _node.op_type == 'MatMul':
                        weight = weight.transpose(0, 1)
                    if _node.op_type == 'Conv':
                        group = [attr for attr in _node.attribute if attr.name == 'group'][0]
                        if group.i != 1:
                            qw_param['bit_width'] = 8
                            dw_convs[_node.name] = {
                                'op_type': _node.op_type,
                                'group': group.i,
                                'inputs': [i for i in _node.input],
                                'outputs': [o for o in _node.output]
                            }
                    if qw_param['bit_width'] < 8 and _node.name in args.w8_layers:
                        qw_param['bit_width'] = 8
                        w8_nodes[_node.name] = {
                            'op_type': _node.op_type,
                            'inputs': [i for i in _node.input],
                            'outputs': [o for o in _node.output]
                        }
                    if qw_param['bit_width'] < 8 and args.w8_max_threshold is not None:
                        min_weight = np.min(weight_range)
                        max_weight = np.max(weight_range)
                        if (min_weight < -args.w8_max_threshold) or (max_weight > args.w8_max_threshold):
                            qw_param['bit_width'] = 8
                            w8_nodes[_node.name] = {
                                'op_type': _node.op_type,
                                'weight_range': (float(min_weight), float(max_weight)),
                                'inputs': [i for i in _node.input],
                                'outputs': [o for o in _node.output]
                            }
                    if qw_param['bit_width'] < 8 and args.w8_p99_threshold is not None:
                        p99_weight = np.percentile(np.abs(weight.cpu().numpy()), 99.0)
                        max_weight = np.max(np.abs(weight_range))
                        if max_weight > p99_weight * args.w8_p99_threshold:
                            qw_param['bit_width'] = 8
                            w8_nodes[_node.name] = {
                                'op_type': _node.op_type,
                                'weight_range': (float(np.min(weight_range)), float(np.max(weight_range))),
                                'inputs': [i for i in _node.input],
                                'outputs': [o for o in _node.output],
                                'p99_weight': float(p99_weight)
                            }
                    scale, q_min, q_max = get_quant_tensor(weight.shape, qw_param, weight_range)
                    rest = (weight / scale) - (weight / scale).floor()
                    qw_tensor = {'scale': scale,
                                 'q_min': q_min,
                                 'q_max': q_max,
                                 'per_channel': qw_param['per_channel'],
                                 'type': 'Linear'}
                else:
                    qw_tensor = {'scale': None,
                                 'q_min': None,
                                 'q_max': None,
                                 'per_channel': None,
                                 'type': 'NNIE'}
                    rest = nnie_rest_init(weight)

                # Generate torch qlayer.
                act_func_type = follow_nolinear(graph, _node)
                logger.info("Following act func for node {}: {}".format(_node.name, act_func_type))
                # get acti quantization param
                following_node = following_nolinear(graph, _node, act_func_type)
                acti_range = clip_val[following_node.output[0]]
                if args.deploy != 'nnie':
                    acti_shape = graph.get_tensor_shape(following_node.output[0])
                    qi_param = platform_setting_table[args.deploy]['qi_params']
                    signed = acti_range[0] < 0
                    scale, q_min, q_max = get_quant_tensor(acti_shape, qi_param, acti_range, signed=signed)
                    qi_tensor = {'scale': scale,
                                 'q_min': q_min,
                                 'q_max': q_max,
                                 'type': 'Linear'}
                else:
                    max_value = max(abs(acti_range[0]), acti_range[1])
                    max_value = torch.from_numpy(np.array(max_value).astype(np.float32)).cuda()
                    qi_tensor = {'max_value': max_value,
                                 'type': 'NNIE'}
                qmid_tensor = None
                if act_func_type in ['silu', 'gelu']:
                    if _node.op_type == 'MatMul' and follow_bias_after_matmul(graph_brecq, _node):
                        mid_out = graph_brecq.get_tensor_consumer(_node.output[0])[0].output[0]
                    else:
                        mid_out = _node.output[0]
                    acti_range = clip_val[mid_out]
                    acti_shape = graph.get_tensor_shape(mid_out)
                    qi_param = platform_setting_table[args.deploy]['qi_params']
                    scale, q_min, q_max = get_quant_tensor(acti_shape, qi_param, acti_range)
                    qmid_tensor = {'scale': scale,
                                   'q_min': q_min,
                                   'q_max': q_max,
                                   'type': 'Linear'}
                ada_layer_list.append(
                    AdaQLayer(_node, weight, bias, rest, reg, qw_tensor, qi_tensor,
                              act_func_type, _node.op_type, args.acti_quant, qmid_tensor)
                )
            # Block output follow relu, silu or gelu.
            act_func_type = follow_nolinear(graph, block_layer_list[-1])
            if act_func_type == 'relu':
                fp_out_tensor = torch.nn.Parameter(F.relu(torch.from_numpy(fp_out_tensor)), False)
            elif act_func_type == 'silu':
                fp_out_tensor = torch.nn.Parameter(F.silu(torch.from_numpy(fp_out_tensor)), False)
            elif act_func_type == 'gelu':
                fp_out_tensor = torch.nn.Parameter(F.gelu(torch.from_numpy(fp_out_tensor)), False)
            else:
                fp_out_tensor = torch.nn.Parameter(torch.from_numpy(fp_out_tensor), False)
            # Learning.
            ada_block = nn.Sequential(*ada_layer_list)
            round_mask_list = learning_round_mask(
                torch.nn.Parameter(torch.from_numpy(q_in_tensor).cuda(), False),
                torch.nn.Parameter(torch.from_numpy(fp_in_tensor).cuda(), False),
                fp_out_tensor.cuda(),
                ada_block, reg, args.ada_bs, args.ada_epoch * len(block_layer_list), args.drop)
            # Deploy new weight.
            for idx, _node in enumerate(block_layer_list):
                weight = numpy_helper.to_array(graph_brecq.initializer[_node.input[1]][0])
                weight = torch.from_numpy(weight).cuda()
                round_mask = round_mask_list[idx]
                if args.deploy != 'nnie':
                    weight_range = clip_val[_node.input[1]]
                    qw_param = platform_setting_table[args.deploy]['qw_params'].copy()
                    if _node.op_type == 'ConvTranspose' or _node.op_type == 'MatMul':
                        weight = weight.transpose(0, 1)
                    if _node.op_type == 'Conv':
                        group = [attr for attr in _node.attribute if attr.name == 'group'][0]
                        if group.i != 1: qw_param['bit_width'] = 8
                    if qw_param['bit_width'] != 8 and _node.name in args.w8_layers:
                        qw_param['bit_width'] = 8
                    if qw_param['bit_width'] != 8 and args.w8_max_threshold is not None:
                        min_weight = np.min(weight_range)
                        max_weight = np.max(weight_range)
                        if (min_weight < -args.w8_max_threshold) or (max_weight > args.w8_max_threshold):
                            qw_param['bit_width'] = 8
                    if qw_param['bit_width'] != 8 and args.w8_p99_threshold is not None:
                        p99_weight = np.percentile(np.abs(weight.cpu().numpy()), 99.0)
                        max_weight = np.max(np.abs(weight_range))
                        if max_weight > p99_weight * args.w8_p99_threshold:
                            qw_param['bit_width'] = 8
                    scale, q_min, q_max = get_quant_tensor(weight.shape, qw_param, weight_range)
                    new_rounded_weight = quant_weight(
                        weight,
                        round_mask, scale, q_min, q_max,
                        qw_param['per_channel'], soft=False)
                    if _node.op_type == 'ConvTranspose' or _node.op_type == 'MatMul':
                        new_rounded_weight = new_rounded_weight.transpose(0, 1)
                else:
                    new_rounded_weight = quant_weight_nnie(weight, round_mask, soft=False)
                new_rounded_weight = new_rounded_weight.cpu().detach().numpy()
                update_weight(graph_brecq, new_rounded_weight, _node.input[1])
                update_weight(graph_q, new_rounded_weight, _node.input[1])
            graph_brecq.update_model()
            graph_q.update_model()
    if dist.get_rank() == 0:
        graph_brecq.save_onnx_model('brecq')
        if len(w8_nodes) > 0:
            with open(osp.join(args.output_dir, 'brecq_w8_nodes.json'), 'w') as f:
                json.dump(w8_nodes, f, indent=4)
        if len(dw_convs) > 0:
            with open(osp.join(args.output_dir, 'brecq_dw_convs.json'), 'w') as f:
                json.dump(dw_convs, f, indent=4)
    # We must use original ranges.
    return graph_brecq


def learning_round_mask(q_in_tensor, fp_in_tensor, fp_out_tensor, ada_block, reg, batch_size, max_epoch, drop):
    opt_list = []
    for layer in ada_block:
        if isinstance(layer, AdaQLayer):
            opt_list.append(layer.round_mask)
    optimizer = torch.optim.Adam(opt_list)
    ada_block = DDP(ada_block, [torch.cuda.current_device()])
    # New train precedure
    cur_iter = 0
    ratio = 0.5 if drop else 1.0
    for epoch in range(max_epoch):
        if ratio < 1.0:
            in_tensor = torch.where(torch.rand_like(q_in_tensor) < ratio, q_in_tensor, fp_in_tensor)
        else:
            in_tensor = q_in_tensor
        for idx in range(np.ceil(len(in_tensor) / batch_size).astype(int)):
            st = idx * batch_size
            ed = st + batch_size
            input = in_tensor[st:ed].squeeze(1)
            fp_output = fp_out_tensor[st:ed].squeeze(1)
            output = ada_block(input)
            loss = L2_norm(output, fp_output)
            for layer in ada_block.module:
                if isinstance(layer, AdaQLayer):
                    loss += reg(layer.round_mask, cur_iter)
            cur_iter += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0 and dist.get_rank() == 0:
            logger.info("Epoch: {:<5} L2 Loss: {:>10.3f} Beta: {:>3.3f}".format(epoch, loss, reg.beta))
    for layer in ada_block.module:
        if isinstance(layer, AdaQLayer):
            res = adaround_reg().rectified_sigmoid(layer.round_mask)
            if dist.get_rank() == 0:
                logger.info("Ceil: {:>5} Floor: {:>5} Total: {:>5} Ratio: {:>.3f}".format(
                    res[res + 1e-4 >= 1.0].numel(), res[res <= 1e-4].numel(), torch.numel(res),
                    (res[res + 1e-4 >= 1.0].numel() + res[res <= 1e-4].numel()) / torch.numel(res)))
    round_mask_list = []
    for layer in ada_block.module:
        if isinstance(layer, AdaQLayer):
            round_mask_list.append(layer.round_mask)
    return round_mask_list
