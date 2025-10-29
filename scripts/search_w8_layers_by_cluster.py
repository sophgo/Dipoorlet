import argparse
import copy
import json
import os
import sys
import time
from typing import List
from collections import defaultdict, deque

import onnx
import numpy as np
from sklearn.cluster import KMeans
from onnxsim import simplify

from dipoorlet.forward_net import ActivationCache
from dipoorlet.quantize import quant_graph
from dipoorlet.tensor_cali import tensor_calibration
from dipoorlet.utils import (ONNXGraph, load_clip_val, logger, rewrite_onnx_gelu,
                             save_clip_val, setup_logger, update_act_clip_val)
from dipoorlet.weight_transform.utils import LEARNABLE_LAYER_TYPES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--model", help="onnx model")
    parser.add_argument("-I", "--input_dir", help="calibration data", required=True)
    parser.add_argument("-O", "--output_dir", help="output data path")
    parser.add_argument("-N", "--data_num", help="num of calibration pics", type=int, required=True)
    parser.add_argument("-A", "--act_quant", help="algorithm of activation quantization",
                        choices=['minmax', 'hist', 'mse'], default='mse')
    parser.add_argument("-D", "--deploy", help="deploy platform",
                        choices=['trt', 'stpu', 'magicmind', 'rv', 'atlas',
                                'snpe', 'ti', 'imx', 'sophgo'], default='sophgo')
    parser.add_argument("--bins", help="bins for histogram and kl", default=2048)
    parser.add_argument("--threshold", help="threshold for histogram", default=0.99999, type=float)
    parser.add_argument("--optim_transformer", help="Transformer model optimization", default=False, action='store_true')
    parser.add_argument("--model_type", help="Transformer model type", choices=["unet", "swin", "vit"], default=None)
    parser.add_argument("--quant_format", default="QDQ", type=str, choices=["QOP", "QDQ"])
    parser.add_argument("--extra_calib_table", type=str, default=None, help="Path to extra calibration table to update activation clip values.")
    parser.add_argument("--skip_layers", help="Skip layer name", default=[], type=str, nargs='+')
    parser.add_argument("--cluster_num", type=int, default=2, help="Number of clusters for weight quantization")
    parser.add_argument("--w8_layers", type=str, nargs='+', default=[], help="Layers for 8-bit weight quantization")
    parser.add_argument("--load_loss_file", type=str, default=None, help="Path to load layer loss file")
    args = parser.parse_args()
    args.local_rank = 0
    args.world_size = 1
    args.rank = 0
    args.w8_max_threshold = None
    args.w8_p99_threshold = None
    args.w8_kurtosis_threshold = None
    args.w8_kurtosis_topk = None
    return args


def get_last_learnable_ops(model: onnx.ModelProto) -> List[onnx.NodeProto]:
    # 1. 获取图
    graph = model.graph

    # 2. 建立 value_name -> 消费它的节点列表 的映射
    value2consuming_ops = defaultdict(list)
    for op in graph.node:
        for inp in op.input:
            value2consuming_ops[inp].append(op)

    # 3. 找出所有 Conv/Gemm/MatMul 节点
    candidates: List[onnx.NodeProto] = [n for n in graph.node if n.op_type in LEARNABLE_LAYER_TYPES]

    # 4. 从每个候选节点出发，BFS 看它后面是否还有同类算子
    def has_conv_gemm_matmul_downstream(start: onnx.NodeProto) -> bool:
        """True 表示身后还有 Conv/Gemm/MatMul"""
        visited = set()
        dq = deque()
        # 把 start 的所有输出张量作为起点
        for out_name in start.output:
            for cons_op in value2consuming_ops[out_name]:
                dq.append(cons_op)
        while dq:
            op = dq.popleft()
            if op.name in visited:
                continue
            visited.add(op.name)
            if op.op_type in LEARNABLE_LAYER_TYPES:
                return True
            # 继续往下游走
            for out_name in op.output:
                for cons_op in value2consuming_ops[out_name]:
                    dq.append(cons_op)
        return False

    # 5. 筛选
    final_ops: List[onnx.NodeProto] = [
        n for n in candidates if not has_conv_gemm_matmul_downstream(n)
    ]
    return final_ops

def cos_loss(fp_tensors, q_tensors):
    total_loss = 0.0
    bs = fp_tensors[0].shape[0]
    for fp, q in zip(fp_tensors, q_tensors):
        fp = fp.reshape(bs, -1)
        q = q.reshape(bs, -1)
        fp_norm = np.linalg.norm(fp, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        total_loss += np.sum(1 - np.sum(fp * q, axis=1, keepdims=True) / (fp_norm * q_norm + 1e-8)) / bs
    total_loss /= len(fp_tensors)
    return total_loss.item()

def snr_loss(fp_tensors, q_tensors):
    total_loss = 0.0
    bs = fp_tensors[0].shape[0]
    for fp, q in zip(fp_tensors, q_tensors):
        fp = fp.reshape(bs, -1)
        q = q.reshape(bs, -1)
        signal_power = np.sum(fp ** 2, axis=1)
        noise_power = np.sum((fp - q) ** 2, axis=1) + 1e-8
        snr = noise_power / signal_power
        total_loss += np.sum(snr) / bs
    total_loss /= len(fp_tensors)
    return total_loss.item()

def search_w8_layers(graph, clip_val, logger, args) -> List[str]:
    nc = args.cluster_num
    fp_act_cache = ActivationCache(graph, args)
    last_learnable_ops = get_last_learnable_ops(graph.model)
    last_learnable_outputs = []
    for op in last_learnable_ops:
        last_learnable_outputs.extend(op.output)
    fp_output_tensors = [np.concatenate(fp_act_cache[output]) for output in last_learnable_outputs]
    candidate_ops = []
    for op in graph.graph.node:
        if op.name in args.skip_layers:
            continue
        if op.op_type in LEARNABLE_LAYER_TYPES and op.name not in candidate_ops:
            if op.op_type == "MatMul" and op.input[1] not in graph.initializer:
                continue
            if op.op_type == 'Conv':
                group = [attr.i for attr in op.attribute if attr.name == 'group'][0]
                if group != 1 and args.deploy == 'sophgo':
                    continue
            candidate_ops.append(op)
    
    layer_losses = {}
    if args.load_loss_file is not None and os.path.isfile(args.load_loss_file):
        with open(args.load_loss_file, 'r') as f:
            layer_losses = json.load(f)
    total_time = 0.0
    for op in candidate_ops:
        start_time = time.time()
        if op.name not in layer_losses:
            args.w8_layers = [_.name for _ in candidate_ops if _ != op]
            quantized_graph, _ = quant_graph(graph, clip_val, args)
            q_act_cache = ActivationCache(quantized_graph, args)
            q_output_tensors = [np.concatenate(q_act_cache[output]) for output in last_learnable_outputs]
            layer_cos_loss = cos_loss(fp_output_tensors, q_output_tensors)
            layer_snr_loss = snr_loss(fp_output_tensors, q_output_tensors)
            layer_losses[op.name] = [layer_cos_loss, layer_snr_loss]
        else:
            layer_cos_loss, layer_snr_loss = layer_losses[op.name]
        end_time = time.time()
        total_time += (end_time - start_time)
        eta = total_time / len(layer_losses) * (len(candidate_ops) - len(layer_losses))
        logger.info(f"Evaluated layer: {op.name}, cos loss: {layer_cos_loss:.6f}, snr loss: {layer_snr_loss:.6f}, eta: {eta:.2f}s")
    
    X = np.array([layer_losses[op.name] for op in candidate_ops])
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
    labels = kmeans.fit_predict(X)
    result = {op.name: label.item() for op, label in zip(candidate_ops, labels)}
    centroids = kmeans.cluster_centers_
    # target_cluster = np.argmax(centroids[:, 0])
    target_clusters = np.argsort(centroids[:, 0])[-nc+1:]
    
    selected_layers = [layer for layer, label in result.items() if label in target_clusters]
    logger.info(f"Selected {len(selected_layers)} layers for 8-bit weight quantization.")
    for layer in selected_layers:
        logger.info(f"Selected layer: {layer}")
    np.save(os.path.join(args.output_dir, 'kmeans_centroids.npy'), centroids)
    with open(os.path.join(args.output_dir, 'layer_clusters.json'), 'w') as f:
        json.dump(result, f, indent=4)
    with open(os.path.join(args.output_dir, 'layer_losses.json'), 'w') as f:
        json.dump(layer_losses, f, indent=4)
    with open(os.path.join(args.output_dir, 'w8_layers.txt'), 'w') as f:
        for layer_name in selected_layers:
            f.write(layer_name + '\n')
    return selected_layers


if __name__ == '__main__':
    args = parse_args()
    if args.model_type is not None:
        args.optim_transformer = True
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if len(args.w8_layers) == 1 and os.path.isfile(args.w8_layers[0]):
        with open(args.w8_layers[0], 'r') as f:
            args.w8_layers = [line.strip() for line in f.readlines()]
    setup_logger(args)

    rewrite_onnx_gelu(args.model)
    if args.optim_transformer:
        model_path = ('/').join(args.model.split('/')[:-1])
        args.infer_shape_dir = os.path.join(os.path.abspath(model_path), "infer_shape.onnx")
        onnx.shape_inference.infer_shapes_path(args.model, args.infer_shape_dir)
        args.optimzed_model_dir = os.path.join(args.output_dir, 'optim_model.onnx')
        os.system("python -m onnxruntime.transformers.optimizer \
                   --input {} --output {} --model_type={} \
                   --use_external_data_format --disable_packed_qkv \
                   --disable_packed_kv --use_gpu --disable_nhwc_conv \
                   --disable_bias_gelu --disable_skip_layer_norm"
                   .format(args.infer_shape_dir, args.optimzed_model_dir, args.model_type))
        
    args.optimzed_model_dir = os.path.join(args.output_dir, 'optim_model.onnx')
    logger.parent = None

    start = time.time()
    if args.optim_transformer:
        model = onnx.load(args.optimzed_model_dir)
        taken: set[str] = {n.name for n in model.graph.node if n.name}
        for node in model.graph.node:
            if not node.name:
                base = f"{node.op_type}"
                idx = 0
                new_name = base
                while new_name in taken:
                    idx += 1
                    new_name = f"{base}_{idx}"
                node.name = new_name
                taken.add(new_name)
    else:
        model = onnx.load(args.model)
        if model.opset_import[0].version < 13:
            model = onnx.version_converter.convert_version(model, 13)
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx_graph = ONNXGraph(model, args.output_dir, args.deploy, args.model_type)
    if not args.optim_transformer:
        try:
            onnx.checker.check_model(onnx_graph.model)
        except onnx.checker.ValidationError as e:
            logger.info("The onnx model is invalid:{}, please rectifie your model and restart Dipoorlet.".format(e))
            sys.exit()

    logger.info("Do tensor calibration...")
    act_clip_val, weight_clip_val = tensor_calibration(onnx_graph, args)
    if args.extra_calib_table is not None:
        logger.info("Update act clip val...")
        update_act_clip_val(act_clip_val, args.extra_calib_table)
    tensor_range = copy.deepcopy(act_clip_val)
    save_clip_val(act_clip_val, weight_clip_val, args,
        act_fname='act_clip_val.json',
        weight_fname='weight_clip_val.json')
    act_clip_val, weight_clip_val = load_clip_val(args)
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    logger.info("Search w8 layers...")
    search_w8_layers(onnx_graph, clip_val, logger, args)