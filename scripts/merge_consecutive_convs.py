import argparse
import numpy as np
import onnx
from onnx import numpy_helper


def parser_args():
    parser = argparse.ArgumentParser(description='Merge consecutive 1x1 Conv layers in an ONNX model')
    parser.add_argument('--input', type=str, required=True, help='Path to the input ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Path to save the merged ONNX model')
    return parser.parse_args()

def get_conv_pairs(onnx_model):
    conv_pairs = []
    for i, node in enumerate(onnx_model.graph.node[:-1]):
        if node.op_type == 'Conv' and node.attribute[0].ints == [1, 1]:
            next_node = onnx_model.graph.node[i + 1]
            if next_node.op_type == 'Conv' and next_node.attribute[0].ints == [1, 1]:
                conv_pairs.append((node, next_node))

    # remove conv pairs that first conv's output is not second conv's input
    # and first conv's output is multiple convs' input
    valid_conv_pairs = []
    for conv1, conv2 in conv_pairs:
        conv1_output = conv1.output[0]
        conv2_input = conv2.input[0]
        if conv1_output == conv2_input:
            consumers = [n for n in onnx_model.graph.node if conv1_output in n.input]
            if len(consumers) == 1:
                valid_conv_pairs.append((conv1, conv2))
    return valid_conv_pairs


def merge_conv_layers(conv1, conv2, onnx_model):    
    # 提取权重和偏置
    conv1_input_name, conv1_weight_name, conv1_bias_name = conv1.input
    conv2_input_name, conv2_weight_name, conv2_bias_name = conv2.input
    
    conv1_name = conv1.name
    conv2_name = conv2.name
    conv1_weight, conv1_bias = None, None
    conv2_weight, conv2_bias = None, None

    for initializer in onnx_model.graph.initializer:
        if initializer.name == conv1_weight_name:
            conv1_weight = numpy_helper.to_array(initializer)
        elif initializer.name == conv1_bias_name:
            conv1_bias = numpy_helper.to_array(initializer)
        elif initializer.name == conv2_weight_name:
            conv2_weight = numpy_helper.to_array(initializer)
        elif initializer.name == conv2_bias_name:
            conv2_bias = numpy_helper.to_array(initializer)

    # 合并权重和偏置
    merged_weight = np.dot(conv2_weight.reshape(conv2_weight.shape[0], -1), 
                           conv1_weight.reshape(conv1_weight.shape[0], -1)).reshape(conv2_weight.shape[0], conv1_weight.shape[1], 1, 1)
    merged_bias = np.dot(conv2_weight.reshape(conv2_weight.shape[0], -1), conv1_bias) + conv2_bias
    
    # 创建新的卷积层
    name = conv1_name + '_merged'
    new_conv = onnx.helper.make_node(
        'Conv',
        inputs=[conv1_input_name, conv1_name + '.weight', conv1_name + '.bias'],
        outputs=conv2.output,
        name=name,
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1
    )
    
    # 替换原卷积层
    for i, node in enumerate(onnx_model.graph.node[:-1]):
        if node == conv1:
            conv1_index = i
            break
    onnx_model.graph.node.insert(conv1_index, new_conv)
    onnx_model.graph.node.remove(conv1)
    onnx_model.graph.node.remove(conv2)
    
    # 添加权重和偏置
    conv1_weight_initializer = onnx.helper.make_tensor(
        name=conv1_name + '.weight',
        data_type=onnx.TensorProto.FLOAT,
        dims=merged_weight.shape,
        vals=merged_weight.flatten().tolist()
    )
    conv1_bias_initializer = onnx.helper.make_tensor(
        name=conv1_name + '.bias',
        data_type=onnx.TensorProto.FLOAT,
        dims=merged_bias.shape,
        vals=merged_bias.flatten().tolist()
    )
    onnx_model.graph.initializer.append(conv1_weight_initializer)
    onnx_model.graph.initializer.append(conv1_bias_initializer)
    
    # 移除旧的权重和偏置
    onnx_model.graph.initializer.remove(next(init for init in onnx_model.graph.initializer if init.name == conv1_weight_name))
    onnx_model.graph.initializer.remove(next(init for init in onnx_model.graph.initializer if init.name == conv1_bias_name))
    onnx_model.graph.initializer.remove(next(init for init in onnx_model.graph.initializer if init.name == conv2_weight_name))
    onnx_model.graph.initializer.remove(next(init for init in onnx_model.graph.initializer if init.name == conv2_bias_name))
    return onnx_model

if __name__ == '__main__':
    args = parser_args()
    # 加载ONNX模型
    onnx_model = onnx.load(args.input)

    conv_pairs = get_conv_pairs(onnx_model)
    for conv1, conv2 in conv_pairs:
        print(f"Merging {conv1.name} and {conv2.name}")
        onnx_model = merge_conv_layers(conv1, conv2, onnx_model)

    # 保存合并后的模型
    onnx.save(onnx_model, args.output)