import argparse
import onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, help='Path to the ONNX model file', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = onnx.load(args.onnx)

    print("Inputs:")
    for inp in model.graph.input:
        name = inp.name
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in inp.type.tensor_type.shape.dim]
        print(f"  name: {name}, shape: {shape}")

    print("Outputs:")
    for out in model.graph.output:
        name = out.name
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in out.type.tensor_type.shape.dim]
        print(f"  name: {name}, shape: {shape}")