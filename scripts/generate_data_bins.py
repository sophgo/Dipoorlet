import argparse
import os
import os.path as osp
from random import shuffle

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class ResizeNormProcessor():
    def __init__(self, imgsz=224, mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262)):
        self.transform = transforms.Compose([
            transforms.Resize(imgsz),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, data_path, save_path):
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        im = Image.open(data_path).convert('RGB')
        im = self.transform(im)
        im.numpy().tofile(save_path)


DATASETS = {
    'imagenet': ResizeNormProcessor(),
    'coco2017': ResizeNormProcessor(640, (0., 0., 0.), (1., 1., 1.)),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=[_ for _ in DATASETS], help='Name of the dataset.')
    parser.add_argument('--data-root', type=str, required=True, help='Path to the dataset root.')
    parser.add_argument('--save-root', type=str, required=True, help='Path to save the processed bin files.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    processor = DATASETS[args.dataset]
    data_root = args.data_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    samples = os.listdir(data_root)
    shuffle(samples)
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        data_path = osp.join(data_root, sample)
        save_path = osp.join(save_root, f"{i}.bin")
        processor(data_path, save_path)