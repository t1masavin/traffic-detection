import argparse
import os

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model import FasterRCNN


def to_script(n_classes, checkpoint_path, save_path):
    det_model = FasterRCNN(n_classes)
    det_model = det_model.load_from_checkpoint(checkpoint_path=checkpoint_path, n_classes=n_classes)
    det_model.eval()
    det_model = det_model.to('cpu')

    st_dict = det_model.state_dict()

    detector = fasterrcnn_resnet50_fpn_v2()
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    st_dict = {key.removeprefix('detector.'): value for key, value in st_dict.items()}
    detector.load_state_dict(st_dict)
    detector.eval()
    traced_model = torch.jit.script(detector)
    path = os.path.join(save_path, 'model.pt')
    # traced_model([images]) input: list of images with shape(C, H, W)
    traced_model.save(path)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='From Pytorch Lightning to torch.script.')
    parser.add_argument(
        '--n_classes',
        help='Number of detection classes.',
        type=int,
        default=3
    )
    parser.add_argument(
        '--checkpoint_path',
        help='Path to the .ckpt model file.',
        type=str,
        default='model.ckpt'

    )
    parser.add_argument(
        '--save_path',
        help='Location to save the torch.script model.',
        type=str,
        default='model.pt'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    n_classes = args.n_classes + 1
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    to_script(n_classes, checkpoint_path, save_path)
