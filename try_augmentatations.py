import os
import sys
import argparse
import progressbar
import logging
import multiprocessing
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffler_dir', required=True, type=str)
    parser.add_argument('--rootdir', required=True, type=str)
    parser.add_argument('--db_file', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--max_transforms', default=10, type=int)
    parser.add_argument('--max_samples', default=100, type=int)
    parser.add_argument(
        "--logging_level",
        type=int,
        choices=[10, 20, 30, 40],
        default=20,
        help="Set logging level. 10: debug, 20: info, 30: warning, 40: error.")
    return parser


def TryAugmentations(args):
    sys.path.append(args.shuffler_dir)
    print('Adding to path: %s' % args.shuffler_dir)
    from interface.pytorch import datasets

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    A_transform = A.Compose([
        A.CLAHE(),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=(0, 0.35),
                           rotate_limit=20,
                           p=1.,
                           border_mode=cv2.BORDER_REPLICATE),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.Resize(224, 224),
        A.HueSaturationValue(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    def transform_group(sample):
        ''' Apply an augmentation tranform to each sample and save to file. '''
        image = A_transform(image=sample['image'])
        imagename = os.path.basename(sample['imagefile'])

        output_dir = os.path.join(args.output_dir, imagename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, '%02d.jpg' % itransform)
        cv2.imwrite(path, cv2.cvtColor(image['image'], cv2.COLOR_BGR2RGB))
        return imagename

    dataset = datasets.ObjectDataset(args.db_file,
                                     rootdir=args.rootdir,
                                     mode='r',
                                     used_keys=['image', 'imagefile', 'name'],
                                     transform_group=transform_group)

    num_workers = multiprocessing.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=num_workers,
                                             num_workers=num_workers,
                                             shuffle=False)

    for itransform in progressbar.progressbar(range(args.max_transforms)):
        for ibatch, _ in enumerate(dataloader):
            if ibatch >= args.max_samples / num_workers:
                break


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=args.logging_level)

    TryAugmentations(args)
    print('ALL COMPLETED.')


if __name__ == '__main__':
    main()
