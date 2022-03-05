import os
import sys
import argparse
import pprint
import logging
import numpy as np
import torch
from torchvision import transforms

import run_networks
import utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--shuffler_dir', required=True, type=str)
    parser.add_argument('--rootdir', required=True, type=str)
    parser.add_argument('--train_db_file', required=True, type=str)
    parser.add_argument('--val_db_file', required=True, type=str)
    parser.add_argument('--init_weights_dir', type=str,
                        help='Load weight from here. Required only for stage 2.')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument(
        "--logging_level",
        type=int,
        choices=[10, 20, 30, 40],
        default=20,
        help="Set logging level. 10: debug, 20: info, 30: warning, 40: error.")
    return parser


def train(args):
    # TODO: Shall we use utils.source_import here too?
    sys.path.append(args.shuffler_dir)
    print('Adding to path: %s' % args.shuffler_dir)
    from interface.pytorch import datasets

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = utils.source_import(args.config).config
    # config['training_opt']['log_dir'] = args.output_dir
    save_model_dir = args.output_dir
    if args.init_weights_dir is not None:
        weights_path = os.path.join(args.init_weights_dir, 'final_model_checkpoint.pth')
        print ('Will load weights from %s' % weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError('Weights not found: %s' % weights_path)
    else:
        weights_path = None
        print ('Init weights dir not provided.')

    # TODO: add random rotation.
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ObjectDataset(
        args.train_db_file,
        rootdir=args.rootdir,
        mode='r',
        used_keys=['imagefile', 'image', 'objectid', 'name', 'name_id'],
        transform_group={
            'image': train_transform,
            'name_id': lambda x: int(x)
        })
    logging.info("Total number of train samples: %d", len(train_dataset))

    # Set num_classes everywhere.
    classes_ids = train_dataset.execute(
        "SELECT DISTINCT(value) FROM properties WHERE key == 'name_id'")
    classes_ids = [x[0] for x in classes_ids]
    num_classes = len(classes_ids)
    logging.info('num_classes: %d', num_classes)
    config['training_opt']['num_classes'] = num_classes
    config['networks']['classifier']['params']['num_classes'] = num_classes
    if 'FeatureLoss' in config['criterions']:
      # This one is only for stage 2.
      config['criterions']['FeatureLoss']['loss_params']['num_classes'] = num_classes

    # open_set = [item for item in data if item["name_id"] == -1]

    # Validation dataset needs to have classes present in training dataset.
    classes_ids_str = "'" + "', '".join(classes_ids) + "'"
    val_dataset = datasets.ObjectDataset(
        args.val_db_file,
        rootdir=args.rootdir,
        where_object="objectid IN "
                     "(SELECT objectid FROM properties WHERE key = 'name_id' AND value IN (%s))" % classes_ids_str,
        mode='r',
        used_keys=['imagefile', 'image', 'objectid', 'name', 'name_id'],
        transform_group={
            'image': val_transform,
            'name_id': lambda x: int(x)
        })
    logging.info("Total number of val samples: %d", len(val_dataset))

    pprint.pprint(config)
    model = run_networks.model(config, test=False, init_weights_path=weights_path)
    logging.info('Created training model.')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training_opt']['batch_size'],
        num_workers=config['training_opt']['num_workers'],
        shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training_opt']['batch_size'],
        num_workers=config['training_opt']['num_workers'])

    logging.info('Created dataloaders.')

    model.train(train_dataloader, val_dataloader, save_model_dir)


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=args.logging_level)

    train(args)
    print('ALL COMPLETED.')


if __name__ == '__main__':
    main()
