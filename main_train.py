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
    parser.add_argument('--db_file', required=True, type=str)
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
    pprint.pprint(config)
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
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ObjectDataset(
        args.db_file,
        rootdir=args.rootdir,
        where_object="name NOT LIKE '%page%' AND "
        "objectid IN (SELECT objectid FROM properties WHERE key = 'name_id')",
        mode='r',
        used_keys=['image', 'objectid', 'name', 'name_id'],
        transform_group={
            'image': transform,
            'name_id': lambda x: int(x)
        })
    logging.info("Total number of samples: %d", len(dataset))

    # Set num_classes everywhere.
    # TODO: Set it in the training split, excluding the validation split.
    num_classes = dataset.execute(
        "SELECT COUNT(DISTINCT(value)) FROM properties WHERE key == 'name_id' AND value != '-1'"
    )[0][0]
    logging.info('num_classes: %d', num_classes)
    config['training_opt']['num_classes'] = num_classes
    config['networks']['classifier']['params']['num_classes'] = num_classes

    # open_set = [item for item in data if item["name_id"] == -1]

    validation_split = .2
    name_ids = dataset.execute(
        'SELECT value FROM properties WHERE key="name_id" ORDER BY objectid')
    train_val_indices = [
        i for i, name_id in enumerate(name_ids) if name_id != ("-1", )
    ]
    print(len(train_val_indices), validation_split,
          validation_split * len(train_val_indices))
    split = int(np.floor(validation_split * len(train_val_indices)))
    logging.info('Split at %d', split)

    # TODO: Need to make splits beforehand via Shuffler.
    train_indices = train_val_indices   # FIXME: ATTENTION: removed [split:]
    val_indices = train_val_indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    # print(train_indices)
    print("Training samples ", len(train_sampler))
    print("Validation samples ", len(valid_sampler))

    model = run_networks.model(config, test=False, init_weights_path=weights_path)
    logging.info('Created training model.')

    # TODO: replace batch_size in configs with command line.
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training_opt']['batch_size'],
        num_workers=config['training_opt']['num_workers'],
        sampler=train_sampler)

    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training_opt']['batch_size'],
        num_workers=config['training_opt']['num_workers'],
        sampler=valid_sampler)

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
