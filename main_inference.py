import os
import sys
import argparse
import logging
import torch
import pprint
import shutil
from torchvision import transforms

import run_networks
import utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='config/stamps/stage_2_meta_embedding.py',
                        type=str)
    parser.add_argument('--shuffler_dir', required=True, type=str)
    parser.add_argument('--rootdir', required=True, type=str)
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--in_db_file', required=True, type=str)
    parser.add_argument('--out_db_file', required=True, type=str)
    # parser.add_argument('--log_dir', required=True, type=str)
    parser.add_argument(
        "--logging_level",
        type=int,
        choices=[10, 20, 30, 40],
        default=20,
        help="Set logging level. 10: debug, 20: info, 30: warning, 40: error.")
    return parser


def inference(args):
    # TODO: Shall we use utils.source_import here too?
    sys.path.append(args.shuffler_dir)
    print('Adding to path: %s' % args.shuffler_dir)
    from interface.pytorch import datasets

    config = utils.source_import(args.config).config
    # config['training_opt']['log_dir'] = args.output_dir
    pprint.pprint(config)
    training_opt = config['training_opt']

    # TODO: change.
    relatin_opt = config['memory']

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    shutil.copyfile(args.in_db_file, args.out_db_file)

    dataset = datasets.ObjectDataset(args.out_db_file,
                                     rootdir=args.rootdir,
                                     mode='w',
                                     used_keys=['image', 'objectid'],
                                     transform_group={'image': transform})

    # TODO: replace batch_size in with command line.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1)

    model = run_networks.model(config)
    model.load_model(args.model_dir)

    objectids, preds, probs = model.infer(dataloader)

    # Record the results.
    c = dataset.cursor
    for objectid, pred, prob in zip(objectids, preds, probs):
        # NOTE: Currently writing only top-1.
        name_id = pred[0]
        score = prob[0]
        c.execute('UPDATE objects SET name=?,score=? WHERE objectid=?',
                  (str(name_id), score, objectid))
    dataset.conn.commit()


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=args.logging_level)

    inference(args)
    print('ALL COMPLETED.')


if __name__ == '__main__':
    main()
