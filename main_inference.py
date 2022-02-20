import os
import sys
import argparse
import logging
import torch
import pprint
import shutil
import simplejson as json
from torchvision import transforms
import progressbar

import run_networks
import utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='config/stamps/stage_2_meta_embedding.py',
                        type=str)
    parser.add_argument('--shuffler_dir', required=True, type=str)
    parser.add_argument('--encoding_file', required=True, type=str)
    parser.add_argument('--rootdir', required=True, type=str)
    parser.add_argument('--weights_dir', required=True, type=str)
    parser.add_argument('--in_db_file', required=True, type=str)
    parser.add_argument('--out_db_file', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=50)
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

    # Read the encoding, and pprepare the decoding table.
    if not os.path.exists(args.encoding_file):
        raise FileNotFoundError('Cant find the encoding file: %s' % args.encoding_file)
    with open(args.encoding_file) as f:
        encoding = json.load(f)
    decoding = {}
    for name, name_id in encoding.items():
        if name_id == -1:
            decoding[name_id] = None
        elif name_id in decoding:
            raise ValueError('Not expecting multiple back mapping.')
        else:
            decoding[name_id] = name
    logging.info('Have %d entries in decoding.', len(decoding))


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
                                     transform_group={'image': transform},
                                     copy_to_memory=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=1)

    # Need to load checkpoint here, because it stores info about num_classes.
    weights_path = os.path.join(args.weights_dir, 'final_model_checkpoint.pth')
    print('Loading model from %s' % weights_path)
    checkpoint = torch.load(weights_path)

    num_classes = checkpoint['num_classes']
    config['training_opt']['num_classes'] = num_classes
    config['networks']['classifier']['params']['num_classes'] = num_classes
    model = run_networks.model(config, test=True)
    model.load_model(checkpoint)

    objectids, preds, probs = model.infer(dataloader)

    # Record the results.
    for objectid, pred, prob in progressbar.progressbar(zip(objectids, preds, probs)):
        # NOTE: Currently writing only top-1.
        objectid = int(objectid)
        name_id = int(pred[0])
        score = float(prob[0])
        if name_id not in decoding:
            raise ValueError('name_id %d not in decoding.')
        name = decoding[name_id]
        print ('Setting name_id %s (name_id %d) with score %.3f to object %d' % (name, name_id, score, objectid))
        dataset.execute('UPDATE objects SET name=?,score=? WHERE objectid=?', (name, score, objectid))
    dataset.conn.commit()
    print('Found distinct names: ', dataset.execute('SELECT COUNT(DISTINCT(name)) FROM objects'))
    dataset.close()


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=args.logging_level)

    inference(args)
    print('ALL COMPLETED.')


if __name__ == '__main__':
    main()
