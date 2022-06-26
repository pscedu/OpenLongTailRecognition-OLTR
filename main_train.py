import os
import sys
import argparse
import pprint
import logging
import torch
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import simplejson as json
import wandb

import run_networks
import utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--shuffler_dir', required=True, type=str)
    parser.add_argument('--rootdir', required=True, type=str)
    parser.add_argument('--train_db_file', required=True, type=str)
    parser.add_argument('--val_db_file', required=True, type=str)
    parser.add_argument('--encoding_file', required=True,
                        help='A json with name encodings, used for logging.')
    parser.add_argument('--init_weights_dir', type=str,
                        help='Load weight from here. Required only for stage 2.')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument(
        "--logging_level",
        type=int,
        choices=[10, 20, 30, 40],
        default=20,
        help="Set logging level. 10: debug, 20: info, 30: warning, 40: error.")
    parser.add_argument('--wandb_mode', 
        choices=['online', 'offline', 'disabled'], default='online')
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

    albumentation_tranform = A.Compose([
        A.CLAHE(),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=(0, 0.35),
                           rotate_limit=20,
                           p=1.,
                           border_mode=cv2.BORDER_REPLICATE),
        A.Blur(blur_limit=1),
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
    train_image_transform = lambda x: albumentation_tranform(image=x)['image']
    val_image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    used_keys = ['imagefile', 'image', 'objectid', 'name_id', 'name',
                 'x_on_page', 'width_on_page', 'y_on_page', 'height_on_page']
    common_transform_group={
            'name_id': lambda x: int(x),
            'x_on_page': lambda x: float(x) if x is not None and x != 'None' else -1,
            'y_on_page': lambda x: float(x) if x is not None and x != 'None' else -1,
            'width_on_page': lambda x: float(x) if x is not None and x != 'None' else -1,
            'height_on_page': lambda x: float(x) if x is not None and x != 'None' else -1,
        }
    train_transform_group = dict(common_transform_group)
    train_transform_group.update({'image': train_image_transform})
    val_transform_group = dict(common_transform_group)
    val_transform_group.update({'image': val_image_transform})

    train_dataset = datasets.ObjectDataset(
        args.train_db_file,
        rootdir=args.rootdir,
        mode='r',
        used_keys=used_keys,
        transform_group=train_transform_group)
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
        used_keys=used_keys,
        transform_group=val_transform_group)
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

    with open(args.encoding_file, 'r') as f:
        encoding = json.loads(f.read())
    model.name_encoding = encoding

    wandb.init(project="stamps", entity="etoropov", config=config, name='test2',
               mode=args.wandb_mode)

    # Log histogram of name_ids, in both train and test.
    popular_names_sql = ("SELECT name FROM objects WHERE name IN ("
                           "SELECT name FROM objects WHERE name NOT LIKE '%%page%%' "
                           "GROUP BY name ORDER BY COUNT(1) DESC LIMIT 10)")
    popular_train_names = [(x, "train") for x, in train_dataset.execute(popular_names_sql)]
    popular_val_names   = [(x, "val") for x, in val_dataset.execute(popular_names_sql)]
    popular_names = popular_train_names + popular_val_names
    popular_names_table = wandb.Table(data=popular_names, columns=['class_name', 'subset'])
    # wandb.log({'gt_names': wandb.plot.scatter(
    #     names_table, x='class_name', y='subset', title="Distribution of names.")})
    names_chart = wandb.plot_table(vega_spec_name="etoropov/gt_histogram",
                data_table=popular_names_table,
                fields={"x": "class_name", "y": "subset"})
    wandb.log({"names_chart": names_chart})
    model.set_names_of_interest([x[0] for x in list(set(popular_names))])
    print (model.set_names_of_interest)

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
