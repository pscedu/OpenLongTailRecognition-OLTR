import os
from models.ResNetFeature import *
from utils import *
        
def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, test=False, 
                 weights_path=None, **args):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(BasicBlock, [1, 1, 1, 1], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None, fc_feat_dim=512)

    if not test:
        if stage1_weights:
            assert dataset
            assert weights_path is not None
            print('Loading %s Stage 1 ResNet 10 Weights from %s.' % (dataset, weights_path))
            assert(os.path.exists(weights_path))
            resnet10 = init_weights(model=resnet10,
                                    weights_path=weights_path)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet10
