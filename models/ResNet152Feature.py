import os
from models.ResNetFeature import *
from utils import *
        
def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, caffe=False, test=False,
                 weights_path=None):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None, fc_feat_dim=2048)
    
    if not test:

        assert(caffe != stage1_weights)

        if caffe:
            print('Loading Caffe Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./pretrained/caffe_resnet152.pth',
                                     caffe=True)
        elif stage1_weights:
            assert dataset
            assert weights_path is not None
            print('Loading %s Stage 1 ResNet 152 Weights from %s.' % (dataset, weights_path))
            assert(os.path.exists(weights_path))
            resnet152 = init_weights(model=resnet152,
                                     weights_path=weights_path)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet152
