import os
import torch.nn as nn
from utils import *

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, feat_dim=None, num_classes=None, **args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x, None
    
def create_model(in_dim=None, num_classes=None, stage1_weights=False, dataset=None, test=False, 
                 weights_path=None, **args):
    assert in_dim is not None
    assert num_classes is not None
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(feat_dim=in_dim, num_classes=num_classes)

    if not test:
        if stage1_weights:
            assert dataset
            assert weights_path is not None
            print('Loading %s Stage 1 Classifier Weights from %s.' % (dataset, weights_path))
            assert(os.path.exists(weights_path))
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=weights_path,
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf