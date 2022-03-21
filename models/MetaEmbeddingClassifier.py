import os
import torch
import torch.nn as nn
from models.CosNormClassifier import CosNorm_Classifier
from utils import *

import pdb

class MetaEmbedding_Classifier(nn.Module):
    
    def __init__(self, feat_dim, num_classes):
        super(MetaEmbedding_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)
        
    def forward(self, x, centroids, *args):
        
        # storing direct feature
        direct_feature = x.clone()

        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # set up visual memory
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone()
        
        # computing reachability
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh() 
        x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature
        
        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]
    
def create_model(in_dim=None, num_classes=None, stage1_weights=False, dataset=None, test=False, 
                 weights_path=None, extra_in_dim=0, **args):
    print('Loading Meta Embedding Classifier, extra_in_dim=%d' % extra_in_dim)
    assert in_dim is not None
    assert num_classes is not None
    # Add num_extra_input_features: feat_dim=in_dim+num_extra_input_features
    clf = MetaEmbedding_Classifier(feat_dim=(in_dim + extra_in_dim), num_classes=num_classes)

    if not test:
        if stage1_weights:
            assert dataset
            assert weights_path is not None
            print('Loading %s Stage 1 Classifier Weights from %s.' % (dataset, weights_path))
            assert(os.path.exists(weights_path))
            clf.fc_hallucinator = init_weights(model=clf.fc_hallucinator,
                                               weights_path=weights_path,
                                               classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
