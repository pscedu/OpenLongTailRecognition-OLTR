import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.metrics import f1_score
import importlib
import pdb
import wandb
import math
import cv2

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model

def shot_acc (preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):
    
    training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))          
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
        
def F_measure(preds, labels, openset=False, theta=None):
    
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1

def class_count(data):
    return [
        x[0] for x in data.dataset.execute(
            "SELECT COUNT(1) FROM properties WHERE key == 'name_id' AND value != '-1' "
            "GROUP BY CAST(value AS INT) ORDER BY CAST(value AS INT)")
    ]

def top_k_accuracy(logits, labels, k):
    ''' Computes top-k accuracy:
    Args:
      logits:    predicted probabilities of classes. torch.tensor of size (npoints,nclasses).
      labels:    ground thuth labels. torch.tensor of size (npoints,).
    Returns:
      number in range [0, 1].
    '''
    assert len(logits.shape) == 2, logits.shape
    assert len(labels.shape) == 1, labels.shape
    assert logits.shape[0] == labels.shape[0], (logits.shape, labels.shape)
    assert k < logits.shape[1], (k, logits.shape)

    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    num_correct_topk = correct.reshape(-1).float().sum(0)
    return (num_correct_topk.mul_(100.0 / labels.size(0))).item()

def get_tp_fp_fn(logits, labels):
    '''
    Compute the number of true positives, false positives, and false negatives.
    Args:
      logits:    predicted probabilities of classes. torch.Tensor of size (npoints,nclasses).
      labels:    ground thuth labels. torch.Tensor of size (npoints,).
    Returns:
      tp, fp, fn:  np.arrays of size (nclasses,)
    '''
    assert torch.is_tensor(logits), type(logits)
    assert torch.is_tensor(labels), type(labels)
    assert len(logits.shape) == 2, logits.shape
    assert len(labels.shape) == 1, labels.shape
    assert logits.shape[0] == labels.shape[0], (logits.shape, labels.shape)
    num_classes = logits.shape[1]

    _, pred = logits.topk(1, dim=1, largest=True)
    pred = pred.flatten()
    assert pred.shape == labels.shape

    tp = torch.zeros((num_classes,), dtype=int)
    fp = torch.zeros((num_classes,), dtype=int)
    fn = torch.zeros((num_classes,), dtype=int)
    for c in range(num_classes):
        tp[c] = torch.logical_and(labels == c, pred == c).sum()
        fp[c] = torch.logical_and(labels != c, pred == c).sum()
        fn[c] = torch.logical_and(labels == c, pred != c).sum()

    assert fp.sum() == fn.sum(), \
        'Every FP for one class is FN for another, %d vs %d.' % (fp.sum(), fn.sum())
    assert tp.sum() + fp.sum() == logits.shape[0], \
        'Each prediction is either correct or not, %d + %d = %d.' % (tp.sum(), fp.sum(), logits.shape[0])

    return tp, fp, fn


def build_tp_fp_fn_wandb_chart(logits, labels, name_ids, names):
    '''
    Make a chart that can be logged into wandb.
    Args:
      logits:    predicted probabilities of classes. torch.Tensor of size (npoints,nclasses).
      labels:    ground thuth labels. torch.Tensor of size (npoints,).
    Returns:
      Chart from preset etoropov/tf_fp_fn.
    Usage:
      wandb.log({"TP/FP/FN": build_tp_fp_fn_wandb_chart(logits, labels, name_ids, names)})
    '''
    assert torch.is_tensor(logits), type(logits)
    assert torch.is_tensor(labels), type(labels)
    tp, fp, fn = get_tp_fp_fn(logits, labels)
    tp = tp[name_ids]
    fp = fp[name_ids]
    fn = fn[name_ids]
    tp_entries = [(x, name, 'tp') for x, name in zip(tp, names)]
    fp_entries = [(x, name, 'fp') for x, name in zip(fp, names)]
    fn_entries = [(x, name, 'fn') for x, name in zip(fn, names)]
    entries = tp_entries + fp_entries + fn_entries
    print ('build_tp_fp_fn_wandb_chart', len(entries))
    tp_fp_fn_table = wandb.Table(data=entries, columns=['quantity', 'class_name', 'variety'])
    tp_fp_fn_chart = wandb.plot_table(vega_spec_name="etoropov/tf_fp_fn",
        data_table=tp_fp_fn_table,
        fields={"x": "quantity", "y": "class_name", 'color': 'variety'})
    return tp_fp_fn_chart


def get_confusion_matrix(logits, labels, N):
    ''' Computes the confusion matrix.
    Args:
      logits:    predicted probabilities of classes. torch.tensor of size (npoints,nclasses).
      labels:    ground thuth labels. torch.tensor of size (npoints,).
      N:         number of classes.
    Returns:
      Matrix NxN. The element at row R, col C is the number of labels R 
                 predicted as C, normalized to the number of labels R.
    '''
    assert len(logits.shape) == 2, logits.shape
    assert len(labels.shape) == 1, labels.shape
    assert logits.shape[0] == labels.shape[0], (logits.shape, labels.shape)
    
    _, pred = logits.topk(1, dim=1, largest=True, sorted=True)
    print(pred.shape)

    confusion_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        confusion_matrix[labels[i], pred[i]] += 1
    
    # Normalize to the number of labels of each class.
    for i in range(N):
        confusion_matrix[i, :] /= np.sum(confusion_matrix[i, :])

    return confusion_matrix

def make_grid_with_labels(tensor, labels, nrow=8, limit=20, padding=2, value_range=(-3, 3),
                          scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (list):  ( [labels_1,labels_2,labels_3,...labels_n]) where labels is Bx1 vector of some labels
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if not isinstance(labels, list):
        raise ValueError
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit]

    font = 1
    fontScale = 3
    color = (0, 255, 0)
    thickness = 2

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if value_range is not None:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            working_tensor = tensor[k]
            if labels is not None:
                org = (0, tensor[k].shape[1])
                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.numpy(), (1, 2, 0)) * 255).astype('uint8'))
                image = cv2.putText(working_image, f'{str(labels[k])}', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = torchvision.transforms.ToTensor()(image.get())
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid

# def dataset_dist (in_loader):

#     """Example, dataset_dist(data['train'][0])"""
    
#     label_list = np.array([x[1] for x in in_loader.dataset.samples])
#     total_num = len(data_list)

#     distribution = []
#     for l in np.unique(label_list):
#         distribution.append((l, len(label_list[label_list == l])/total_num))
        
#     return distribution

