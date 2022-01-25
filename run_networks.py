import os
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import time
import numpy as np
np.set_printoptions(threshold=np.inf)
import json
import progressbar

from utils import source_import


class model ():

    TOP_N = 3

    def __init__(self, config, test, init_weights_path=None):
        '''
        Args:
          init_weights_path need to be passed when config['networks']['params']['stage1_weights'] is set to True.
        '''

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']

        self.init_models(test, init_weights_path)

        # Has to be called by user in order to make logic more straightforward:
        # this function has to be called during training, but not inference.
        # self.init_models(weights_path)

    def init_models(self, test, init_weights_path):
        logging.info('Initializng models...')

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        for key, val in networks_defs.items():
            logging.debug('key: %s, value: %s', key, str(val))

            # Networks
            def_file = val['def_file']
            model_args = list(val['params'].values())

            self.networks[key] = source_import(def_file).create_model(weights_path=init_weights_path, test=test, *model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)

            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for modulated attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'modulatedatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})

    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']

            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.scheduler_params['step_size'],
                                              gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def train(self, training_data, val_data, save_model_dir):
        self.test_mode = False

        # Initialize model optimizer and scheduler
        print('Initializing model optimizer.')
        self.scheduler_params = self.training_opt['scheduler_params']
        self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(
            self.model_optim_params_list)
        self.init_criterions()
        if self.memory['init_centroids']:
            self.criterions['FeatureLoss'].centroids.data = \
                self.centroids_cal(training_data)

        # When training the network
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']
        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            print('epoch: %d' % epoch)

            for model in self.networks.values():
                model.train()
                model.cuda()

            torch.cuda.empty_cache()

            # Iterate over dataset
            epoch_loss = 0.0
            for batch in progressbar.progressbar(training_data):

                # Break when step equal to epoch step
                # if step == self.epoch_steps:
                #     break

                inputs = batch['image']
                labels = batch['name_id']
                objectids = batch['objectid']

                inputs, labels = inputs.cuda(), labels.cuda()

                with torch.set_grad_enabled(True):

                    self.features, _ = self.networks['feat_model'](inputs)
                    feature_ext = False
                    if not feature_ext:
                        # During training, calculate centroids if needed to
                        if self.memory['use_centroids'] and 'FeatureLoss' in self.criterions.keys(
                        ):
                            self.centroids = self.criterions[
                                'FeatureLoss'].centroids.data
                        else:
                            self.centroids = None

                            # Calculate logits with classifier
                        self.logits, self.direct_memory_feature = self.networks[
                            'classifier'](self.features, self.centroids)

                    self.loss_perf = self.criterions['PerformanceLoss'](
                        self.logits,
                        labels) * self.criterion_weights['PerformanceLoss']

                    # Add performance loss to total loss
                    self.loss = self.loss_perf

                    # Apply loss on features if set up
                    if 'FeatureLoss' in self.criterions.keys():
                        self.loss_feat = self.criterions['FeatureLoss'](
                            self.features, labels)
                        self.loss_feat = self.loss_feat * self.criterion_weights[
                            'FeatureLoss']
                        self.loss += self.loss_feat

                    epoch_loss = epoch_loss + self.loss

                    self.model_optimizer.zero_grad()
                    if self.criterion_optimizer:
                        self.criterion_optimizer.zero_grad()
                    # Back-propagation from loss outputs
                    self.loss.backward()
                    self.model_optimizer.step()
                    if self.criterion_optimizer:
                        self.criterion_optimizer.step()

            _, preds = torch.max(self.logits, 1)
            epoch_accuracy = (preds == labels).sum().item() / len(labels)
            print("Epoch-train-loss:'", epoch_loss.item(),
                  " Epoch-train-accuracy:'", epoch_accuracy)

            # After every epoch, validation
            res = self.eval(val_data)

            # Under validation, the best model need to be updated
            if res > best_acc:
                best_epoch = copy.deepcopy(epoch)
                best_acc = copy.deepcopy(res)
                best_centroids = copy.deepcopy(self.centroids)
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                self.save_model(save_model_dir,
                                epoch,
                                best_epoch,
                                best_model_weights,
                                best_acc,
                                centroids=best_centroids)

        print('Training Complete.')
        print('Done')

    def eval(self, data, is_train_phase, openset=False):
        '''
        Evaluate the loaded model on the provided data. 
        This function is called after each epoch at training.
        Args:
          is_train_phase:   bool
        '''
        self.test_mode = False

        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' %
                  self.training_opt['open_threshold'])

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
            model.cuda()
        # # If on training phase, disable and later enable gradients
        # if is_train_phase:
        #     torch.set_grad_enabled(False)

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for valid in progressbar.progressbar(data):
            inputs = valid["image"].to(self.device)
            labels = valid["name_id"].to(self.device)

            # In validation or testing
            self.features, _ = self.networks['feat_model'](inputs)
            feature_ext = False
            # If not just extracting features, calculate logits
            if not feature_ext:

                # During training, calculate centroids if needed to
                if is_train_phase:
                    if self.memory['use_centroids'] and 'FeatureLoss' in self.criterions.keys(
                    ):
                        self.centroids = self.criterions[
                            'FeatureLoss'].centroids.data
                    else:
                        self.centroids = None

                # Calculate logits with classifier
                self.logits, self.direct_memory_feature = self.networks[
                    'classifier'](self.features, self.centroids)

                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
        #        self.total_paths = np.concatenate((self.total_paths, paths))

        _, pred = self.total_logits.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(self.total_labels.view(1, -1).expand_as(pred))

        #print(correct.shape)

        correct_k = correct[:5].reshape(-1).float().sum(0)
        res = (correct_k.mul_(100.0 / self.total_labels.size(0))).item()
        print("Eval-Accuracy :", res)
        return res

    def centroids_cal(self, data):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        for model in self.networks.values():
            model.eval()

        with torch.set_grad_enabled(False):

            for p in progressbar.progressbar(data):
                inputs, labels = p["image"].to(self.device), p["name_id"].to(
                    self.device)

                # Calculate Features of each training data
                self.features, _ = self.networks['feat_model'](inputs)

                feature_ext = True
                # If not just extracting features, calculate logits
                if not feature_ext:

                    # During training, calculate centroids if needed to
                    if not self.test_mode:
                        if centroids and 'FeatureLoss' in self.criterions.keys(
                        ):
                            self.centroids = self.criterions[
                                'FeatureLoss'].centroids.data
                        else:
                            self.centroids = None

                    # Calculate logits with classifier
                    self.logits, self.direct_memory_feature = self.networks[
                        'classifier'](self.features, self.centroids)

                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def load_model(self, weights_path):
        '''
        Has to be called by user at the INFERENCE stage after the constructor.
        '''
        print('Loading model from %s' % (weights_path))

        checkpoint = torch.load(weights_path)
        model_state = checkpoint['state_dict_best']

        if 'centroids' in checkpoint:
            print ('Found centroids in checkpoint.')
            self.centroids = checkpoint['centroids']
        else:
            print ('No centroids in checkpoint.')
            self.centroids = None

        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {
                k: weights[k]
                for k in weights if k in model.state_dict()
            }
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)

    def save_model(self,
                   model_dir,
                   epoch,
                   best_epoch,
                   best_model_weights,
                   best_acc,
                   centroids=None):

        model_states = {
            'epoch': epoch,
            'best_epoch': best_epoch,
            'state_dict_best': best_model_weights,
            'best_acc': best_acc,
            'centroids': centroids
        }

        print('Saving model at: %s' % model_dir)
        # TODO: probably need to save for every epoch.
        model_path = os.path.join(model_dir, 'final_model_checkpoint.pth')

        torch.save(model_states, model_path)

    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'],
                                'logits_%s' % ('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename,
                 logits=self.total_logits.detach().cpu().numpy(),
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)

    def infer(self, unlabeled_data):
        time.sleep(0.25)

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
            model.cuda()

        # The full dataset.
        full_objectids = None  # Lazy init.
        full_probs = None  # Lazy init.
        full_preds = None  # Lazy init.

        # with torch.set_grad_enabled(False):   # <- not needed since model.eval() has been called.
        for batch in progressbar.progressbar(unlabeled_data):
            if batch is None:
                assert 0, 'How did this happen?'
            inputs = batch["image"].to(self.device)
            object_ids = batch['objectid'].cpu().numpy()
            # TODO: is it needed?
            # inputs = Variable(inputs)

            # In validation or testing
            features, _ = self.networks['feat_model'](inputs)
            logits, _ = self.networks['classifier'](features, self.centroids)
            #self.total_logits = torch.cat((self.total_logits, self.logits))

            probs, preds = F.softmax(logits.detach(), dim=1).topk(k=3,
                                                                    dim=1)

            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()

            # Openset is thresholded at postprocessing.
            # Unthresholded preds are needed for ROC curves.
            # preds[probs < self.training_opt['open_threshold']] = -1

            full_objectids = np.concatenate((full_objectids, object_ids)) if full_objectids is not None else object_ids
            full_probs = np.concatenate((full_probs, probs)) if full_probs is not None else probs
            full_preds = np.concatenate((full_preds, preds)) if full_preds is not None else preds

        return full_objectids, full_preds, full_probs