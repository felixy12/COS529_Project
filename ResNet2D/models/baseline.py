import os
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models import basenet
from models import load_data
import utils

class BasicModel():
    def __init__(self, opt):
        self.epoch = 0
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)
        self.best_acc = 0.

    def set_network(self, opt):
        if opt['input_type'] == 'latefusion':         
            print('Initializing model which takes in two frames')
            self.network = basenet.ResNet50(n_classes=opt['output_dim'],
                                            pretrained=True,
                                            dropout=opt['dropout']).to(self.device)
        elif opt['input_type'] == 'randomframe' or opt['input_type'] == 'centerframe':
            print('Initializing model which takes in one frame')
            self.network = basenet.ResNet50_base(n_classes=opt['output_dim'],
                                                 pretrained=True,         
                                                 dropout=opt['dropout']).to(self.device)

    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

    def set_data(self, opt):
        """Set up the dataloaders"""
        
        data_setting = opt['data_setting']
        if opt['input_type'] == 'latefusion':
            print('Feeding in first and last frames as input')
            loader_class = load_data.MyDataset_latefusion 
            dataset_kwargs = {}
        elif opt['input_type'] == 'randomframe':
            print('Feeding in a random frame as input')
            loader_class = load_data.MyDataset_singleframe
            dataset_kwargs = {'get_random':True} 
        elif opt['input_type'] == 'centerframe':
            print('Feeding in center frame as input')
            loader_class = load_data.MyDataset_singleframe
            dataset_kwargs = {'get_random':False}
        self.loader_train = load_data.create_dataset_actual(data_setting['path'], data_setting['train_params'], loader_class, kwargs=dataset_kwargs) 
        self.loader_test = load_data.create_dataset_actual(data_setting['path'], data_setting['test_params'],  loader_class, split='test', kwargs=dataset_kwargs)

        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer']( 
                            params=filter(lambda p: p.requires_grad, self.network.parameters()), 
                            lr=optimizer_setting['lr'],
                            weight_decay=optimizer_setting['weight_decay']
                            )
        
    def _criterion(self, output, target):
        return F.cross_entropy(output, target)
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        
        train_loss = 0
        output_list = []
        target_list = []
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            #print(images.shape)
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            #self.log_result('Train iteration', {'loss': loss.item()},
            #                len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}'.format(
                      self.epoch, i+1, len(loader), loss.item()), flush=True)
        
            output_list.append(outputs)
            target_list.append(targets)
        #self.log_result('Train epoch', {'loss': train_loss/len(loader)}, self.epoch)
        self.epoch += 1
        
        return train_loss, torch.cat(output_list), torch.cat(target_list)

    def _test(self, loader):
        """Compute model output on test set"""
        
        self.network.eval()

        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)
                target_list.append(targets)


        return test_loss, torch.cat(output_list), torch.cat(feature_list), torch.cat(target_list)

    def inference(self, output, detach=False):
        predict_prob = torch.sigmoid(output)
        if detach:
            return predict_prob.cpu().detach().numpy()
        return predict_prob.cpu().numpy()
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
    
    def train(self):
        """Train the model for one epoch, evaluate on validation set and 
        save the best model
        """
        
        start_time = datetime.now()
        train_loss, train_output, targets = self._train(self.loader_train)
        train_predict_prob = self.inference(train_output, True)
        train_acc = utils.get_accuracy(train_predict_prob, targets.cpu().numpy(), k=3)
        self.save_model(os.path.join(self.save_path, 'current.pth'))
        
        test_loss, test_output, _ , targets = self._test(self.loader_test)
        test_predict_prob = self.inference(test_output)
        test_acc = utils.get_accuracy(test_predict_prob, targets.cpu().numpy(), k=3)
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.save_model(os.path.join(self.save_path,'best.pth'))
       

        duration = datetime.now() - start_time
        print('Finish training epoch {}, time used: {}, train_acc: {}, test_acc: {}'.format(self.epoch, duration, train_acc, test_acc))

    
    def test(self):
        # Test and save the result

        test_loss, test_output, _ , targets= self._test(self.loader_test)
        test_predict_prob = self.inference(test_output)
        

        acc = utils.get_accuracy(test_predict_prob, targets.cpu().numpy(), k=3)
