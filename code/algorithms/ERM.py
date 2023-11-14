import sys
sys.path.append(sys.path[0].replace('algorithms', ''))
import torch
import torch.nn as nn
from configs.default import imagenet_pretrain_transform_test, imagenet_pretrain_transform_train, num_classes_dict, log_dir_path
from models import resnet18, resnet50
from models.optimizer.SAM import SAM
from algorithms.base_trainer import Base_Trainer
import os
import shutil
import time
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from utils.metrics import Classification
from utils.logger import Get_Logger
from data import *
from tqdm import tqdm


class ERM_Trainer(Base_Trainer):
    def __init__(self, args) -> None:
        self.trainer_name = self.__class__.__name__
        self.args = args
        if self.args.sub_log_dir == 'none':
            self.log_dir_path = log_dir_path
        else:
            self.log_dir_path = os.path.join(log_dir_path, self.args.sub_log_dir + '/')
            os.makedirs(self.log_dir_path, exist_ok=True)
            
        self.get_log()
        self.log_file.info(self.args)
        self.set_seed(self.args.seed)
        self.initilize()
        self.log_file.info(f'{self.trainer_name} initilize done')
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def get_log_name(self):
        args = self.args
        file_name = self.trainer_name
        start_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime())
        backbone = args.backbone
        backbone = backbone.replace('/', '-')
        log_name = f"{start_time}-{file_name}-{args.dataset}-{backbone}-{args.test_domain}"\
            + f"-{args.lr}-bs{args.batch_size}-r{args.epochs}"\
            +f"-{args.note}"
        return log_name
    
    def get_log_dir(self):
        # 根据args生成log_name
        log_name = self.get_log_name()
        log_dir = self.log_dir_path + log_name + '/' # 组合绝对路径    
        os.makedirs(log_dir) # 新建文件夹
        tensorboard_dir = log_dir + '/tensorboard/'
        os.makedirs(tensorboard_dir)
        return log_dir, tensorboard_dir

    
    def get_log(self):
        log_dir, tensorboard_dir = self.get_log_dir()
        self.log_dir = log_dir
        self.save_dir = log_dir + 'checkpoints/'
        os.makedirs(self.save_dir)
        self.tensorboard_dir = tensorboard_dir
        self.log_ten = SummaryWriter(log_dir=tensorboard_dir)
        self.log_file = Get_Logger(file_name=log_dir + 'train.log', display=self.args.display)
        
    def initilize(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.scheduler = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = self.args.epochs
        self.test_domain = self.args.test_domain
        
        self.get_data_aug()
        self.get_model()
        self.get_data()
        self.get_optimizer()
        self.get_metric()
        
        
    def get_data(self):
        if self.args.dataset == 'pacs':
            self.data_obj = PACS_DG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
        elif self.args.dataset == 'domainnet':
            self.data_obj = DomainNet_DG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
        elif self.args.dataset == 'domainnet_open':
            self.data_obj = DomainNet_OpenDG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
        elif self.args.dataset == 'officehome':
            self.data_obj = OfficeHome_DG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, seed=self.args.dataset_seed, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
        elif self.args.dataset == 'officehome_open':
            self.data_obj = OfficeHome_OpenDG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, seed=self.args.dataset_seed, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
        elif self.args.dataset == 'terrainc':
            self.data_obj = TerraInc_DG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, seed=self.args.dataset_seed, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
        elif self.args.dataset == 'vlcs':
            self.data_obj = VLCS_DG(test_domain=self.args.test_domain, batch_size=self.args.batch_size, val_batch_size=self.args.val_batch_size, transform_train=self.train_transform, transform_test=self.test_transform)
            self.mask_idx = {}
            self.mask_idx['a'] = list(set(range(1000)).difference(set(self.data_obj.class_idx_with_domain['a'])))
            self.mask_idx['r'] = list(set(range(1000)).difference(set(self.data_obj.class_idx_with_domain['r'])))
        else:
            raise ValueError('dataset not supported')
        
        self.dataloaders_dict, self.datasets_dict = self.data_obj.get_data()
        self.domain_text = self.data_obj.domain_text
        self.class_text = self.data_obj.class_text
        self.domain_list = self.data_obj.domain_list

    def get_model(self):
        self.train_transform = imagenet_pretrain_transform_train
        self.test_transform = imagenet_pretrain_transform_test
        
        if self.args.backbone == 'resnet18':
            self.model = resnet18(pretrained=True, num_classes=num_classes_dict[self.args.dataset])
        elif self.args.backbone == 'resnet50':
            self.model = resnet50(pretrained=True, num_classes=num_classes_dict[self.args.dataset]) 
        else:
            raise ValueError('backbone not supported')
        self.model.to(self.device)

        
    def get_optimizer(self):
        if self.args.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        elif self.args.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=False, dampening=0)
            
        if self.args.lr_policy == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
        elif self.args.lr_policy == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
    
            
    def get_metric(self):
        self.metric = Classification()
    
    def save_checkpoint(self, epochs, model, results_dict, best_acc, save_dir, is_best=False, prefix=''):
        '''保存global model'''
        state = {
        'args': self.args,
        'epochs': epochs,
        'model': model.state_dict(),
        'results': results_dict,
        'best_acc': best_acc,
        }
        torch.save(state, save_dir + prefix + 'last_checkpoint.pth')
        if is_best:
            shutil.copyfile(save_dir + prefix + 'last_checkpoint.pth', save_dir + prefix + 'model_best.pth')
            
    def train(self, n_epoch, dataloader, model=None, optimizer=None, prefix='merged'):
        if model is None:
            model = self.model
            optimizer = self.optimizer
            
        model.train()

        for i, data_list in tqdm(enumerate(dataloader)):
            
            if len(data_list) == 3:
                imgs, labels, domain_labels = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            if self.data_aug is not None:
                imgs = self.data_aug(imgs)
            output = model(imgs)
            loss = self.criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_train_loss', results_dict['loss'], n_epoch)
    

    def val(self, n_epoch, domain_name, dataloader, model=None, prefix='val'):
        if model is None:
            model = self.model
        
        model.eval()
        with torch.no_grad():
            for i, data_list in enumerate(dataloader):
                if len(data_list) == 3:
                    imgs, labels, domain_labels = data_list
                else:
                    imgs, labels = data_list
                imgs = imgs.to(self.device)
                output = model(imgs)
                if self.args.dataset == 'imagenet' and domain_name in ['r', 'a']:
                    output[:,self.mask_idx[domain_name]] = 0.
                self.metric.update(output, labels)
                
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_{domain_name}_loss', results_dict['loss'], n_epoch)
        self.log_ten.add_scalar(f'{prefix}_{domain_name}_acc', results_dict['acc'], n_epoch)
        self.log_file.info(f'{prefix} Epoch: {n_epoch:3d} | Domain: {domain_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

        return results_dict
    
    def run(self):
        self.total_results_dict = {}
        self.best_acc = 0.
        for i in range(self.epochs):
            self.current_epoch = i        
            self.total_results_dict[self.current_epoch] = {}
            self.train(i, self.dataloaders_dict['merged']['train'], self.model, self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()

            val_results = self.val(i, 'lodo', self.dataloaders_dict['merged']['val'], self.model, prefix='val') # lodo: leave-one-domain-out
            self.total_results_dict[self.current_epoch]['val'] = val_results
            is_best = val_results['acc'] > self.best_acc
            self.best_acc = max(val_results['acc'], self.best_acc)
            if is_best:
                self.log_file.info(f'Get Best acc: {self.best_acc*100:.2f}% on epoch {i}')
                
            test_results = self.val(i, self.test_domain, self.dataloaders_dict[self.test_domain]['test'], self.model, prefix='unseen_test')
                
            self.total_results_dict[self.current_epoch]['test'] = test_results

            self.save_checkpoint(i, self.model, self.total_results_dict, self.best_acc, self.save_dir, is_best=is_best, prefix='lodo_')
    
        
class SAM_Trainer(ERM_Trainer):
    def get_optimizer(self):
        if self.args.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        elif self.args.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        elif self.args.optim == 'sam' or self.args.optim == 'sam_domain':
            self.optimizer = SAM(self.model.parameters(), base_optimizer=torch.optim.SGD, lr=self.args.lr, rho=self.args.rho, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=False, dampening=0)
        elif self.args.optim == 'sam_adam':
            self.optimizer = SAM(self.model.parameters(), base_optimizer=torch.optim.Adam, lr=self.args.lr, rho=self.args.rho, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        elif self.args.optim == 'sam_adamw':
            self.optimizer = SAM(self.model.parameters(), base_optimizer=torch.optim.AdamW, lr=self.args.lr, rho=self.args.rho, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=False, dampening=0)
        if self.args.lr_policy == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
        elif self.args.lr_policy == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
            
    def sam_train(self, n_epoch, dataloader, model=None, optimizer=None, prefix='merged'):
        if model is None:
            model = self.model
            optimizer = self.optimizer
        model.eval()


        for i, data_list in tqdm(enumerate(dataloader)):
            
            if len(data_list) == 3:
                imgs, labels, domain_labels = data_list
            else:
                imgs, labels = data_list
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            if self.data_aug is not None:
                imgs = self.data_aug(imgs)
            
            if self.args.optim in ['sagm', 'gsam']:
                optimizer.set_closure(self.criterion, imgs, labels)
                output, loss = optimizer.step()
                optimizer.update_rho_t()
            else: # 标准SAM
                output = model(imgs)
                loss = self.criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                output = model(imgs)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)

            self.metric.update(output, labels)
        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_train_loss', results_dict['loss'], n_epoch)
    
    def run(self):
        self.total_results_dict = {}
        self.best_acc = 0.
        for i in range(self.epochs):
            self.current_epoch = i        
            self.total_results_dict[self.current_epoch] = {}
            if 'sam' in self.args.optim or 'sagm' in self.args.optim:
                self.sam_train(i, self.dataloaders_dict['merged']['train'], self.model, self.optimizer)
            else:
                self.train(i, self.dataloaders_dict['merged']['train'], self.model, self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()

            val_results = self.val(i, 'lodo', self.dataloaders_dict['merged']['val'], self.model, prefix='val') # lodo: leave-one-domain-out
            self.total_results_dict[self.current_epoch]['val'] = val_results
            is_best = val_results['acc'] > self.best_acc
            self.best_acc = max(val_results['acc'], self.best_acc)
            if is_best:
                self.log_file.info(f'Get Best acc: {self.best_acc*100:.2f}% on epoch {i}')
                
            test_results = self.val(i, self.test_domain, self.dataloaders_dict[self.test_domain]['test'], self.model, prefix='unseen_test')
                
            self.total_results_dict[self.current_epoch]['test'] = test_results

            self.save_checkpoint(i, self.model, self.total_results_dict, self.best_acc, self.save_dir, is_best=is_best, prefix='lodo_')
    
    









