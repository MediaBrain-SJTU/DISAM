import sys
sys.path.append(sys.path[0].replace('code/data', 'code'))

import os
import torch
from data.utils import MetaDataset, MetaDGDataset
from configs.default import officehome_path, remove_underline, default_transform_train, default_transform_test
import random
officehome_name_dict = {
    'p': 'Product',
    'a': 'Art',
    'c': 'Clipart',
    'r': 'Real_World',
}


class OfficeHome_SingleDomain():
    def __init__(self, root_path=officehome_path, domain_name='p', split='train', train_transform=None, seed=0):
        self.domain_name = domain_name
        assert domain_name in officehome_name_dict.keys(), 'domain_name must be in {}'.format(officehome_name_dict.keys())
        self.root_path = root_path
        self.domain = officehome_name_dict[domain_name]
        domain_list = list(officehome_name_dict.keys())
        domain_list.sort()
        self.domain_label = domain_list.index(domain_name)
        self.txt_path = os.path.join(root_path, '{}_img_label_list.txt'.format(self.domain))
        
        self.split = split
        assert self.split in ['train', 'val', 'test'] , 'split must be train, val or test'
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = default_transform_test
        self.seed = seed
        
        self.imgs, self.labels = OfficeHome_SingleDomain.read_txt(self.txt_path)
        
        if self.split == 'train' or self.split == 'val':
            random.seed(self.seed)
            train_img, val_img = OfficeHome_SingleDomain.split_list(self.imgs, 0.9)
            random.seed(self.seed)
            train_label, val_label = OfficeHome_SingleDomain.split_list(self.labels, 0.9)
            if self.split == 'train':
                self.imgs, self.labels = train_img, train_label
            elif self.split == 'val':
                self.imgs, self.labels = val_img, val_label
                
        self.dataset = MetaDataset(self.imgs, self.labels, self.domain_label, self.transform) # get数据集
    
    @staticmethod
    def split_list(l, ratio):
        assert ratio > 0 and ratio < 1
        random.shuffle(l)
        train_size = int(len(l)*ratio)
        train_l = l[:train_size]
        val_l = l[train_size:]
        return train_l, val_l
        
    @staticmethod
    def read_txt(txt_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            contents = f.readlines()
            
        for line_txt in contents:
            line_txt = line_txt.replace('\n', '')
            line_txt_list = line_txt.split(' ')
            imgs.append(line_txt_list[0])
            labels.append(int(line_txt_list[1]))
            
        return imgs, labels


class OfficeHome_DG(MetaDGDataset):
    def __init__(self, root_path=officehome_path, test_domain='p', batch_size=16, seed=0, val_batch_size=512, transform_train=default_transform_train, transform_test=default_transform_test):
        self.domain_text = list(officehome_name_dict.values())
        self.domain_text.sort()
        
        self.class_text = os.listdir(os.path.join(root_path, self.domain_text[0]))
        self.class_text.sort()
        
        self.domain_text = remove_underline(self.domain_text)
        self.class_text = remove_underline(self.class_text)
        
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.root_path = root_path
        self.domain_list = list(officehome_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        self.seed = seed
        
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.get_single_domain = OfficeHome_SingleDomain
        
        self.datasets_dict = {}
        self.dataloaders_dict = {}
        for domain_name in self.domain_list:
            self.datasets_dict[domain_name] = self.get_singlesite(self.root_path, domain_name)
        
        self.test_dataset = self.datasets_dict[self.test_domain]['test']
        self.get_merged_data(test_domain=self.test_domain)
        self.datasets_dict['merged'] = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset,
        }
    
    def get_singlesite(self, root_path, domain_name, seed=0):
        dataset_dict = {
            'train': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='train', train_transform=self.transform_train, seed=seed).dataset,
            'val': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='val', train_transform=self.transform_test, seed=seed).dataset,
            'test': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='test', train_transform=self.transform_test, seed=seed).dataset,
        }
        return dataset_dict
    

class OfficeHome_Open_SingleDomain(OfficeHome_SingleDomain):
    def __init__(self, root_path=officehome_path, domain_name='p', split='train', train_transform=None, seed=0, train_class=None, test_class=None):
        self.domain_name = domain_name
        assert domain_name in officehome_name_dict.keys(), 'domain_name must be in {}'.format(officehome_name_dict.keys())
        self.root_path = root_path
        self.domain = officehome_name_dict[domain_name]
        domain_list = list(officehome_name_dict.keys())
        domain_list.sort()
        self.domain_label = domain_list.index(domain_name)
        self.txt_path = os.path.join(root_path, '{}_img_label_list.txt'.format(self.domain))
        
        self.split = split
        assert self.split in ['train', 'val', 'test'] , 'split must be train, val or test'
        
        self.train_class_idx_list = train_class
        self.test_class_idx_list = test_class
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = default_transform_test
        self.seed = seed
        
        self.imgs, self.labels = OfficeHome_SingleDomain.read_txt(self.txt_path)
        
        if self.split == 'train' or self.split == 'val':
            random.seed(self.seed)
            train_img, val_img = OfficeHome_SingleDomain.split_list(self.imgs, 0.9)
            random.seed(self.seed)
            train_label, val_label = OfficeHome_SingleDomain.split_list(self.labels, 0.9)
            if self.split == 'train':
                self.imgs, self.labels = train_img, train_label
            elif self.split == 'val':
                self.imgs, self.labels = val_img, val_label
        self.new_class_imgs, self.new_class_labels = self.split_class(self.imgs, self.labels, self.test_class_idx_list)
        self.old_class_imgs, self.old_class_labels = self.split_class(self.imgs, self.labels, self.train_class_idx_list)
        
        self.dataset = MetaDataset(self.imgs, self.labels, self.domain_label, self.transform)
        self.new_class_dataset = MetaDataset(self.new_class_imgs, self.new_class_labels, self.domain_label, self.transform)
        self.old_class_dataset = MetaDataset(self.old_class_imgs, self.old_class_labels, self.domain_label, self.transform)
            
    def split_class(self, imgs, labels, class_list):
        new_imgs = []
        new_labels = []
        for img, label in zip(imgs, labels):
            if label in class_list:
                new_imgs.append(img)
                new_labels.append(label)    
        return new_imgs, new_labels
    
class OfficeHome_OpenDG(MetaDGDataset):
    def __init__(self, root_path=officehome_path, test_domain='p', batch_size=16, seed=0, val_batch_size=512, transform_train=default_transform_train, transform_test=default_transform_test):
        self.domain_text = list(officehome_name_dict.values())
        self.domain_text.sort()
        
        self.class_text = os.listdir(os.path.join(root_path, self.domain_text[0]))
        self.class_text.sort()
        
        self.domain_text = remove_underline(self.domain_text)
        self.class_text = remove_underline(self.class_text)
        
        self.class_idx_list = list(range(len(self.class_text)))
        self.seed = seed
        
        shuffled_class_idx_list = self.class_idx_list.copy() 
        self.train_class_idx_list = shuffled_class_idx_list[:int(len(self.class_idx_list)*0.5)]
        self.test_class_idx_list = shuffled_class_idx_list[int(len(self.class_idx_list)*0.5):]
        
        print('train class idx list: ', self.train_class_idx_list)
        print('test class idx list: ', self.test_class_idx_list)
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.root_path = root_path
        self.domain_list = list(officehome_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        self.seed = seed
        
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.get_single_domain = OfficeHome_Open_SingleDomain
        
        self.datasets_dict = {}
        self.dataloaders_dict = {}
        for domain_name in self.domain_list:
            self.datasets_dict[domain_name] = self.get_singlesite(self.root_path, domain_name)
        
        self.test_dataset = self.datasets_dict[self.test_domain]['test']
        self.get_merged_data(test_domain=self.test_domain)
        self.datasets_dict['merged'] = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'val_new': self.val_new_dataset,
            'val_all': self.val_all_dataset,
            'test': self.test_dataset,
            'test_new': self.datasets_dict[self.test_domain]['test_new'],
            'test_all': self.datasets_dict[self.test_domain]['test_all']
        }
    
    def get_merged_data(self, test_domain=None):
        if test_domain is None:
            test_domain = self.test_domain
        self.train_dataset_list = []
        self.val_dataset_list = []
        self.val_new_dataset_list = []
        self.val_all_dataset_list = []
        for domain in self.train_domain_list:
            self.train_dataset_list.append(self.datasets_dict[domain]['train'])
            self.val_dataset_list.append(self.datasets_dict[domain]['val'])
            self.val_new_dataset_list.append(self.datasets_dict[domain]['val_new'])
            self.val_all_dataset_list.append(self.datasets_dict[domain]['val_all'])
            
        self.train_dataset = torch.utils.data.ConcatDataset(self.train_dataset_list)
        self.val_dataset = torch.utils.data.ConcatDataset(self.val_dataset_list)
        self.val_new_dataset = torch.utils.data.ConcatDataset(self.val_new_dataset_list)
        self.val_all_dataset = torch.utils.data.ConcatDataset(self.val_all_dataset_list)
        
        
    def get_singlesite(self, root_path, domain_name, seed=0):
        dataset_dict = {
            'train': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='train', train_transform=self.transform_train, seed=seed, train_class=self.train_class_idx_list, test_class=self.test_class_idx_list).old_class_dataset,
        }
        
        val_dataset = self.get_single_domain(root_path=root_path, domain_name=domain_name, split='val', train_transform=self.transform_test, seed=seed, train_class=self.train_class_idx_list, test_class=self.test_class_idx_list)
        dataset_dict['val'] = val_dataset.old_class_dataset
        dataset_dict['val_new'] = val_dataset.new_class_dataset
        dataset_dict['val_all'] = val_dataset.dataset
        
        test_dataset = self.get_single_domain(root_path=root_path, domain_name=domain_name, split='test', train_transform=self.transform_test, seed=seed, train_class=self.train_class_idx_list, test_class=self.test_class_idx_list)
        dataset_dict['test'] = test_dataset.old_class_dataset
        dataset_dict['test_new'] = test_dataset.new_class_dataset
        dataset_dict['test_all'] = test_dataset.dataset
        return dataset_dict

def GenFileList():
    total_class_list = None
    for domain_name in officehome_name_dict.keys():
        domain = officehome_name_dict[domain_name]
        domain_path = os.path.join(officehome_path, domain)
        class_list = os.listdir(domain_path)
        class_list.sort()
        if total_class_list is None:
            total_class_list = class_list
        
        assert total_class_list == class_list, 'class_list must be same'
    
    domain_file_dict = {}
    for domain_name in officehome_name_dict.keys():
        domain_file_dict[domain_name] = []
        domain = officehome_name_dict[domain_name]
        domain_path = os.path.join(officehome_path, domain)
        for label_idx, class_name in enumerate(total_class_list):
            class_path = os.path.join(domain_path, class_name)
            file_list = os.listdir(class_path)
            
            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                domain_file_dict[domain_name].append(file_path + ' ' + str(label_idx) + '\n')
        
    return domain_file_dict

def WriteFile(file_path, file_list):
    with open(file_path, 'w') as f:
        for file_name in file_list:
            f.write(file_name)




if __name__ == '__main__':
    pass
