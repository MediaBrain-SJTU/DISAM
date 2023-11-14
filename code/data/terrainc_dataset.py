import sys
sys.path.append(sys.path[0].replace('code/data', 'code'))
from configs.default import terra_incognita_path, remove_underline, default_transform_train, default_transform_test
import os
from data.utils import MetaDataset, MetaDGDataset
import random
terra_incognita_name_dict = {
    '100': 'location_100',
    '38': 'location_38',
    '43': 'location_43',
    '46': 'location_46',
}


class TerraInc_SingleDomain():
    def __init__(self, root_path=terra_incognita_path, domain_name='100', split='train', train_transform=None, seed=0):
        self.domain_name = domain_name
        assert domain_name in terra_incognita_name_dict.keys(), 'domain_name must be in {}'.format(terra_incognita_name_dict.keys())
        self.root_path = root_path
        self.domain = terra_incognita_name_dict[domain_name]
        self.domain_label = list(terra_incognita_name_dict.keys()).index(domain_name)
        self.txt_path = os.path.join(root_path, '{}_img_label_list.txt'.format(self.domain))
        
        self.split = split
        assert self.split in ['train', 'val', 'test'] , 'split must be train, val or test'
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = default_transform_test
        self.seed = seed
        
        self.imgs, self.labels = TerraInc_SingleDomain.read_txt(self.txt_path)
        
        if self.split == 'train' or self.split == 'val':
            random.seed(self.seed)
            train_img, val_img = TerraInc_SingleDomain.split_list(self.imgs, 0.8)
            random.seed(self.seed)
            train_label, val_label = TerraInc_SingleDomain.split_list(self.labels, 0.8)
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
    

class TerraInc_DG(MetaDGDataset):
    def __init__(self, root_path=terra_incognita_path, test_domain='100', batch_size=64, seed=0, val_batch_size=512, transform_train=default_transform_train, transform_test=default_transform_test):
        self.domain_text = list(terra_incognita_name_dict.values())
        self.domain_text.sort()
        
        self.class_text = os.listdir(os.path.join(root_path, self.domain_text[0]))
        self.class_text.sort()
        
        self.domain_text = remove_underline(self.domain_text)
        self.class_text = remove_underline(self.class_text)
        # print(self.class_text)
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.root_path = root_path
        self.domain_list = list(terra_incognita_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        self.seed = seed
        
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.get_single_domain = TerraInc_SingleDomain
        
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
        

def GenFileList():
    total_class_list = None
    for domain_name in terra_incognita_name_dict.keys():
        domain = terra_incognita_name_dict[domain_name]
        domain_path = os.path.join(terra_incognita_path, domain)
        class_list = os.listdir(domain_path)
        class_list.sort()
        if total_class_list is None:
            total_class_list = class_list
        
        assert total_class_list == class_list, 'class_list must be same'
    
    domain_file_dict = {}
    for domain_name in terra_incognita_name_dict.keys():
        domain_file_dict[domain_name] = []
        domain = terra_incognita_name_dict[domain_name]
        domain_path = os.path.join(terra_incognita_path, domain)
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


def count_class_num(dataset_dict_in):
    domain_count_per_class = {}
    for domain_name in dataset_dict_in.keys():
        dataset = dataset_dict_in[domain_name]['test']
        labels = dataset.labels
        domain_count_per_class[domain_name] = {}
        for i in range(10):
            domain_count_per_class[domain_name][i] = labels.count(i)
    return domain_count_per_class

if __name__ == '__main__':
    pass
    

