import sys
sys.path.append(sys.path[0].replace('code/data', 'code'))

import os
from data.utils import MetaDataset, MetaDGDataset
from configs.default import vlcs_path, default_transform_train, default_transform_test, BICUBIC


vlcs_name_dict = {
    'v': 'PASCAL',
    'l': 'LABELME',
    'c': 'CALTECH',
    's': 'SUN',
}



split_dict = {
    'train': 'train',
    'val': 'crossval',
    'total': 'test',
}

class_dict = {
    '0': 'bird',
    '1': 'car',
    '2': 'chair',
    '3': 'dog',
    '4': 'person'
}


class VLCS_SingleDomain():
    def __init__(self, root_path=vlcs_path, domain_name='v', split='total', train_transform=None):
        if domain_name in vlcs_name_dict.keys():
            self.domain_name = vlcs_name_dict[domain_name]
            domain_list = list(vlcs_name_dict.keys())
            domain_list.sort()
            self.domain_label = domain_list.index(domain_name)
        else:
            raise ValueError('domain_name should be in v l c s')
        
        self.root_path = os.path.join(root_path,)
        self.split = split
        self.split_file = os.path.join(root_path, f'{self.domain_name}_{split_dict[self.split]}' + '.txt')
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = default_transform_test
                
        imgs, labels = VLCS_SingleDomain.read_txt(self.split_file, self.root_path)
        self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)
        
    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            labels.append(int(line_txt[1]))
        return imgs, labels

class VLCS_DG(MetaDGDataset):
    def __init__(self, root_path=vlcs_path, test_domain='v', batch_size=16, val_batch_size=512, transform_train=default_transform_train, transform_test=default_transform_test):
        self.domain_text = list(vlcs_name_dict.values())
        self.domain_text.sort()
        self.class_text = list(class_dict.values())
        self.class_text.sort()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.domain_list = list(vlcs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        self.root_path = root_path
        
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.get_single_domain = VLCS_SingleDomain
        
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


if __name__ == '__main__':
    pass