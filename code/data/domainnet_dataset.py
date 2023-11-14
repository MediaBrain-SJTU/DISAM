import sys
sys.path.append(sys.path[0].replace('code/data', 'code'))
import os
import torch
from configs.default import domainNet_path, ceph_flag, domainNet_ceph_path, remove_underline, default_transform_train, default_transform_test
from data.utils import MetaDataset, MetaDGDataset

domainNet_name_dict = {
    'c': 'clipart',
    'i': 'infograph',
    'p': 'painting',
    'q': 'quickdraw',
    'r': 'real',
    's': 'sketch',
}
def get_domainnet_class_text(root_path=domainNet_path):
    file_path = os.path.join(root_path, 'clipart_train.txt')
    with open(file_path, 'r') as f:
        content = f.readlines()
    class_name_dict = {}
    for line in content:
        file_path, label = line.split(' ')
        label = int(label.replace('\n', ''))
        class_name_dict[label] = file_path.split('/')[1].replace('_', ' ')
        
    class_name_list = []
    for i in range(len(class_name_dict.keys())):
        class_name_list.append(class_name_dict[i])
    return class_name_list

class DomainNet_SingleDomain():
    def __init__(self, root_path=domainNet_path, domain_name='c', split='train', train_transform=None):
        self.root_path = root_path
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = default_transform_test
        assert domain_name in domainNet_name_dict.keys(), 'domain_name must be in {}'.format(domainNet_name_dict.keys())    
        self.domain_name = domain_name
        self.domain = domainNet_name_dict[domain_name]
        self.domain_label = list(domainNet_name_dict.keys()).index(domain_name)
        split_map_table = {'train':'train', 'val':'test', 'total':'test'}
        assert split_map_table[split] in ['train', 'test', 'total'] , 'split must be train, test or total'
        self.split = split_map_table[split]
        
        imgs, labels = DomainNet_SingleDomain.ReadSplitFile(self.root_path, self.domain, self.split)
        self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)
        
    @staticmethod
    def ReadSplitFile(root_path, domain_name, split):
        if split == 'train' or split == 'test':
            txt_file = os.path.join(root_path, domain_name + f'_{split}.txt')
            with open(txt_file, 'r') as f:
                content = f.readlines()
        elif split == 'total':
            txt_file = os.path.join(root_path, domain_name + '_train.txt')
            with open(txt_file, 'r') as f:
                train_content = f.readlines()
            txt_file = os.path.join(root_path, domain_name + '_test.txt')
            with open(txt_file, 'r') as f:
                test_content = f.readlines()
                
            content = train_content + test_content
        
        imgs = []
        labels = []
        for line_text in content:
            file_path, label = line_text.split(' ')
            label = int(label.replace('\n', ''))
            if ceph_flag:
                imgs.append(os.path.join(domainNet_ceph_path, file_path))
            else:    
                imgs.append(os.path.join(root_path, file_path))
            labels.append(label)
        
        return imgs, labels        
        

def resort(class_list):
    for i, class_name in enumerate(class_list):
        if '_' in class_name:
            class_list[i] = class_name.replace('_', ' ')
    name_dict = {}
    for class_name in class_list:
        name_dict[class_name] = class_name.lower()
    class_list.sort(key=lambda x: name_dict[x])
    return class_list

class DomainNet_DG(MetaDGDataset):
    def __init__(self, root_path=domainNet_path, test_domain='c', batch_size=64, val_batch_size=512, transform_train=default_transform_train, transform_test=default_transform_test) -> None:
        self.domain_text = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        self.domain_text = remove_underline(self.domain_text)
        self.class_text = get_domainnet_class_text()
        
        self.root_path = root_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.domain_list = list(domainNet_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain) 
        
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.get_single_domain = DomainNet_SingleDomain
        
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
    

    
class DomainNet_Open_SingleDomain(DomainNet_SingleDomain):
    def __init__(self, root_path=domainNet_path, domain_name='p', split='train', train_transform=None, train_class=None, test_class=None):
        self.domain_name = domain_name
        assert domain_name in domainNet_name_dict.keys(), 'domain_name must be in {}'.format(domainNet_name_dict.keys())
        self.root_path = root_path
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = default_transform_test
        
        self.domain = domainNet_name_dict[domain_name]
        domain_list = list(domainNet_name_dict.keys())
        domain_list.sort()
        self.domain_label = list(domainNet_name_dict.keys()).index(domain_name)
        split_map_table = {'train':'train', 'val':'test', 'total':'test'} # 只有test
        assert split_map_table[split] in ['train', 'test', 'total'] , 'split must be train, test or total'
        self.split = split_map_table[split]
        
        self.train_class_idx_list = train_class
        self.test_class_idx_list = test_class
        
        self.imgs, self.labels = DomainNet_SingleDomain.ReadSplitFile(self.root_path, self.domain, self.split)
        self.new_class_imgs, self.new_class_labels = self.split_class(self.imgs, self.labels, self.test_class_idx_list)
        self.old_class_imgs, self.old_class_labels = self.split_class(self.imgs, self.labels, self.train_class_idx_list)
        
        self.dataset = MetaDataset(self.imgs, self.labels, self.domain_label, self.transform) # get数据
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
    
class DomainNet_OpenDG(MetaDGDataset):
    def __init__(self, root_path=domainNet_path, test_domain='p', batch_size=16, seed=0, val_batch_size=512, transform_train=default_transform_train, transform_test=default_transform_test):
        self.domain_text = list(domainNet_name_dict.values())
        self.domain_text.sort()
        
        self.class_text = os.listdir(os.path.join(root_path, self.domain_text[0]))
        self.class_text.sort()
        
        self.domain_text = remove_underline(self.domain_text)
        self.class_text = get_domainnet_class_text()
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
        self.domain_list = list(domainNet_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        self.seed = seed
        
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.get_single_domain = DomainNet_Open_SingleDomain
        
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
            'train': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='train', train_transform=self.transform_train, train_class=self.train_class_idx_list, test_class=self.test_class_idx_list).old_class_dataset,
        }
        
        val_dataset = self.get_single_domain(root_path=root_path, domain_name=domain_name, split='val', train_transform=self.transform_test, train_class=self.train_class_idx_list, test_class=self.test_class_idx_list)
        dataset_dict['val'] = val_dataset.old_class_dataset
        dataset_dict['val_new'] = val_dataset.new_class_dataset
        dataset_dict['val_all'] = val_dataset.dataset
        
        test_dataset = self.get_single_domain(root_path=root_path, domain_name=domain_name, split='total', train_transform=self.transform_test, train_class=self.train_class_idx_list, test_class=self.test_class_idx_list)
        dataset_dict['test'] = test_dataset.old_class_dataset
        dataset_dict['test_new'] = test_dataset.new_class_dataset
        dataset_dict['test_all'] = test_dataset.dataset
        return dataset_dict


if __name__ == '__main__':
    pass       
                

        
        