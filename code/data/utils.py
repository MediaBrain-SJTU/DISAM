import torch
from torch.utils.data import Dataset
from PIL import Image
from configs.default import dataloader_kwargs, ceph_flag, bucket_name
import numpy as np
import cv2
from torchvision import transforms
if ceph_flag:
    from petrel_client.utils.data import DataLoader
    from configs.default import ceph_client
    
    
def GetDataLoaderDict(dataset_dict, batch_size, dataloader_kwargs=dataloader_kwargs, val_batch_size=512):
    dataloader_dict = {}
    for dataset_name in dataset_dict.keys():
        if 'train' in dataset_name:
            if ceph_flag:
                dataloader_dict[dataset_name] = DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=True, drop_last=True, prefetch_factor=4, persistent_workers=True,  **dataloader_kwargs)
            else:
                dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=True, drop_last=True, **dataloader_kwargs)
        
        else:
            if ceph_flag:
                dataloader_dict[dataset_name] = DataLoader(dataset_dict[dataset_name], batch_size=batch_size, shuffle=False, drop_last=False, prefetch_factor=4, persistent_workers=True,  **dataloader_kwargs)
            else: # batch size调大
                dataloader_dict[dataset_name] = torch.utils.data.DataLoader(dataset_dict[dataset_name], batch_size=val_batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)
            

    return dataloader_dict

def ceph_img_read(img_path, bucket_name, client):
    file_url = 's3://' + bucket_name + img_path
    img_bytes = client.get(file_url)
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = Image.fromarray(img)
    return img

class MetaDataset(Dataset):
    def __init__(self, imgs, labels, domain_label, transform=None, ceph_flag=ceph_flag):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
        self.transform = transform
        self.ceph_flag = ceph_flag
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_class_label = self.labels[index]
        if self.ceph_flag:
            img = ceph_img_read(img_path, bucket_name, ceph_client)
        else:
            img = Image.open(img_path).convert("RGB")

            
        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)
            
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_class_label, self.domain_label

    def __len__(self):
        return len(self.imgs)
    

class MetaDGDataset():
    def __init__(self, root_path=None, test_domain=None, batch_size=None):
        self.batch_size = batch_size
        self.test_domain = test_domain
        self.root_path = root_path
        
        self.val_batch_size = 512
        
        self.domain_list = None
        self.domain_text = None
        self.class_text = None
        
        self.get_single_domain = None
        
        self.transform_train = None
        self.transform_test = None
        
    def get_merged_data(self, test_domain=None):
        if test_domain is None:
            test_domain = self.test_domain
        self.train_dataset_list = []
        self.val_dataset_list = []
        for domain in self.train_domain_list:
            self.train_dataset_list.append(self.datasets_dict[domain]['train'])
            self.val_dataset_list.append(self.datasets_dict[domain]['val'])
            
        self.train_dataset = torch.utils.data.ConcatDataset(self.train_dataset_list)
        self.val_dataset = torch.utils.data.ConcatDataset(self.val_dataset_list)
    
    def get_singlesite(self, root_path, domain_name):
        dataset_dict = {
            'train': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='train', train_transform=self.transform_train).dataset,
            'val': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='val', train_transform=self.transform_test).dataset,
            'test': self.get_single_domain(root_path=root_path, domain_name=domain_name, split='total', train_transform=self.transform_test).dataset,
        }
        return dataset_dict
        
    def get_data(self, batch_size=None, val_batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if val_batch_size is None:
            val_batch_size = self.val_batch_size
        for domain_name in self.datasets_dict.keys():
            self.dataloaders_dict[domain_name] = GetDataLoaderDict(self.datasets_dict[domain_name], batch_size, val_batch_size=val_batch_size)
        return self.dataloaders_dict, self.datasets_dict
        

class MetaDataset_MultiDomain(Dataset):
    def __init__(self, imgs, labels, domain_labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_labels
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_class_label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_class_label, self.domain_labels[index]

    def __len__(self):
        return len(self.imgs)
    
    
class PathMetaDataset(Dataset):
    def __init__(self, imgs, labels, domain_label, transform=None, show_path=False):
        self.imgs = imgs
        self.labels = labels
        self.domain_label = domain_label
        self.transform = transform
        self.show_path = show_path
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_class_label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        if len(img.split()) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)
            
        if self.transform is not None:
            img = self.transform(img)
        if self.show_path:
            return img, img_path, img_class_label, self.domain_label
        else:
            return img, img_path, img_class_label, self.domain_label
        
    def __len__(self):
        return len(self.imgs)
    
    