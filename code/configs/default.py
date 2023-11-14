from torchvision import transforms
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def remove_underline(class_list):
    for i, class_name in enumerate(class_list):
        if '_' in class_name:
            class_list[i] = class_name.replace('_', ' ')
        class_list[i] = class_list[i].lower()
    return class_list

default_transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),            
            ])

default_transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

clipood_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

clipood_augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


num_classes_dict = {
    'pacs': 7,
    'officehome': 65,
    'terrainc': 10,
    'domainnet': 345,
    'vlcs': 5,
    'imagenet': 1000,
}

imagenet_pretrain_transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

imagenet_pretrain_transform_test = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

dataloader_kwargs = {'num_workers': 8, 'pin_memory': True}
pacs_domain_list = ['p', 'a', 'c', 's']
domainNet_domain_list = ['c', 'i', 'p', 'q', 'r', 's']
officehome_domain_list = ['p', 'a', 'c', 'r']
vlcs_domain_list = ['v', 'l', 'c', 's']
terra_incognita_domain_list = ['100', '38', '43', '46']
terrainc_domain_list = ['100', '38', '43', '46']



log_dir_path = ''


officehome_path = ''
terra_incognita_path = ''
pacs_path = ''
vlcs_path = ''
domainNet_path = ''
ceph_flag = False
domainNet_ceph_path = ''