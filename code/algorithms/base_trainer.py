'''
2023.3.17
基础trainer框架设计
'''

class Base_Trainer(object):
    def __init__(self) -> None:
        '''初始化框架'''
        NotImplementedError

    def initilize(self):
        '''初始化具体执行位置'''
        NotImplementedError
    
    def get_data(self):
        '''获取数据 获得一个我们自定义的MetaDGDataset类 通过GetData获得对应的dataloader和dataset'''
        NotImplementedError
    
    def get_model(self):
        '''获取预训练CLIP model 内部含有针对data中获得的domain_text class_text的prompt处理'''
        NotImplementedError
        
    def get_optimizer(self):
        '''获取具体的优化器部分'''
        NotImplementedError
        
    def get_prompt(self):
        '''获取具体的prompt部分'''
        NotImplementedError
    
    def get_logger(self):
        NotImplementedError
    
    
    def save_checkpoint(self):
        '''保存模型'''
        NotImplementedError
    
    def train(self):
        '''具体的训练部分 实际是本项目主要聚焦于大模型的finetune'''
        NotImplementedError
    
    def val(self):
        '''具体的验证部分'''
        NotImplementedError
    
    def run(self):
        '''整个流程控制'''
        NotImplementedError
    

