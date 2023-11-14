class Base_Trainer(object):
    def __init__(self) -> None:
        NotImplementedError

    def initilize(self):
        NotImplementedError
    
    def get_data(self):
        NotImplementedError
    
    def get_model(self):
        NotImplementedError
        
    def get_optimizer(self):
        NotImplementedError
        
    def get_prompt(self):
        NotImplementedError
    
    def get_logger(self):
        NotImplementedError
    
    
    def save_checkpoint(self):
        NotImplementedError
    
    def train(self):
        NotImplementedError
    
    def val(self):
        NotImplementedError
    
    def run(self):
        NotImplementedError
    

