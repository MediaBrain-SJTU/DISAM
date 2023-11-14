import sys
sys.path.append(sys.path[0].replace('algorithms', ''))

from algorithms.ERM import SAM_Trainer
from tqdm import tqdm

class DISAM_Trainer(SAM_Trainer):
    def get_log_name(self):
        return super().get_log_name() + f'_lambda{self.args.lambda_weight}'
    
    @staticmethod
    def compute_variance_penalty(domain_loss_list):
        mu = sum(domain_loss_list) / len(domain_loss_list)
        mean_var = 0.
        for domain_loss in domain_loss_list:
            mean_var += (domain_loss - mu)**2
        mean_var /= len(domain_loss_list)
        return mean_var
    
    @staticmethod
    def get_domain_loss(preds, labels, domain_labels, criterion):
        domain_list = list(set(domain_labels))
        domain_list.sort()
        domain_loss_list = []
        for domain_name in domain_list:
            domain_mask = domain_labels == domain_name
            labels_per_domain = labels[domain_mask]
            preds_pre_domain = preds[domain_mask]
            domain_loss_list.append(criterion(preds_pre_domain, labels_per_domain))
        return domain_loss_list
    
    
    def sam_train(self, n_epoch, dataloader, model=None, optimizer=None, prefix='merged'):
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
            
            if self.args.optim in ['sagm', 'gsam', 'sagm_adamw', 'gsam_adamw']:
                optimizer.set_closure(self.verx_ce_loss, imgs, labels)
                output, loss = optimizer.step()
                optimizer.update_rho_t()
            else:
                output = model(imgs)
                domain_loss_list = DISAM_Trainer.get_domain_loss(output, labels, domain_labels, self.criterion)
                loss_ce = self.criterion(output, labels)
                loss_verx = DISAM_Trainer.compute_variance_penalty(domain_loss_list)
                loss = loss_ce - self.args.lambda_weight * loss_verx
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
        
        