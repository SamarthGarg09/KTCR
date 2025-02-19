import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning as L
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *
import numpy as np
import random
from collections import Counter

class CAVDataModule(L.LightningDataModule):
    
    def __init__(self, config):
        
        super().__init__()
        self.batch_size = config['cav']['data_module']['batch_size']
        self.dataset_path = config['cav']['data_module']['dataset_path']
        self.save_dataset_path = config['cav']['data_module']['save_dataset_path']
        self.dataset_mapping_size = config['aec']['data_module']['dataset_mapping_size']
        self.test_size = config['aec']['data_module']['test_size']
        self.tokenizer = AutoTokenizer.from_pretrained(config['aec']['lightning_module']['teacher_model'])
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.num_workers = config['cav']['data_module']['num_workers']
        self.seed = config['aec']['data_module']['seed']

    # def prepare_data(self):
    #     df = pd.read_csv(self.dataset_path)

    #     dataset = Dataset.from_pandas(df)

    #     def tokenize(batch):
    #         return self.tokenizer(batch['Text'], truncation=True, padding=True)

    #     dataset = dataset.map(tokenize, batched=True, batch_size=self.dataset_mapping_size)
    #     dataset = dataset.train_test_split(test_size=self.test_size, shuffle=True, seed=self.seed)

    #     self.dataset = dataset.copy()
    #     self.dataset['validation'] = self.dataset.pop('test')
    #     self.dataset['test'] = dataset['test']

    #     for k in self.dataset.keys():
    #         self.dataset[k].set_format('torch', columns=['input_ids', 'attention_mask', 'Labels'])

    #     # save the dataset
    #     self.dataset = DatasetDict(self.dataset)
    #     self.dataset.save_to_disk(self.save_dataset_path)
    def balance_dataset(self, dataset):

        labels = [int(label) for label in dataset['Labels']]
        
        label_counts = Counter(labels)

        min_count = min(label_counts.values())
        
        balanced_indices = {label: [] for label in label_counts.keys()}
        
        for idx, label in enumerate(labels):
            balanced_indices[label].append(idx)

        balanced_indices = {label: random.sample(indices, min_count) 
                            for label, indices in balanced_indices.items()}
        
        balanced_indices = [idx for indices in balanced_indices.values() for idx in indices]
        balanced_indices.sort()
        
        balanced_dataset = dataset.select(balanced_indices)
        
        return balanced_dataset

    def setup(self, stage=None):
        dataset = load_from_disk(self.save_dataset_path)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = self.balance_dataset(dataset['train'])
            self.val_dataset = dataset['validation']
        
        if stage == 'test' or stage is None:            
            self.test_dataset = dataset['test']   
    # def setup(self, stage=None):
        
    #     dataset = load_from_disk(self.save_dataset_path)
    #     if stage == 'fit' or stage is None:
    #         self.train_dataset = dataset['train']
    #         self.val_dataset = dataset['validation']
    #     if stage == 'test' or stage is None:            
    #         self.test_dataset = dataset['test']

    def train_dataloader(self):
        
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator, num_workers=self.num_workers)

    def val_dataloader(self):
        
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers)

    def test_dataloader(self):
        
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers)

class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim_teacher, bottleneck_dim, input_dim_student=768):
        
        super().__init__()
        self.encoder = nn.Linear(input_dim_teacher, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, input_dim_teacher)
        self.reduction = nn.Linear(input_dim_student, bottleneck_dim)

    def forward(self, x, student_input):
        
        bottleneck = F.relu(self.encoder(x))
        reconstructed = F.relu(self.decoder(bottleneck))
        
        return bottleneck, reconstructed, None

class CAVModule(pl.LightningModule):
    
    def __init__(self, teacher_model, autoencoder, config, **kwargs):
        
        super().__init__()
        self.learning_rate = config['cav']['lightning_module']['learning_rate']
        self.teacher_model = teacher_model
        # self.autoencoder = AutoEncoder(self.teacher_model.config.hidden_size, bottleneck_dim=256)
        self.autoencoder = autoencoder
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.output_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.num_training_steps = kwargs['cav_data_module_len'] * config['cav']['lightning_module']['max_epochs'] 
        self.cav = None
        # self.train_outputs = []
        # self.test_outputs = []
        # self.val_outputs = []
        self.train_preds = []
        self.train_labels = []
        self.train_losses = []
        self.val_losses = []
        self.val_preds = []
        self.val_labels = []
        
        self.log_file_path = 'training_metrics/cav_training.pkl'
        self.curves_file_path = 'training_curve/cav_training.png'
        initialize_loss_log_cav(self.log_file_path)
        
    def forward(self, x):
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
            ).last_hidden_state[:, 0, :]
        
        # Ensure the tensors require gradient after no_grad block
        teacher_outputs = teacher_outputs.detach().requires_grad_(True)
        
        bottleneck, _, _ = self.autoencoder(teacher_outputs, teacher_outputs)
        x = self.output_model(bottleneck)
        
        return x
        
    def get_cav(self):
        
        self.save_cav()

        return self.cav.detach().cpu().numpy()
    
    def training_step(self, batch, batch_idx):
        
        x, y = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
        }, batch['Labels']
        y_hat = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.float().unsqueeze(1))
        accuracy = (y_hat.round() == y.float().unsqueeze(1)).float().mean()
        # self.train_outputs.append({'loss': loss, 'accuracy': accuracy})
        preds = torch.argmax(y_hat, dim=-1)
        
        self.train_preds.append(preds.detach().cpu().numpy())
        self.train_labels.append(y.detach().cpu().numpy())
        self.train_losses.append(loss.detach().cpu().numpy())
        # return {'loss': loss, 'accuracy': accuracy}
        return loss
    
    def on_train_epoch_end(self):
        
        avg_loss_epoch = np.mean(self.train_losses)
        all_preds = np.concatenate(self.train_preds)
        all_labels = np.concatenate(self.train_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        loss_log = load_loss_log(self.log_file_path)
        loss_log['train_losses'].append(avg_loss_epoch)
        loss_log['train_f1'].append(f1)
        save_loss_log(self.log_file_path, loss_log)
        
        self.train_preds = []
        self.train_labels = []
        self.train_losses = []

        torch.cuda.empty_cache()
        
        self.log('train_loss_epoch', avg_loss_epoch, prog_bar=True, logger=True)
        self.log('train_f1_epoch', f1, prog_bar=True, logger=True)
        # avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean()
        # avg_accuracy = torch.stack([x['accuracy'] for x in self.train_outputs]).mean()
        

        # loss_log = load_loss_log(self.log_file_path)
        # loss_log['train_losses'].append(avg_loss.detach().cpu().numpy())
        # # loss_log['train_accuracy'].append(avg_accuracy)
        # save_loss_log(self.log_file_path, loss_log)
        
        # self.train_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        
        x, y = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
        }, batch['Labels']
        y_hat = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.float().unsqueeze(1))
        accuracy = (y_hat.round() == y.float().unsqueeze(1)).float().mean()
        preds = torch.argmax(y_hat, dim=-1)
        self.val_preds.append(preds.detach().cpu().numpy())
        self.val_labels.append(y.detach().cpu().numpy())
        self.val_losses.append(loss.detach().cpu().numpy())
        
        # self.val_outputs.append({'loss': loss, 'accuracy': accuracy})
        # self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        # self.log('val_accuracy', accuracy, on_step=True, prog_bar=True, logger=True)
        
        # return {'loss': loss, 'accuracy': accuracy}

        return loss
    
    def on_validation_epoch_end(self):

        avg_loss_epoch = np.mean(self.val_losses)
        all_preds = np.concatenate(self.val_preds)
        all_labels = np.concatenate(self.val_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        loss_log = load_loss_log(self.log_file_path)
        loss_log['val_losses'].append(avg_loss_epoch)
        loss_log['val_f1'].append(f1)
        save_loss_log(self.log_file_path, loss_log)

        self.log('val_loss_epoch', avg_loss_epoch, prog_bar=True, logger=True)
        self.log('val_f1_epoch', f1, prog_bar=True, logger=True)
        
        self.val_preds = []
        self.val_labels = []
        self.val_losses = []

        torch.cuda.empty_cache()
        
        # avg_loss = torch.stack([x['loss'] for x in self.val_outputs]).mean()
        # avg_accuracy = torch.stack([x['accuracy'] for x in self.val_outputs]).mean()
        
        # self.log('val_loss_epoch', avg_loss, prog_bar=True, logger=True)
        # self.log('val_accuracy_epoch', avg_accuracy, prog_bar=True, logger=True)

        # loss_log = load_loss_log(self.log_file_path)
        # loss_log['val_losses'].append(avg_loss.detach().cpu().numpy())
        # # loss_log['val_accuracy'].append(avg_accuracy)
        # save_loss_log(self.log_file_path, loss_log)
        
        # self.val_outputs.clear()
        

    def test_step(self, batch, batch_idx):
        x, y = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
        }, batch['Labels']
        y_hat = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.float().unsqueeze(1))
        accuracy = (y_hat.round() == y.float().unsqueeze(1)).float().mean()
        f1 = f1_score(y_hat.round().cpu(), y.float().unsqueeze(1).cpu(), average='macro')
        
        self.test_outputs.append({'loss': loss, 'accuracy': accuracy, 'f1': f1, 'y_true': y.cpu(), 'y_pred': y_hat.round().cpu()})
        
        return {'loss': loss, 'accuracy': accuracy, 'f1': f1}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in self.test_outputs]).mean()
        avg_f1 = torch.tensor([x['f1'] for x in self.test_outputs]).mean()
        self.log('test_loss_epoch', avg_loss, prog_bar=True, logger=True)
        self.log('test_accuracy_epoch', avg_accuracy, prog_bar=True, logger=True)
        self.log('test_f1_epoch', avg_f1, prog_bar=True, logger=True)
        
        # Compute confusion matrix
        y_true = torch.cat([x['y_true'] for x in self.test_outputs])
        y_pred = torch.cat([x['y_pred'] for x in self.test_outputs])
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        df_cm = pd.DataFrame(cm, index=[i for i in range(cm.shape[0])], columns=[i for i in range(cm.shape[1])])
        plt.figure(figsize=(10,7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()

        self.test_outputs.clear()


    def plot_losses(self):
        loss_log = load_loss_log(self.log_file_path)
        
        # Plot for losses
        plt.figure(figsize=(10, 6))
        
        if loss_log['train_losses']:
            train_losses = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['train_losses']]
            plt.plot(train_losses, label='Training Loss')
            
        if loss_log['val_losses']:
            val_losses = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['val_losses']]
            plt.plot(val_losses, label='Validation Loss')
            
        if loss_log['test_losses']:
            test_losses = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['test_losses']]
            plt.plot(test_losses, label='Test Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.savefig(self.curves_file_path.replace('.png', '_loss.png'))
        plt.show()
    
        # Plot for F1 scores
        plt.figure(figsize=(10, 6))
        
        if loss_log['train_f1']:
            train_f1 = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['train_f1']]
            plt.plot(train_f1, label='Training F1-score')
            
        if loss_log['val_f1']:
            val_f1 = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['val_f1']]
            plt.plot(val_f1, label='Validation F1-score')
        
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Curves')
        plt.legend()
        plt.savefig(self.curves_file_path.replace('.png', '_f1.png'))
        plt.show()

        
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            betas=(0.9, 0.999), 
            weight_decay=0.01
        )
        total_steps = self.num_training_steps
        warmup_steps = int(0.1 * total_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss_epoch'
            }
        }

    def save_cav(self):
        
        self.cav = self.output_model[-1].weight.data[0]
        # shape-->tensor(64,)
        torch.save(self.cav, './models/cav.pth')