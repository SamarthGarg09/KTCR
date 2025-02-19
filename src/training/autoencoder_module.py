import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as L
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
from pytorch_lightning.callbacks import LearningRateFinder
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from collections import defaultdict
import pandas as pd
import yaml
import os
import warnings
from transformers import logging as transformers_logging
from utils import *
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

class AECDataModule(L.LightningDataModule):
    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()
        self.teacher_ckpt = config['aec']['lightning_module']['teacher_model_checkpoint']
        self.student_ckpt = config['stmod']['lightning_module']['student_checkpoint_path']
        self.batch_size = config['aec']['data_module']['batch_size']
        self.dataset_path = config['aec']['data_module']['dataset_path']
        self.tokenized_max_length_teacher = config['aec']['data_module']['tokenized_max_length_teacher']
        self.tokenized_max_length_student = config['aec']['data_module']['tokenized_max_length_student']
        self.dataset_mapping_size = config['aec']['data_module']['dataset_mapping_size']
        self.num_workers = config['aec']['data_module']['num_workers']
        self.test_size = config['aec']['data_module']['test_size']
        self.seed = config['aec']['data_module']['seed']
        self.save_dataset_path = config['aec']['data_module']['save_dataset_path']

    # def prepare_data(self):
    #     df = pd.read_csv(self.dataset_path)

    #     dataset = Dataset.from_pandas(df)
    #     self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_ckpt)
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.student_ckpt)
        

    #     def tokenize(example):
    #         teacher_inputs = self.teacher_tokenizer(example['Text'], truncation=True, padding='max_length', max_length=self.tokenized_max_length_teacher)
    #         student_inputs = self.tokenizer(example['Text'], truncation=True, padding='max_length', max_length=self.tokenized_max_length_student)
    #         return {
    #             'teacher_input_ids': teacher_inputs['input_ids'],
    #             'teacher_attention_mask': teacher_inputs['attention_mask'],
    #             'student_input_ids': student_inputs['input_ids'],
    #             'student_attention_mask': student_inputs['attention_mask']
    #         }
            
    #     dataset = dataset.map(tokenize, batched=True, batch_size=self.dataset_mapping_size)
    #     dataset = dataset.train_test_split(test_size=self.test_size, shuffle=True, seed=self.seed)

    #     self.dataset = dataset.copy()
    #     self.dataset['validation'] = self.dataset.pop('test')
    #     self.dataset['test'] = dataset['test']

    #     for k in self.dataset.keys():
    #         self.dataset[k].set_format('torch', columns=['Labels', 'teacher_input_ids', 'teacher_attention_mask', 'student_input_ids', 'student_attention_mask'])

    #     # save the dataset
    #     self.dataset = DatasetDict(self.dataset)
    #     self.dataset.save_to_disk(self.save_dataset_path)

    def setup(self, stage=None):

        dataset = load_from_disk(self.save_dataset_path)
        if stage == 'fit' or stage is None:
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
        if stage == 'test' or stage is None:
            self.test_dataset = dataset['test']

    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim_teacher, bottleneck_dim, input_dim_student=768):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim_teacher, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, input_dim_teacher)
        self.reduction = nn.Linear(input_dim_student, bottleneck_dim)

    def forward(self, x, student_input):
        
        bottleneck = F.relu(self.encoder(x))
        reconstructed = F.relu(self.decoder(bottleneck))
        
        if student_input.size(-1)!=1024:
            student_reduced_output = F.relu(self.reduction(student_input))

            return bottleneck, reconstructed, student_reduced_output
       
        return bottleneck, reconstructed, None


class FinalModel(nn.Module):

    def __init__(self, student_model, teacher_model, autoencoder_model):
        super().__init__()
        
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.autoencoder = autoencoder_model

        for param in self.student_model.parameters():
            param.requires_grad = False
            
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch['teacher_input_ids'],
                attention_mask=batch['teacher_attention_mask']
            ).last_hidden_state[:, 0, :]

            student_outputs = self.student_model(
                input_ids=batch['student_input_ids'],
                attention_mask=batch['student_attention_mask']
            ).last_hidden_state[:, 0, :]

        bottleneck, reconstructed, student_reduced_outputs = self.autoencoder(teacher_outputs, student_outputs)

        return bottleneck, student_reduced_outputs, reconstructed, teacher_outputs


class AecModule(L.LightningModule):

    def __init__(self, student_model, teacher_model, autoencoder_model, config, **kwargs):
        super().__init__()
        
        self.final_model = FinalModel(student_model, teacher_model, autoencoder_model)
        self.reconstruction_criterion = nn.MSELoss()
        self.distillation_criterion = nn.MSELoss()
        self.learning_rate = config['aec']['lightning_module']['learning_rate']
        self.num_training_steps = kwargs['aec_data_module_len'] * config['aec']['lightning_module']['max_epochs']
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.log_file_path = 'training_metrics/aec_training.pkl'
        self.curves_file_path = 'training_curve/aec_training.png'
        initialize_loss_log_aec(self.log_file_path)

    def forward(self, batch):
        batch = {k:v.to('cuda') for k, v in batch.items()}
        return self.final_model(batch)

    def training_step(self, batch, batch_idx):
        
        bottleneck, student_reduced_outputs, reconstructed, teacher_outputs = self.final_model(batch)
        bottleneck.requires_grad=True
        student_reduced_outputs.requires_grad=True
        reconstructed.requires_grad=True
        teacher_outputs.requires_grad=True
        reconstruction_loss = self.reconstruction_criterion(reconstructed, teacher_outputs)
        distillation_loss = self.distillation_criterion(student_reduced_outputs, bottleneck)
        loss = reconstruction_loss + distillation_loss
        # self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss)
        
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True, logger=True)
        loss_log = load_loss_log(self.log_file_path)
        loss_log['train_losses'].append(avg_loss.detach().cpu().numpy())
        save_loss_log(self.log_file_path, loss_log)
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        
        bottleneck, student_reduced_outputs, reconstructed, teacher_outputs = self.final_model(batch)
        bottleneck.requires_grad=True
        student_reduced_outputs.requires_grad=True
        reconstructed.requires_grad=True
        teacher_outputs.requires_grad=True
        reconstruction_loss = self.reconstruction_criterion(reconstructed, teacher_outputs)
        distillation_loss = self.distillation_criterion(student_reduced_outputs, bottleneck)
        loss = reconstruction_loss + distillation_loss
        # self.log("val_loss", loss, prog_bar=True)
        self.val_losses.append(loss)
        
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('val_loss_epoch', avg_loss, prog_bar=True, logger=True)

        loss_log = load_loss_log(self.log_file_path)
        loss_log['val_losses'].append(avg_loss.detach().cpu().numpy())
        save_loss_log(self.log_file_path, loss_log)
        
        self.val_losses.clear()
        
    def test_step(self, batch, batch_idx):
        
        bottleneck, student_reduced_outputs, reconstructed, teacher_outputs = self.final_model(batch)
        bottleneck.requires_grad=True
        student_reduced_outputs.requires_grad=True
        reconstructed.requires_grad=True
        teacher_outputs.requires_grad=True
        reconstruction_loss = self.reconstruction_criterion(reconstructed, teacher_outputs)
        distillation_loss = self.distillation_criterion(student_reduced_outputs, bottleneck)
        loss = reconstruction_loss + distillation_loss
        # self.log("test_loss", loss, prog_bar=True)
        self.test_losses.append(loss)
        
        return loss

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log('test_loss_epoch', avg_loss, prog_bar=True, logger=True)
        self.test_losses.clear()

    def predict_step(self, batch, batch_idx):
        
        bottleneck, student_reduced_outputs, reconstructed, teacher_outputs = self.final_model(batch)
        
        return bottleneck, student_reduced_outputs, reconstructed, teacher_outputs

    def plot_losses(self):
        loss_log = load_loss_log(self.log_file_path)
        
        plt.figure(figsize=(10, 6))
        
        if loss_log['train_losses']:
            plt.plot(loss_log['train_losses'], label='Training Loss')
        if loss_log['val_losses']:
            plt.plot(loss_log['val_losses'], label='Validation Loss')
        if loss_log['test_losses']:
            plt.plot(loss_log['test_losses'], label='Test Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.savefig(self.curves_file_path)
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
                'monitor': 'val_loss'
            }
        }


# if __name__=='__main__':

#     with open("./config.yaml", "r") as f:
#         config = yaml.safe_load(f)
    
#     data_module = AECDataModule(config)
#     data_module.prepare_data()
#     data_module.setup()
    
#     teacher_model = AutoModel.from_pretrained(config['aec']['lightning_module']['teacher_model_checkpoint'])
#     student_model = AutoModel.from_pretrained(config['stmod']['lightning_module']['student_model_checkpoint'])
#     autoencoder_model = AutoEncoder(input_dim_teacher=1024, bottleneck_dim=256, input_dim_student=768)

#     aec_model = AecModule(student_model, teacher_model, autoencoder_model, config)

#     checkpoint_callback = ModelCheckpoint(
#         monitor='val_loss',
#         dirpath='path/to/save/checkpoints',
#         filename='model-{epoch:02d}-{val_loss:.2f}',
#         save_top_k=3,
#         mode='min'
#     )
    
#     lr_monitor = LearningRateMonitor(logging_interval='step')
    
#     trainer = L.Trainer(
#         max_epochs=10,
#         callbacks=[checkpoint_callback, lr_monitor],
#     )

#     trainer.fit(aec_model, datamodule=data_module)

#     trainer.test(datamodule=data_module)
#     predictions = trainer.predict(datamodule=data_module)
#     torch.save(model.final_model.state_dict(), "./scripts/models/aec/final_model.pth")