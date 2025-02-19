import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.cluster import KMeans
import pytorch_lightning as L
import numpy as np
from tqdm import tqdm
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from pytorch_lightning import LightningModule
import pandas as pd
from datasets import Dataset, concatenate_datasets
from torchviz import make_dot
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from collections import Counter
import matplotlib.pyplot as plt
from utils import *
import torch

class StudentDataModule(L.LightningDataModule):
    
    def __init__(self, config):
        
        super().__init__()
        self.save_hyperparameters()
        self.config=config
        self.num_workers = config['stmod']['data_module']['num_workers']
        self.batch_size = config['stmod']['data_module']['batch_size']
        self.tokenizer = AutoTokenizer.from_pretrained(config['stmod']['lightning_module']['student_model'])
        self.save_dataset_path = config['stmod']['data_module']['save_dataset_path']
        self.tokenized_max_length_source = config['stmod']['data_module']['tokenized_max_length_source']
        self.tokenized_max_length_target = config['stmod']['data_module']['tokenized_max_length_target']
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.train_dataset=None
        self.val_dataset=None
        self.test_dataset=None

    # def prepare_data(self):
    #     df = pd.read_csv("./Data/csv_files/wiki_dataset_comments.csv")
    #     dataset = Dataset.from_pandas(df)
    #     self.dataset = dataset.map(self.tokenize, batched=True)
    #     self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
    #     self.dataset = self.dataset.shuffle(seed=42)

    #     pos_label = 1
    #     positive_samples = self.dataset.filter(lambda x: x['Label'] == pos_label)
    #     negative_samples = self.dataset.filter(lambda x: x['Label'] != pos_label)

    #     train_size = 43737
    #     dev_size = 32128
    #     test_size = 31866

    #     train_pos_samples = int(17 / 100 * train_size)
    #     train_neg_samples = train_size - train_pos_samples
    #     dev_pos_samples = int(9 / 100 * dev_size)
    #     dev_neg_samples = dev_size - dev_pos_samples
    #     test_pos_samples = int(9 / 100 * test_size)
    #     test_neg_samples = test_size - test_pos_samples
        
    #     train_positive_samples = positive_samples.select(range(min(train_pos_samples, len(positive_samples))))
    #     train_negative_samples = negative_samples.select(range(min(train_neg_samples, len(negative_samples))))
    #     dev_positive_samples = positive_samples.select(range(min(train_pos_samples, len(positive_samples)), min(train_pos_samples + dev_pos_samples, len(positive_samples))))
    #     dev_negative_samples = negative_samples.select(range(min(train_neg_samples, len(negative_samples)), min(train_neg_samples + dev_neg_samples, len(negative_samples))))
    #     test_positive_samples = positive_samples.select(range(min(train_pos_samples + dev_pos_samples, len(positive_samples)), min(train_pos_samples + dev_pos_samples + test_pos_samples, len(positive_samples))))
    #     test_negative_samples = negative_samples.select(range(min(train_neg_samples + dev_neg_samples, len(negative_samples)), min(train_neg_samples + dev_neg_samples + test_neg_samples, len(negative_samples))))

    #     # Combine positive and negative samples for each set
    #     train_data = concatenate_datasets([train_positive_samples, train_negative_samples]).shuffle(seed=42)
    #     dev_data = concatenate_datasets([dev_positive_samples, dev_negative_samples]).shuffle(seed=42)
    #     test_data = concatenate_datasets([test_positive_samples, test_negative_samples]).shuffle(seed=42)

    #     self.train_dataset = train_data
    #     self.val_dataset = dev_data
    #     self.test_dataset = test_data

    #     print('Train: ', len(self.train_dataset))
    #     print('Validation: ', len(self.val_dataset))
    #     print('Test: ', len(self.test_dataset))

    #     self.train_dataset.save_to_disk(self.save_dataset_path + '/train_set')
    #     self.val_dataset.save_to_disk(self.save_dataset_path + '/val_set')
    #     self.test_dataset.save_to_disk(self.save_dataset_path + '/test_set')

    def tokenize(self, example):
        student_inputs = self.tokenizer(example['Text'], truncation=True, max_length=self.tokenized_max_length_source)
        return {
            'input_ids': student_inputs['input_ids'],
            'attention_mask': student_inputs['attention_mask'],
            'labels': example['Label']
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = load_from_disk(self.config['stmod']['data_module']['save_dataset_path'] + '/train_set')
            self.val_dataset = load_from_disk(self.config['stmod']['data_module']['save_dataset_path'] + '/val_set')
            # self.train_dataset = self.train_dataset.select(range(min(self.config['stmod']['data_module']['train_sample_size'], len(self.train_dataset))))
            # self.val_dataset = self.val_dataset.select(range(min(self.config['stmod']['data_module']['validation_sample_size'], len(self.val_dataset))))

        if stage == 'test' or stage is None:
            self.test_dataset = load_from_disk(self.config['stmod']['data_module']['save_dataset_path'] + '/test_set')
            # self.test_dataset = self.test_dataset.select(range(min(self.config['stmod']['data_module']['test_sample_size'], len(self.test_dataset))))

        print('Train: ', len(self.train_dataset))
        print('Validation: ', len(self.val_dataset))
        print('Test: ', len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers, pin_memory=True)#, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers, pin_memory=True)#, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers, pin_memory=True)#, persistent_workers=True)

def compute_class_weights(labels):
    counts = Counter(labels)
    total_samples = len(labels)
    
    class_weights = {cls: total_samples / (len(counts) * count) for cls, count in counts.items()}
    return class_weights

class StudentModule(LightningModule):
    def __init__(
        self, 
        student_model,  
        cav_model,
        data_module, 
        config,
        **kwargs
    ):
        super().__init__()
      
        self.model = student_model
        self.aec = cav_model.autoencoder
        # self.output_model = cav_model.output_model
        for param in self.model.parameters():
            param.requires_grad=True
            
        self._freeze_model(self.aec)
        # self._freeze_model(self.output_model)
        
        self.learning_rate = config['stmod']['lightning_module']['learning_rate']
        self.lambda_ = config['stmod']['lightning_module']['lambda']
        self.alpha = config['main']['alpha']
        self.config=config
        self.steps_per_epoch = len(data_module.train_dataloader())
        self.data_module = data_module
        self.val_dataloader = data_module.val_dataloader()
        self.test_dataloader = data_module.test_dataloader()
        self.class_labels = np.arange(2)
        self.proto_dic = {}
        self.bottleneck_layer = config['stmod']['lightning_module']['bottleneck_layer']
        self.do_proto_mean = True
        self.knn_k = config['stmod']['lightning_module']['n_clusters']
        self.cls_head = nn.Linear(768, 2)
        self.mse_loss = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=0)
        self.use_proto = True
        self.cav = None  
        self.kmeans = KMeans(
            n_clusters=config['stmod']['lightning_module']['n_clusters'], 
            random_state=config['stmod']['lightning_module']['random_state_kmeans'], 
            n_init='auto'
        )
        self.val_preds = []
        self.val_labels = []
        self.train_losses = []
        self.train_preds = []
        self.train_labels = []
        self.val_losses = []
        self.test_probs = []
        self.test_preds = []
        self.test_labels = []
        self.train_clf_losses = []
        self.train_concept_losses = []
        self.val_clf_losses = []
        self.val_concept_losses = []

        all_labels = []
        for batch in data_module.train_dataloader():
            all_labels.extend(batch['Label'].numpy())
        class_weights = compute_class_weights(all_labels)
        weight_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)

        self.log_file_path = 'training_metrics/student_training.pkl'
        self.curves_file_path = 'training_curve/student_training.png'
        initialize_loss_log_student(self.log_file_path)
        self.classification_criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-100)
        
    def _freeze_model(self, model):
        
        for param in model.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask, labels=None):
        
        # def custom_forward(*inputs):
        #     return self.model(*inputs)
        # outputs = checkpoint(custom_forward, input_ids, attention_mask)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs

    def update_cav(self, cav):
        if isinstance(cav, np.ndarray):
            cav = torch.from_numpy(cav)

        if not cav.is_cuda:
            cav = cav.to('cuda')
        
        self.cav = cav
        # return self.cav
    
    def get_intermediate_activations(self, input_ids, attention_mask, stage=None):
        
        activations = None

        def hook(module, input, output):
            nonlocal activations
            activations = output[0][:, 0, :]
        
        handle = self.model.encoder.layer[self.bottleneck_layer].register_forward_hook(hook)
        if not stage:
            with torch.set_grad_enabled(True):
                self.model.to(input_ids.device)  
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            self.model.to(input_ids.device)  
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        handle.remove()
    
        if len(activations) == 0:
            raise RuntimeError("Hook did not capture activations")
            
        return activations, outputs


    def get_concept_loss(self, input_ids, attention_mask, labels, affect=False):
        
        dot_lis = []
        cos_lis = []
        mse_dvec_lis = []
        
        for i in range(input_ids.shape[0]):
            activation = None
            
            def hook(module, input, output):
                nonlocal activation
                activation = output[0][:, 0, :]
            
            handle = self.model.encoder.layer[self.bottleneck_layer].register_forward_hook(hook)
            
            with torch.set_grad_enabled(True):
                self.model.to(input_ids.device)  # Ensure model is on the same device as input_ids
                outputs = self.model(input_ids=input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
                
            handle.remove()
            # activation = self.output_model[:-1](F.relu(self.aec.reduction(activation)))
            activation = self.aec.reduction(activation)

            if len(activation) == 0:
                raise RuntimeError("Hook did not capture activation")

            if self.use_proto:
                for j in range(self.knn_k):
                    
                    data = self.proto_dic[labels[i].item()]['cluster_' + str(j)]
                    tensor_data = torch.from_numpy(data).to('cuda').requires_grad_() if isinstance(data, np.ndarray) else data.to('cuda').requires_grad_()
                    tensor_data = tensor_data.requires_grad_(True) 
                    # tensor_data = self.output_model[:-1](F.relu(self.aec.reduction(tensor_data)))
                    tensor_data = self.aec.reduction(tensor_data)
                    
                    if j == 0:
                        loss_a = self.mse_loss(activation, tensor_data)
                    else:
                        loss_a += self.mse_loss(activation, tensor_data)
                        
                    assert activation.requires_grad, "activation does not require gradients"
                    
                grad_ = torch.autograd.grad(loss_a, activation, retain_graph=True, create_graph=True)
            else:
                grad_ = torch.autograd.grad(outputs[0], activation, retain_graph=True, create_graph=True)
    
            if grad_[0] is None:
                raise RuntimeError(f"Gradients not calculated properly for activation[{i}]")
            # print(grad_[0].to('cuda').float().squeeze(0).flatten().shape)
            # print(self.cav.to('cuda').float().shape)
            self.cav_resized = self.cav.to('cuda').float().repeat(4)[:256]
            self.cav = self.cav_resized
            dot = torch.dot(grad_[0].to('cuda').float().squeeze(0).flatten(), self.cav.to('cuda').float()) / torch.linalg.norm(self.cav.to('cuda').float())
            dot_lis.append(dot)
    
            unit_grad = grad_[0].to('cuda').float().squeeze(0).flatten()
            unit_direc = self.cav.to('cuda').float() / torch.linalg.norm(self.cav.to('cuda').float())
            unit_grad = unit_grad / torch.linalg.norm(unit_grad)
            mse_dvec_lis.append(self.mse_loss(unit_grad, unit_direc))
    
            cos_ = self.cos(grad_[0].to('cuda').float().squeeze(0).flatten(), self.cav.to('cuda').float())
            cos_lis.append(cos_)
    
        dot_lis = torch.stack(dot_lis)
        mse_dvec_lis = torch.stack(mse_dvec_lis)
        cos_lis = torch.stack(cos_lis)
        
        if affect == False:
            loss_ = torch.sum(torch.abs(cos_lis))  # L1
        else:
            loss_ = self.mse_loss(cos_lis, torch.ones(len(cos_lis)).to('cuda'))
    
        return loss_

    def backward(self, loss):
        
        loss.backward(retain_graph=True)
    
    def training_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['Label']
        outputs = self(input_ids, attention_mask)
        logits = outputs[0] 
        cls_embed = logits[:, 0, :]
        logits = self.cls_head(F.relu(cls_embed))
        labels = labels.long()
        classification_loss = self.classification_criterion(logits, labels.view(-1))
        
        if not self.proto_dic:
            self.initialize_prototypes(self.data_module)
        
        concept_loss = self.get_concept_loss(input_ids, attention_mask, labels, affect=False)
        total_loss = classification_loss + self.lambda_ * concept_loss

        # self.log('classification_loss', classification_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('concept_loss', concept_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        preds = torch.argmax(logits, dim=1)
        self.train_preds.append(preds.detach().cpu().numpy())
        self.train_labels.append(labels.detach().cpu().numpy())
        self.train_losses.append(total_loss.detach().cpu().numpy())
        self.train_clf_losses.append(classification_loss.detach().cpu().numpy())
        self.train_concept_losses.append(concept_loss.detach().cpu().numpy())

        del input_ids, attention_mask, labels, outputs, logits, preds
        torch.cuda.empty_cache()

        return total_loss

    def on_train_epoch_end(self):
        avg_loss_epoch = np.mean(self.train_losses)
        avg_clf_loss = np.mean(self.train_clf_losses)
        avg_concept_loss = np.mean(self.train_concept_losses)
        all_preds = np.concatenate(self.train_preds)
        all_labels = np.concatenate(self.train_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        loss_log = load_loss_log(self.log_file_path)
        loss_log['train_classification_loss'].append(avg_clf_loss)
        loss_log['train_concept_loss'].append(avg_concept_loss)
        loss_log['train_losses'].append(avg_loss_epoch)
        loss_log['train_f1'].append(f1)
        save_loss_log(self.log_file_path, loss_log)
        
        self.log("train_loss", avg_loss_epoch, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_F1_score", f1, on_epoch=True, prog_bar=True, logger=True)

        self.train_preds = []
        self.train_labels = []
        self.train_losses = []
        self.train_clf_losses = [] 
        self.train_concept_losses = []

        torch.cuda.empty_cache()
        
    def initialize_prototypes(self, data_module, phase='train'):
        
        self.model.to('cuda')
            
        for c in self.class_labels:
            if phase == 'train':
                dataset = data_module.train_dataset.filter(lambda x: x['Label'] == c)
            elif phase == 'val':
                dataset = data_module.val_dataset.filter(lambda x: x['Label'] == c)
            else:
                dataset = data_module.test_dataset.filter(lambda x: x['Label'] == c)
                
            if len(dataset) > self.knn_k:
                
                dataloader = DataLoader(dataset, batch_size=128, collate_fn=data_module.data_collator, shuffle=True, num_workers=self.config['stmod']['data_module']['num_workers'])
                activations = []
                with torch.no_grad():
                    for batch in tqdm(dataloader):
                        batch = {k: v.to('cuda') for k, v in batch.items()}
                        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['Label']
                        
                        activation, _ = self.get_intermediate_activations(input_ids, attention_mask, 'proto')
                        activations.append(activation.detach().cpu().numpy())

                activations = np.concatenate(activations, axis=0)
                activations = activations.reshape(activations.shape[0], -1)
                self.kmeans.fit(activations)
                centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float, device='cuda', requires_grad=True)
                for f in range(self.knn_k):
                    if not self.do_proto_mean:
                        if c in self.proto_dic:
                            self.proto_dic[c]['cluster_' + str(f)] = centers[f].reshape(1, -1)
                        else:
                            self.proto_dic[c] = {}
                            self.proto_dic[c]['cluster_' + str(f)] = centers[f].reshape(1, -1)
                    else:
                        if c in self.proto_dic and 'cluster_' + str(f) in self.proto_dic[c]:
                            self.proto_dic[c]['cluster_' + str(f)] = (1 - self.alpha) * self.proto_dic[c]['cluster_' + str(f)] + self.alpha * centers[f].reshape(1, -1)
                        elif c in self.proto_dic:
                            self.proto_dic[c]['cluster_' + str(f)] = centers[f].reshape(1, -1)
                        else:
                            self.proto_dic[c] = {}
                            self.proto_dic[c]['cluster_' + str(f)] = centers[f].reshape(1, -1)              
    
        return self.proto_dic
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['Label']
        outputs = self(input_ids, attention_mask, labels)
        logits = outputs[0]
        cls_embed = logits[:, 0, :]
        logits = self.cls_head(nn.functional.relu(cls_embed))
        labels = labels.long()
        classification_loss = self.classification_criterion(logits, labels.view(-1))
        
        if not self.proto_dic:
            self.initialize_prototypes(self.data_module, phase='val')
            
        with torch.set_grad_enabled(True):    
            concept_loss = self.get_concept_loss(
                input_ids, 
                attention_mask, 
                labels, 
                affect=False
            )
        
        total_loss = classification_loss + self.lambda_ * concept_loss
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.argmax(logits, dim=1)
        self.val_preds.append(preds.detach().cpu().numpy())
        self.val_labels.append(labels.detach().cpu().numpy())
        self.val_losses.append(total_loss.detach().cpu().numpy())
        self.val_clf_losses.append(classification_loss.detach().cpu().numpy())
        self.val_concept_losses.append(concept_loss.detach().cpu().numpy())
        
        return {'val_loss': total_loss}

    def on_validation_epoch_end(self):
        
        avg_loss_epoch = np.mean(self.val_losses)
        avg_clf_loss = np.mean(self.val_clf_losses)
        avg_concept_loss = np.mean(self.val_concept_losses)
        all_preds = np.concatenate(self.val_preds)
        all_labels = np.concatenate(self.val_labels)
        
        f1 = f1_score(all_labels, all_preds, average='macro')

        loss_log = load_loss_log(self.log_file_path)
        loss_log['val_losses'].append(avg_loss_epoch)
        loss_log['val_f1'].append(f1)
        loss_log['val_classification_loss'].append(avg_clf_loss)
        loss_log['val_concept_loss'].append(avg_concept_loss)
        save_loss_log(self.log_file_path, loss_log)

        self.log('val_loss_epoch', avg_loss_epoch, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_epoch', f1, on_epoch=True, prog_bar=True, logger=True)

        self.val_losses = []
        self.val_preds = []
        self.val_labels = []
        self.val_clf_losses = []
        self.val_concept_losses = []

        torch.cuda.empty_cache()
        
    def test_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['Label']
        outputs = self(input_ids, attention_mask, labels)
        logits = outputs[0]
        cls_embed = logits[:, 0, :]
        logits = self.cls_head(nn.functional.relu(cls_embed))
        labels = labels.long()
        classification_loss = self.classification_criterion(logits, labels.view(-1))

        # if not self.proto_dic:
        #     self.initialize_prototypes(self.data_module, phase='test')
        # with torch.set_grad_enabled(True):
        #     concept_loss = self.get_concept_loss(
        #         input_ids, 
        #         attention_mask, 
        #         labels,
        #         affect=False
        #     )
        # total_loss = classification_loss + self.lambda_ * concept_loss
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=1)
        self.test_probs.append(probs.detach().cpu().numpy())
        self.test_preds.append(preds.detach().cpu().numpy())
        self.test_labels.append(labels.detach().cpu().numpy())
        return None#{"test_loss": loss}
        
    def plot_and_save_confusion_matrix(self, conf_matrix, labels, output_path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(output_path)
        plt.show()
    
    def on_test_epoch_end(self):

        all_preds = np.concatenate(self.test_preds, axis=0)
        all_labels = np.concatenate(self.test_labels, axis=0)
        all_probs = np.concatenate(self.test_probs, axis=0)

        f1 = f1_score(all_labels, all_preds, average='macro')
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # loss_log = load_loss_log(self.log_file_path)
        # loss_log['test_losses'].append(avg_loss_epoch)
        # loss_log['test_f1'].append(f1)
        # save_loss_log(self.log_file_path, loss_log)

        # self.log("F1_score", f1, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ROC_AUC", roc_auc, on_epoch=True, prog_bar=True, logger=True)
        
        class_labels = ['Non-Abusive', 'Abusive']  
        output_path = "confusion_matrix.png"
        self.plot_and_save_confusion_matrix(conf_matrix, class_labels, output_path)
        
        # Optionally log the confusion matrix image if your logging framework supports it
        # Example: self.logger.experiment.add_image("Confusion Matrix", plt.imread(output_path))

        self.test_preds = []
        self.test_labels = []
        self.test_probs = []
        torch.cuda.empty_cache()

    def plot_losses(self):
        loss_log = load_loss_log(self.log_file_path)
        
        # Plot for general losses
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
        plt.savefig(self.curves_file_path.replace('.png', '_general_loss.png'))
        plt.show()

        # Plot for classification losses
        plt.figure(figsize=(10, 6))
        
        if loss_log['train_classification_loss']:
            train_classification_loss = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['train_classification_loss']]
            plt.plot(train_classification_loss, label='Training Classification Loss')
            
        if loss_log['val_classification_loss']:
            val_classification_loss = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['val_classification_loss']]
            plt.plot(val_classification_loss, label='Validation Classification Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Classification Loss')
        plt.title('Classification Loss Curves')
        plt.legend()
        plt.savefig(self.curves_file_path.replace('.png', '_classification_loss.png'))
        plt.show()

        # Plot for concept losses
        plt.figure(figsize=(10, 6))
        
        if loss_log['train_concept_loss']:
            train_concept_loss = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['train_concept_loss']]
            plt.plot(train_concept_loss, label='Training Concept Loss')
            
        if loss_log['val_concept_loss']:
            val_concept_loss = [x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in loss_log['val_concept_loss']]
            plt.plot(val_concept_loss, label='Validation Concept Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Concept Loss')
        plt.title('Concept Loss Curves')
        plt.legend()
        plt.savefig(self.curves_file_path.replace('.png', '_concept_loss.png'))
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
        plt.savefig('val_train_f1_2.png')
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            betas=(0.9, 0.999), 
            weight_decay=0.01
        )
        total_steps = self.steps_per_epoch 
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
