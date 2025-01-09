import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import RobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import os
import wandb
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import yaml
import warnings
from transformers import logging as transformers_logging
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from transformers import DataCollatorWithPadding

# from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
# from torchmetrics import MetricCollection

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained('./models/student_roberta_base_wiki')
class WikiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_token_len=256):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment
        labels = data_row.toxicity

        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels, dtype=torch.long)
        )

import torch
import pytorch_lightning as pl
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained('./models/student_roberta_base_wiki', num_labels=n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.test_preds = []
        self.test_labels = []

    def forward(self, input_ids, attention_mask, labels=None):
        labels = labels.long()
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["Label"]  # Ensure this matches the key in your data batch
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["Label"]  # Ensure this matches the key in your data batch
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["Label"]  # Ensure this matches the key in your data batch
        loss, logits = self(input_ids, attention_mask, labels)
        preds = logits.argmax(dim=-1)
        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds, dim=0)
        all_labels = torch.cat(self.test_labels, dim=0)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
        self.log("F1_score", f1, on_epoch=True, prog_bar=True, logger=True)
        self.test_preds = []
        self.test_labels = []
        cm = confusion_matrix(all_labels, all_preds)   
        df_cm = pd.DataFrame(cm, index=[i for i in range(cm.shape[0])], columns=[i for i in range(cm.shape[1])])
        plt.figure(figsize=(10,7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_TEACHER.png')
        plt.show()
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        num_train_steps = self.n_epochs * self.steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_steps * 0.1), num_training_steps=num_train_steps)
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}



wandb.init(project='concept_distillation')
logger=WandbLogger()

checkpoint_dir = "./models/student_roberta_base_wiki_plus_3k_wiki_lightning"
os.makedirs(checkpoint_dir, exist_ok=True)

    
# df = pd.read_csv("./Data/csv_files/wiki_dataset.csv")
# assert 'split' in df.columns, "Dataframe must include a 'split' column."
# train_df = df[df.split == 'train']
# val_df = df[df.split == 'dev']
# test_df = df[df.split == 'test']

def tokenize( example):
    student_inputs = tokenizer(example['Text'], truncation=True, max_length=config['stmod']['data_module']['tokenized_max_length_source'])
    return {
        'input_ids': student_inputs['input_ids'],
        'attention_mask': student_inputs['attention_mask'],
        'labels': example['Label']
    }
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
train_df = pd.read_csv("../Augmentation/balanced_ea_student_3k.csv")
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
train_dataset = train_dataset.shuffle(seed=42)

val_df = pd.read_csv("../Augmentation/wiki_augment.csv")
val_dataset = Dataset.from_pandas(val_df)
val_dataset = val_dataset.map(tokenize, batched=True)
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
val_dataset = val_dataset.shuffle(seed=42)

test_df = pd.read_csv("../Augmentation/test_ea.csv")
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
test_dataset = test_dataset.shuffle(seed=42)

train_dataset.save_to_disk('./models/student_roberta_base_wiki_3k_ea_balanced' + '/train_set')
val_dataset.save_to_disk('./models/student_roberta_base_wiki_3k_ea_balanced' + '/val_set')
test_dataset.save_to_disk('./models/student_roberta_base_wiki_3k_ea_balanced' + '/test_set')
train_dataset = load_from_disk('./models/student_roberta_base_wiki_3k_ea_balanced' + '/train_set')
val_dataset = load_from_disk('./models/student_roberta_base_wiki_3k_ea_balanced' + '/val_set')
test_dataset = load_from_disk('./models/student_roberta_base_wiki_3k_ea_balanced' + '/test_set')
# train_dataset = WikiDataset(train_df, tokenizer)
# val_dataset = WikiDataset(val_df, tokenizer)
# test_dataset = WikiDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=7, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=7, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=7, collate_fn=data_collator)

model = ToxicCommentClassifier(n_classes=2, steps_per_epoch=len(train_loader), n_epochs=6)

lr_monitor = LearningRateMonitor(logging_interval='step')
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=3,
    verbose=True,
    mode='min'
)
trainer = pl.Trainer(max_epochs=6, fast_dev_run=False, callbacks=[lr_monitor, early_stop_callback], logger=logger)

trainer.fit(model, train_loader, val_loader)

trainer.test(model, test_loader)

model.model.save_pretrained('./models/student_roberta_base_wiki_3k_ea_balanced')
tokenizer.save_pretrained('./models/student_roberta_base_wiki_3k_ea_balanced')
