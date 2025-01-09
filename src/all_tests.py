import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from student_module import StudentDataModule, StudentModule
from src.autoencoder_module import AECDataModule, AecModule, AutoEncoder
from src.cav_classifier_module import CAVDataModule, CAVModule
from sklearn.metrics import f1_score
import torch
from transformers import AutoModel
import yaml
import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch

def load_checkpoint_cav(model, path):
    checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint)
    # epoch = checkpoint['epoch']
    return checkpoint

def print_results(aec_checkpoint_path, cav_checkpoint_path, student_checkpoint_path, teacher_model_path, student_model_path, test_set_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    aec_data_module = AECDataModule(config=config)
    aec_data_module.setup()
    cav_data_module = CAVDataModule(config=config)
    cav_data_module.setup()
    student_data_module = StudentDataModule(config=config)
    student_data_module.setup()

    teacher_model = AutoModel.from_pretrained(teacher_model_path)
    student_model = AutoModel.from_pretrained(student_model_path) 
    autoencoder_model = AutoEncoder(input_dim_teacher=1024, bottleneck_dim=256)  
    aec_data_module_len = len(aec_data_module.train_dataloader())
    cav_data_module_len = len(cav_data_module.train_dataloader())
    student_data_module_len = len(student_data_module.test_dataloader())

    lit_model = AecModule(
        student_model=student_model, 
        teacher_model=teacher_model, 
        autoencoder_model=autoencoder_model, 
        config=config,
        aec_data_module_len=aec_data_module_len
    )

    cav_model = CAVModule(
        teacher_model=teacher_model, 
        autoencoder = autoencoder_model,
        config=config,
        cav_data_module_len=cav_data_module_len
    )

    student_module = StudentModule(
        student_model=student_model, 
        cav_model = cav_model,
        data_module=student_data_module, 
        config=config,
        student_data_module_len=student_data_module_len
    )

    state_dict = torch.load(student_checkpoint_path)
    new_state_dict = {}
    for n, p in state_dict['model_state_dict'].items():
        if n.startswith('cls_head'):
            new_key = n[9:]
            new_state_dict[new_key] = p
    print(new_state_dict)
    cls_head = torch.nn.Linear(768, 2)
    cls_head.load_state_dict(new_state_dict)

    if os.path.exists(aec_checkpoint_path):
        lit_model, lit_epoch = load_checkpoint(lit_model, aec_checkpoint_path)
    if os.path.exists(cav_checkpoint_path):
        cav_model = load_checkpoint_cav(cav_model, cav_checkpoint_path)
    if os.path.exists(student_checkpoint_path):
        student_module, student_epoch = load_checkpoint(student_module, student_checkpoint_path)

    df = pd.read_csv(test_set_path)

    df = df[['Text', 'Label']]
    # concept_df = pd.read_csv('./Data/csv_files/concept_set.csv')

    # # Remove rows where df['text'] matches concept_df['Text']
    # df = df[~df['text'].isin(concept_df['Text'])]

    # # Optional: Reset index after removing rows
    # df.reset_index(drop=True, inplace=True)


    # label_mapping = {
    #     'none_of_the_above': 0,ch
    #     'entity_directed_hostility': 1,
    #     'entity_directed_criticism': 1,
    #     'discussion_of_eastasian_prejudice': 0,
    #     'counter_speech': 0
    # }

    # df['Label'] = df['expert'].map(label_mapping)

    df = df.dropna(subset=['Label'])

    hf_dataset = Dataset.from_pandas(df[['Text', 'Label']])

    # hf_dataset = load_from_disk('./models/student_roberta_base_wiki/test_set')

    tokenizer = AutoTokenizer.from_pretrained(student_model_path)


    def tokenize_function(examples):
        return tokenizer(examples['Text'], padding='max_length', truncation=True)

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])

    dataloader = DataLoader(tokenized_dataset, batch_size=50, shuffle=True)

    val_preds = []
    val_labels = []
    val_probs = []

    student_module = student_module.to(device)


    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['Label'].to(device)

        # Forward pass
        outputs = student_module(input_ids, attention_mask, labels)
        logits = outputs[0]

        # Process the logits and pass through cls_head
        cls_embed = logits[:, 0, :]
        logits = student_module.cls_head(nn.functional.relu(cls_embed))

        # Convert labels to long type
        labels = labels.long()

        # Get the predicted probabilities
        probs = F.softmax(logits, dim=1)  # Use softmax to get probabilities for each class
        preds = torch.argmax(logits, dim=1)

        # Store predictions, true labels, and probabilities for evaluation
        val_preds.append(preds.detach().cpu().numpy())
        val_labels.append(labels.detach().cpu().numpy())
        val_probs.append(probs.detach().cpu().numpy())

    val_preds = np.concatenate(val_preds)  # Combine list of numpy arrays into one
    val_labels = np.concatenate(val_labels)  # Same for labels
    val_probs = np.concatenate(val_probs)  # Same for probabilities

    f1 = f1_score(val_labels, val_preds, average='macro')
    print(f1)

    auc = roc_auc_score(val_labels, val_probs[:, 1])  # AUC for the positive class
    print(f"AUC: {auc}")

if __name__ == "__main__":
    folder_paths = ['../saved_models/concept_chcekpoints_wiki_only_student_3k_teacher_concept/', '../saved_models/concept_chcekpoints_wiki_only_student_5k_teacher_concept/', '../saved_models/concept_chcekpoints_wiki_plus_3k_student/', '../saved_models/concept_chcekpoints_wiki_plus_3k_student_5k_teacher_concept/', '../saved_models/concept_chcekpoints_wiki_plus_3k_student_5k_teacher_concept_all_balanced/','../saved_models/concept_chcekpoints_wiki_plus_3k_student_8k_teacher_concept/']

    aec_files = ['aec_model_mepoch_2.pth', 'aec_model_mepoch_2.pth', 'aec_model_mepoch_5.pth', 'aec_model_mepoch_2.pth', 'aec_model_mepoch_2.pth', 'aec_model_mestudent_roberta_base_wiki_plus_3k_wiki_mepoch_2.pthpoch_2.pth', 'aec_model_mepoch_2']

    cav_files = ['cav.pth' for _ in range(len(folder_paths))]

    student_files = ['student_roberta_base_wiki_only_mepoch_2.pth', 'student_roberta_base_wiki_only_mepoch_2.pth', 'student_model_wiki_mepoch_5.pth', 'student_model_wiki_3k_wiki_mepoch_2.pth', 'student_roberta_base_wiki_3k_ea_balanced_mepoch_2.pth', 'student_roberta_base_wiki_plus_3k_wiki_mepoch_2.pth', 'student_roberta_base_wiki_3k_ea_balanced_mepoch_2.pth']

    teacher_models = ['../saved_models/teacher_3k_ea/teacher_ft_ea', '../saved_models/teacher_5k_ea/teacher_ft_ea_5k', '../saved_models/teacher_3k_ea/teacher_ft_ea', '../saved_models/teacher_5k_ea/teacher_ft_ea_5k', './models/teacher_ft_ea_5k_balanced', '../saved_models/teacher_8k_ea/teacher_ft_ea_8k']

    student_models = ['../saved_models/student_wiki_only/student_roberta_base_wiki', '../saved_models/student_wiki_only/student_roberta_base_wiki', '../saved_models/student_wiki_plus_3k/student_roberta_base_wiki_plus_3k_wiki', '../saved_models/student_wiki_plus_3k/student_roberta_base_wiki_plus_3k_wiki', './models/student_roberta_base_wiki_3k_ea_balanced', '../saved_models/student_wiki_plus_3k/student_roberta_base_wiki_plus_3k_wiki']

    test_set_path = '../Augmentation/test_ea.csv'

    for path, aec, cav, std, teacher_model, student_model in zip(folder_paths, aec_files, cav_files, student_files, teacher_models, student_models):
        print_results(path + aec, path + cav, path + std, teacher_model, student_model, test_set_path)
    print_results('./models/aec_model_mepoch_2.pth', './models/cav.pth', './models/student_roberta_base_wiki_3k_ea_balanced_mepoch_2.pth', './models/teacher_ft_ea_3k_balanced', './models/student_roberta_base_wiki_3k_ea_balanced/', test_set_path)


