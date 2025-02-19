import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Load the test set from a CSV file
dataset_path = '../Augmentation/ch.csv'  # Path to your test set CSV file
df = pd.read_csv(dataset_path)

# Load the tokenizer and model from your pre-trained checkpoint
model_name_or_path = './models/student_roberta_base_wiki_plus_3k_wiki'  # Path to your saved RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
model = RobertaForSequenceClassification.from_pretrained(model_name_or_path)

# Check if GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a custom dataset class
class ToxicCommentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.loc[index, 'Text'])
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # If label column exists, get it, otherwise return None
        if 'Label' in self.data.columns:
            label = self.data.loc[index, 'Label']
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# Create the dataset and dataloader
test_dataset = ToxicCommentDataset(df, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Put the model in evaluation mode
model.eval()

# Initialize lists to store predictions and probabilities
predictions = []
probabilities = []  # To store probabilities for AUC calculation

# Evaluate the model
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get the predicted probabilities for each class
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probabilities.extend(probs[:, 1])  # We need probabilities for the positive class (class 1)
        
        # Get the predicted class (0 or 1) for each sample in the batch
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(preds)

# Add predictions to the original dataframe
df['Predicted_Labels'] = predictions
df['Predicted_Probabilities'] = probabilities  # Optionally, add the probabilities to the dataframe

# Print the first few rows of the dataframe with predictions
print(df.head())

# Optionally save the dataframe with predictions to a new CSV file
df.to_csv('predictions_output.csv', index=False)

# If ground truth labels are available in the dataset, compute evaluation metrics
if 'Label' in df.columns:
    true_labels = df['Label'].values
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    auc = roc_auc_score(true_labels, probabilities)  # AUC based on true labels and predicted probabilities

    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

