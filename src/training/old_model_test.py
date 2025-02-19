import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

# Load the dataset from a saved .arrow file (using load_from_disk for datasets in HF format)
dataset_path = '/scratch/Abusiveness-Detection-via-CD/Augmentation/test_ea.csv'  # Path to the Hugging Face dataset (.arrow format)
dataset = load_from_disk(dataset_path)

# Load the tokenizer and model from your pre-trained checkpoint
model_name_or_path = './models/retrained_teacher_roberta_large_doe_false_true'  # Path to your saved RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
model = RobertaForSequenceClassification.from_pretrained(model_name_or_path)

# Check if GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a preprocessing function to tokenize the text data
def preprocess_function(examples):
    return tokenizer(examples['Text'], truncation=True, max_length=512)

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Remove the 'Text' column since it is not needed after tokenization
tokenized_dataset = tokenized_dataset.remove_columns(['Text'])

# Convert tokenized inputs to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Use DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a DataLoader for evaluation
from torch.utils.data import DataLoader

eval_dataloader = DataLoader(tokenized_dataset, batch_size=16, collate_fn=data_collator)

# Put the model in evaluation mode
model.eval()

# Initialize lists to store predictions and probabilities
predictions = []
probabilities = []  # To store probabilities for AUC calculation

# Evaluate the model
with torch.no_grad():
    for batch in tqdm(eval_dataloader):
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

# Convert the dataset back to a Pandas DataFrame for easier manipulation and analysis
df = dataset.to_pandas()

# Add predictions to the original dataframe
df['Predicted_Labels'] = predictions
df['Predicted_Probabilities'] = probabilities  # Optionally, add the probabilities to the dataframe

# Print the first few rows of the dataframe with predictions
print(df.head())

# Optionally save the dataframe with predictions to a new CSV file
df.to_csv('predictions_output.csv', index=False)
# print(predictions)

# If ground truth labels are available in the dataset, compute evaluation metrics
if 'Label' in df.columns:
    true_labels = df['Label'].values
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    auc = roc_auc_score(true_labels, probabilities)  # AUC based on true labels and predicted probabilities

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")
