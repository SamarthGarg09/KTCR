import pickle
import os

def initialize_loss_log_student(file_path):
    # if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({'train_losses': [], 'val_losses': [], 'test_losses': [], 'train_f1': [], 'val_f1': [], 'val_classification_loss': [], 'train_classification_loss': [], 'train_concept_loss': [], 'val_concept_loss':[]}, f)

def initialize_loss_log_cav(file_path):
    # if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({'train_losses': [], 'val_losses': [], 'test_losses': [], 'val_f1': [], 'train_f1': []}, f)

def initialize_loss_log_aec(file_path):
    # if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({'train_losses': [], 'val_losses': [], 'test_losses': []}, f)

def load_loss_log(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_loss_log(file_path, loss_log):
    with open(file_path, 'wb') as f:
        pickle.dump(loss_log, f)
