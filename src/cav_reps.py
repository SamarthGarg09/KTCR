import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import random
from datasets import load_from_disk
import numpy as np
import yaml
import torch.nn as nn
import torch.nn.functional as F

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, output_dim)
        self.decoder = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

class CAVExtractor:
    def __init__(self, autoencoder, config):
        self.dataset_path = config['cav']['data_module']['save_dataset_path']
        self.tokenizer = AutoTokenizer.from_pretrained(config['aec']['lightning_module']['teacher_model'])
        self.model = AutoModel.from_pretrained(config['aec']['lightning_module']['teacher_model'])
        self.model.eval()
        self.autoencoder = autoencoder  # Assume the hidden size and adjust accordingly.
        self.autoencoder.eval()
        state_dict = torch.load('./models/aec_model_mepoch_0.pth')
        encoder_state_dict = {k.replace('final_model.autoencoder.encoder.', ''): v for k, v in state_dict['model_state_dict'].items() if 'final_model.autoencoder.encoder.' in k}
        decoder_state_dict = {k.replace('final_model.autoencoder.decoder.', ''): v for k, v in state_dict['model_state_dict'].items() if 'final_model.autoencoder.decoder.' in k}
        self.autoencoder.encoder.load_state_dict(encoder_state_dict)
        self.autoencoder.decoder.load_state_dict(decoder_state_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.autoencoder.to(self.device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.batch_size = config['cav']['data_module']['batch_size']
        self.num_workers = 8
        self.cav = None

    def setup(self):
        dataset = load_from_disk(self.dataset_path)
        self.concept_dataset = dataset['train'].filter(lambda x: x['Labels'] == 1)

    def extract_cls_embeddings(self):
        dataloader = DataLoader(
            self.concept_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.data_collator, 
            num_workers=self.num_workers
        )
        
        autoencoder_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                model_outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                self.autoencoder.to('cuda')
                encoded, _, _ = self.autoencoder(model_outputs.last_hidden_state[:, 0, :], model_outputs.last_hidden_state[:, 0, :])
                autoencoder_embeddings.extend(encoded.cpu().numpy())
                
        return autoencoder_embeddings
    
    def calculate_cav(self, embeddings, num_samples=100):
        # selected_embeddings = random.sample(embeddings, num_samples)
        # cav = np.mean(selected_embeddings, axis=0)
        cav = np.mean(embeddings, axis=0)
        return cav

    def save_cav(self, cav, path='./models/cav.npy'):
        np.save(path, cav)
        print(f"CAV saved to {path}")

    def get_cav(self):
        return self.cav
    
    def run(self):
        self.setup()
        cls_embeddings = self.extract_cls_embeddings()
        self.cav = self.calculate_cav(cls_embeddings, random.randint(5, 10))
        self.save_cav(self.cav)

if __name__ == "__main__":
    extractor = CAVExtractor(config)
    extractor.run()
    cav = extractor.get_cav()
    print(f"Extracted CAV: {cav}")
