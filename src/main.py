import torch
import transformers
# import wandb
from src.autoencoder_module import AECDataModule, AecModule, AutoEncoder
from src.cav_classifier_module import CAVDataModule, CAVModule
from student_module import StudentDataModule, StudentModule
from cav_reps import CAVExtractor
from transformers import AutoModel
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from itertools import islice
from tqdm.auto import tqdm
import numpy as np
import json
import yaml
import os
import gc

os.environ['TOKENIZERS_PARALLELISM']='false'

transformers.logging.set_verbosity_error()
torch.set_float32_matmul_precision('medium')

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
def save_checkpoint(model, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch

def concept_distillation():
    print('Initialize data modules')
    aec_data_module = AECDataModule(config=config)
    aec_data_module.setup()
    
    cav_data_module = CAVDataModule(config=config)
    cav_data_module.setup()
    
    student_data_module = StudentDataModule(config=config)
    # student_data_module.prepare_data()
    student_data_module.setup()
    
    teacher_model = AutoModel.from_pretrained(config['aec']['lightning_module']['teacher_model_checkpoint'])
    student_model = AutoModel.from_pretrained(config['stmod']['lightning_module']['student_model']) 
    autoencoder_model = AutoEncoder(input_dim_teacher=1024, bottleneck_dim=256)  
    
    aec_data_module_len = len(aec_data_module.train_dataloader())
    cav_data_module_len = len(cav_data_module.train_dataloader())
    student_data_module_len = len(student_data_module.train_dataloader())
    
    AEC_CHECKPOINT_PATH = config['aec']['lightning_module']['aec_checkpoint_path']
    CAV_CHECKPOINT_PATH = config['cav']['lightning_module']['cav_checkpoint_path']
    STMOD_CHECKPOINT_PATH = config['stmod']['lightning_module']['student_checkpoint_path']
    
    lit_model = AecModule(
        student_model=student_model, 
        teacher_model=teacher_model, 
        autoencoder_model=autoencoder_model, 
        config=config,
        aec_data_module_len=aec_data_module_len
    )
    cav_model = CAVModule(
        teacher_model=teacher_model, 
        autoencoder=autoencoder_model,
        config=config,
        cav_data_module_len=cav_data_module_len
    )
    # cav_model = CAVExtractor(autoencoder_model, config)
    student_module = StudentModule(
        student_model=student_model, 
        cav_model = cav_model,
        data_module=student_data_module, 
        config=config,
        student_data_module_len=student_data_module_len
    )
    # new_cav = np.load('./models/cav.npy')
    N = 0
    cav_update_frequency = config['main']['cav_update_frequency']
    proto_update_freq = config['main']['proto_update_frequency']
    # student_module.initialize_prototypes(student_data_module)

    while N < config['main']['max_epochs']:
        if N % cav_update_frequency == 0:
            # wandb_logger_aec = WandbLogger(name='Trainer_AEC', project='CD-Abusive2', id='aec', resume='allow')
            lit_model.final_model.student_model = student_module.model
            trainer_aec = pl.Trainer(
                max_epochs=config['aec']['lightning_module']['max_epochs'], 
                callbacks=[TQDMProgressBar(aec_data_module_len+1)], 
                accelerator='gpu', 
                devices=1,
                fast_dev_run=config['main']['fast_dev_run'],
                log_every_n_steps=20,
                # logger=wandb_logger_aec,
                enable_checkpointing=False,
                gradient_clip_val=1.0
            )
            trainer_aec.fit(lit_model, aec_data_module)
            # wandb.finish()
            clear_memory()
            
            # wandb_logger_cav = WandbLogger(name='Trainer_CAV', project='CD-Abusive2', id='cav', resume='allow')
            cav_model.autoencoder = lit_model.final_model.autoencoder
            trainer_cav = pl.Trainer(
                max_epochs=config['cav']['lightning_module']['max_epochs'], 
                callbacks=[TQDMProgressBar(cav_data_module_len+1)], 
                accelerator='gpu', 
                devices=1,
                fast_dev_run=config['main']['fast_dev_run'],
                log_every_n_steps=20,
                # logger=wandb_logger_cav,
                enable_checkpointing=False,
                gradient_clip_val=1.0
            )
            trainer_cav.fit(cav_model, cav_data_module)
            # cav_model.run()
            
            # wandb.finish()
            # clear_memory()
            
            new_cav = cav_model.get_cav()
            
            student_module.update_cav(new_cav)
            
        if N % proto_update_freq == 0:
            student_module.initialize_prototypes(student_data_module)
            

        # wandb_logger_student = WandbLogger(name='Trainer_Student', project='CD-Abusive2', id='stmod', resume='allow')
        student_module.aec = cav_model.autoencoder
        # student_module.output_model = cav_model.output_model
        trainer_student = pl.Trainer(
            precision=16,
            max_epochs=config['stmod']['lightning_module']['max_epochs'], 
            callbacks=[TQDMProgressBar(student_data_module_len+1)], 
            accelerator='gpu', 
            devices=1,
            fast_dev_run=config['main']['fast_dev_run'],
            # logger=wandb_logger_student,
            enable_checkpointing=False,
            gradient_clip_val=1.0
        )
        trainer_student.fit(student_module, student_data_module)
        # wandb.finish()
        clear_memory()

        if N % proto_update_freq == 0:
            save_checkpoint(lit_model, N, f"{AEC_CHECKPOINT_PATH}_mepoch_{N}.pth")
            # save_checkpoint(cav_model, N, f"{CAV_CHECKPOINT_PATH}_mepoch_{N}.pth")
            save_checkpoint(student_module, N, f"{STMOD_CHECKPOINT_PATH}_mepoch_{N}.pth")

        print('\n\n' + '-'*50)
        print(f"Epoch {N} completed ")
        print('-'*50 + '\n\n')
        
        N += 1
        
    lit_model.plot_losses()
    student_module.plot_losses()
    # cav_model.plot_losses()

if __name__ == '__main__':
    concept_distillation()
