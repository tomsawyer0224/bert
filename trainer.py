import torch
import torch.nn as nn
import os
from tqdm import tqdm
import pandas as pd

import utils

class Trainer(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, num_epochs = 3, root_dir = '.'):
        super().__init__()
        self.num_epochs = num_epochs
        self.root_dir = root_dir
    def train(self, model, train_loader, val_loader = None, checkpoint = None):
        model.to(self.device)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
        for i in range(self.num_epochs):
            training_epoch_outputs = []
            with tqdm(total = len(train_loader)) as training_pbar:
                training_pbar.set_description(f'training on epoch {i+1}/{self.num_epochs}')
                for training_batch in train_loader:
                    if isinstance(training_batch, list):
                        training_batch = [tb.to(self.device) for tb in training_batch]
                    else:
                        training_batch = training_batch.to(self.device)
                    training_step_outputs = model.training_step(training_batch)

                    training_pbar.set_postfix(training_step_outputs)

                    training_epoch_outputs.append(training_step_outputs)

                    training_pbar.update()
                training_pbar.set_postfix(model.on_train_epoch_end(training_epoch_outputs))
            if val_loader:
                val_epoch_outputs = []
                with tqdm(total = len(val_loader)) as val_pbar:
                    val_pbar.set_description(f'validating on epoch {i+1}/{self.num_epochs}')
                    for val_batch in val_loader:
                        if isinstance(val_batch, list):
                            val_batch = [vb.to(self.device) for vb in val_batch]
                        else:
                            val_batch = val_batch.to(self.device)
                        val_step_outputs = model.validation_step(val_batch)

                        val_epoch_outputs.append(val_step_outputs)
                        
                        val_pbar.set_postfix(val_step_outputs)
                        
                        val_pbar.update()
                    val_pbar.set_postfix(model.on_val_epoch_end(val_epoch_outputs))
        model_name = model.__class__.__name__
        checkpoint_name = model_name + f'_at_epoch_{self.num_epochs}' + '.pt'
        os.makedirs(
            os.path.join(self.root_dir, 'checkpoint'),
            exist_ok = True
        )
        checkpoint_path = os.path.join(self.root_dir, 'checkpoint', checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
    
