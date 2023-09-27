import torch
import torch.nn as nn
#import torchtext
from special_vars import SEP_ID, PAD_ID
import utils
#from torch.nn.functional import multi_margin_loss


class JointEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size, 
            embedding_dim = embed_dim
        )
        self.segment_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim
        )
        self.positional_emb = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim
        )
        self.norm = nn.LayerNorm(normalized_shape = embed_dim)
    def forward(self, input_tensor, segment_tensor = None):
        '''
        inputs: 
            input_tensor: tensor of shape (batch_size, seq_len)
            segment_tensor: tensor of shape(batch_size, seq_len)
                         seq = seq A + [SEP] + seq B -> seg_id of A = 0, seg_id of B = 1
        outputs:
            tensor of shape (batch_size, seq_len, embed_dim)
        '''
        seq_len = input_tensor.size(-1) # sequence length
        indices = torch.arange(seq_len).long().expand_as(input_tensor) # indices from 0 to seq_len-1
        
        position_tensor = torch.arange(start=0, end=seq_len, step=1)#.long()
        position_tensor = position_tensor.expand_as(input_tensor).type_as(input_tensor)

        if segment_tensor is None:
            segment_tensor = torch.zeros_like(input_tensor).long()
        
        '''
        _, index_SEP = (input_tensor==SEP_ID).nonzero(as_tuple=True)
        index_SEP = index_SEP.unsqueeze(-1)
        segment_tensor = torch.where(indices<=index_SEP, 0, 1).long()
        '''
        out = self.token_embedding(input_tensor)+self.positional_emb(position_tensor)+self.segment_embedding(segment_tensor)
        out = self.norm(out)
        return out

class BERT(nn.Module):
    '''
    model BERT includes only encoders
    '''
    def __init__(self, 
        vocab_size,
        num_layers = 8, 
        d_model = 512,
        nhead = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.embedding = JointEmbedding(
            vocab_size = vocab_size,
            embed_dim = d_model
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers
        )
        #self.prediction = nn.Linear(in_features = d_model, out_features = vocab_size)
        #self.classification = nn.Linear(in_features= d_model, out_features = 1)
    def forward(self, input_tensor, segment_tensor = None):
        '''
        inputs:
            input_tensor: tensor of shape (batch_size, seq_len)
            segment_tensor:
                tensor of shape(batch_size, seq_len)
                seq = seq A + [SEP] + seq B -> seg_id of A = 0, seg_id of B = 1
        outputs:
            logits: tensor of shape (batch_size, seq_len, embed_dim)
        '''
        out = self.embedding(input_tensor = input_tensor, segment_tensor = segment_tensor)

        padding_mask = input_tensor == PAD_ID
        out = self.encoder(src = out, src_key_padding_mask = padding_mask)
        # return encoder_output
        return out
class BertForPretraining(nn.Module):
    def __init__(self, backbone, nsp_factor = 0.7, mlm_factor = 0.3):
        super().__init__()
        self.backbone = backbone
        self.nsp_factor = nsp_factor
        self.mlm_factor = mlm_factor
        self.nsp_cls_layer = nn.Linear(
            in_features = self.backbone.d_model,
            out_features = 1
        )
        self.sigmoid_layer = nn.Sigmoid()
        self.mlm_layer = nn.Linear(
            in_features = self.backbone.d_model,
            out_features = self.backbone.vocab_size
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = 0.0001
        )
        self.nsp_loss_fn = nn.BCELoss()
        self.mlm_loss_fn = nn.CrossEntropyLoss()
    def forward(self, input_tensor, segment_tensor):
        '''
        inputs:
            input_tensor: tensor of shape (batch_size, seq_len)
        outputs:
            nsp_outputs (probabilities): tensor of shape (batch_size, seq_len)
            mlm_outputs (logits): tensor of shape (batch_size, seq_len, vocab_size)
        '''
        backbone_outputs = self.backbone(input_tensor, segment_tensor)

        #nsp outputs
        nsp_first_word = backbone_outputs[:,0,:] # CLS tokens 
        nsp_outputs = self.nsp_cls_layer(nsp_first_word)
        nsp_outputs = self.sigmoid_layer(nsp_outputs).squeeze()
        #mlm outputs
        mlm_outputs = self.mlm_layer(backbone_outputs)
        return nsp_outputs, mlm_outputs
    def configure_optimizers(self):
        pass
    def get_optimizers(self):
        return self.optimizer
    def training_step(self, batch):
        self.train()
        optimizer = self.get_optimizers()
        optimizer.zero_grad()
        #org_nsp_sequence, masked_nsp_sequence, segment_ids, is_next = batch
        #mlm_target = org_nsp_sequence, input_tensor = masked_nsp_sequence
        #segment_tensor = segment_ids, cls_target
        mlm_target, input_tensor, segment_tensor, cls_target = batch

        #pass original_nsp_sequence (without masking) to get nsp_outputs
        nsp_outputs, _ = self(
            input_tensor = mlm_target, segment_tensor = segment_tensor
        )
        #pass masked nsp sequence to get mlm_outputs
        _, mlm_outputs = self(
            input_tensor = input_tensor, segment_tensor = segment_tensor
        )

        '''
        nsp_outputs, mlm_outputs = self(
            input_tensor = input_tensor, segment_tensor = segment_tensor
        )
        '''
        nsp_loss = self.nsp_loss_fn(nsp_outputs, cls_target)
        mlm_loss = self.mlm_loss_fn(
            mlm_outputs.transpose(-2,-1), mlm_target
        )
        loss = self.nsp_factor*nsp_loss + self.mlm_factor*mlm_loss
        loss.backward()
        optimizer.step()
        return {
            'train_step_loss': loss.detach().item(),
            'train_step_nsp_loss': nsp_loss.detach().item(),
            'train_step_mlm_loss': mlm_loss.detach().item(),
            'train_step_nsp_acc' : utils.nsp_accuracy(cls_target, nsp_outputs.detach()),
            'train_step_mlm_acc' : utils.mlm_accuracy(mlm_target, input_tensor, mlm_outputs.detach())
        }
    def on_train_epoch_end(self, training_step_outputs):
        '''
        inputs:
            training_step_outputs: list of training_step outputs
        outputs:
            average over all steps
        '''
        training_epoch_dict = {
            k.replace('step', 'epoch'):[tso[k] for tso in training_step_outputs] for k in training_step_outputs[0].keys()
        }
        training_epoch_mean = {
            k:sum(training_epoch_dict[k])/len(training_epoch_dict[k]) for k in training_epoch_dict.keys()
        }
        """
        training_epoch_dict = {
            'train_epoch_loss' : [tso['train_step_loss'] for tso in training_step_outputs],
            'train_epoch_nsp_loss': [tso['train_step_nsp_loss'] for tso in training_step_outputs],
            'train_epoch_mlm_loss': [tso['train_step_mlm_loss'] for tso in training_step_outputs],
            'train_epoch_nsp_acc': [tso['train_step_nsp_acc'] for tso in training_step_outputs],
            'train_epoch_mlm_acc': [tso['train_step_mlm_acc'] for tso in training_step_outputs]
        }
        """
        return training_epoch_mean

    def validation_step(self, batch):
        mlm_target, input_tensor, segment_tensor, cls_target = batch
        with torch.no_grad():
            _, mlm_outputs = self(
                input_tensor = input_tensor, segment_tensor = segment_tensor
            )
            nsp_outputs, _ = self(
                input_tensor = mlm_target, segment_tensor = segment_tensor
            )
            '''
            nsp_outputs, mlm_outputs = self(
                input_tensor = input_tensor, segment_tensor = segment_tensor
            )
            '''
        nsp_loss = self.nsp_loss_fn(nsp_outputs, cls_target)
        mlm_loss = self.mlm_loss_fn(
            mlm_outputs.transpose(-2,-1), mlm_target
        )
        loss = self.nsp_factor*nsp_loss + self.mlm_factor*mlm_loss
        return {
            'val_loss': loss.item(),
            'val_nsp_loss': nsp_loss.item(),
            'val_mlm_loss': mlm_loss.item(),
            'val_nsp_acc' : utils.nsp_accuracy(cls_target, nsp_outputs),
            'val_mlm_acc' : utils.mlm_accuracy(mlm_target, input_tensor, mlm_outputs)
        }
    def on_val_epoch_end(self, val_step_outputs):
        val_epoch_dict = {
            k.replace('step', 'epoch'):[tso[k] for tso in val_step_outputs] for k in val_step_outputs[0].keys()
        }
        val_epoch_mean = {
            k:sum(val_epoch_dict[k])/len(val_epoch_dict[k]) for k in val_epoch_dict.keys()
        }
        return val_epoch_mean
# class bert for classification
class BertForCLS(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classification_layer = nn.Linear(
            in_features = self.backbone.d_model, out_features = num_classes
        )
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input_tensor, segment_tensor = None):
        '''
        inputs: 
            input_tensor: tensor of shape (batch_size, seq_len)
        outputs:
            probs: tensor of shape (batch_size, num_classes)
        '''
        out = self.backbone(input_tensor = input_tensor, segment_tensor = segment_tensor)
        cls_tokens = out[:,0,:]
        print(f'cls_tokens.shape: {cls_tokens.shape}')
        cls_scores = self.classification_layer(cls_tokens)
        cls_probs = self.softmax(cls_scores)
        return cls_scores, cls_probs
# class bert for masked language model
class BertForMLM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.prediction = nn.Linear(
            in_features = self.backbone.d_model, out_features = self.backbone.vocab_size
        )
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input_tensor, segment_tensor = None):
        out = self.backbone(input_tensor)
        token_scores = self.prediction(out)
        token_probs = self.softmax(token_scores)
        return token_scores, token_probs

