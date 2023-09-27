import torchtext
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from model.bert import BERT, JointEmbedding, BertForCLS, BertForMLM
from special_vars import CLS,PAD,MASK,SEP,UNK,CLS_ID,PAD_ID,MASK_ID,SEP_ID,UNK_ID
import utils
from data import datasets

CSV_FILE = './data/IMDB Dataset.csv'
tokenizer = get_tokenizer('basic_english')
raw_imdb_df = utils.dataframe_from_csv(CSV_FILE)
print(raw_imdb_df)
TEXT_COLUMN_NAME = 'review'
LABEL_COLUMN_NAME = 'sentiment'
CLASS_NAMES = ['negative', 'positive']
vocab = utils.build_vocab(tokenizer = tokenizer, text_series = raw_imdb_df[TEXT_COLUMN_NAME])

print(vocab, len(vocab))
print(vocab.lookup_tokens(list(range(0,10))))
print(vocab.lookup_tokens(list(range(len(vocab)-10, len(vocab)))))

print(raw_imdb_df[TEXT_COLUMN_NAME][100])
print(vocab.lookup_indices(tokenizer(raw_imdb_df[TEXT_COLUMN_NAME][100])))

print("IMDB for pretraining ################################")
imdb_pretraining_dataset = datasets.IMDBpretraining(
    csv_file = CSV_FILE,
    text_column_name = TEXT_COLUMN_NAME,
    tokenizer = tokenizer,
    vocab = vocab
)

ds_iter = iter(imdb_pretraining_dataset)
for i in range(3):
    org_nsp_sequence, masked_nsp_sequence, segment_ids, is_next = next(ds_iter)
    print('len:', len(org_nsp_sequence), len(masked_nsp_sequence), len(segment_ids)) 
    print('org_nsp_sequence\n', org_nsp_sequence)
    print('masked_nsp_sequence\n', masked_nsp_sequence)
    print('segment_ids\n', segment_ids)
    print('nsp_label:', is_next)
    print("##################################")

print("IMDB for classification ################################")
imdb_cls_dataset = datasets.IMDBforCLS(
    csv_file = CSV_FILE, 
    text_column_name = TEXT_COLUMN_NAME,
    label_column_name = LABEL_COLUMN_NAME,
    class_names = CLASS_NAMES,
    tokenizer = tokenizer,
    vocab = vocab
)
print('optimal sequence length:', imdb_cls_dataset.optimal_length)
cls_ds_iter = iter(imdb_cls_dataset)
for i in range(3):
    sequence, label = next(cls_ds_iter)
    print(f'sequence: {len(sequence)}\n', sequence)
    print('label:', label)
    print("###################################")

print("IMDB for MLM ################################")
imdb_mlm_dataset = datasets.IMDBforMLM(
    csv_file = CSV_FILE, 
    text_column_name = TEXT_COLUMN_NAME,
    label_column_name = LABEL_COLUMN_NAME,
    class_names = CLASS_NAMES,
    tokenizer = tokenizer,
    vocab = vocab
)

mlm_ds_iter = iter(imdb_mlm_dataset)
for i in range(3):
    org_sequence, masked_sequence = next(mlm_ds_iter)
    print(f'org_sequence: {len(org_sequence)}\n', org_sequence)
    print(f'masked_sequence: {len(masked_sequence)}\n', masked_sequence)
    print("###################################")
"""
toy_sequence = ['1','2','3','4','5']
print(toy_sequence)
max_len = 7
utils.insert_special_tokens(toy_sequence,max_len,insert_CLS=True, insert_PAD=False)
print(toy_sequence)


df = utils.dataframe_from_csv('./data/IMDB Dataset.csv')
print(df)
text = df['review'][106]
sequence = utils.tokenize_sentence(tokenizer,text)
org_sequence, masked_sequence = utils.mask_sequence(sequence,0.5)
print(org_sequence)
print(masked_sequence)


toy_df = pd.DataFrame({
    'review': ['1.2.3','4.','5.6.','7.8','9'],
    'sentiment': [1,1,1,1,1]
})
print(toy_df)
nsp_df = utils.get_nsp_dataframe(toy_df['review'])
print(nsp_df.to_string())

df = utils.dataframe_from_csv('./data/IMDB Dataset.csv')
#print(df)
#s = df['review'][106]
#print(s)
nsp_df = utils.get_nsp_dataframe(df['review'])
print(nsp_df)
text = df['review'][58]
sentences = text.split('. ')
print(sentences)

f, s, i = utils.get_nsp_pair(text)
print(f)
print(s)
print(i)

vocab = utils.build_vocab(tokenizer, df['review'])
tokenized_text = tokenizer(text)
print(vocab(tokenized_text))
print(vocab['######'], vocab['film'])
print(vocab.lookup_tokens([100, 200]))
print(vocab.lookup_indices([CLS,PAD,MASK,SEP,UNK]))
print(CLS_ID,PAD_ID,MASK_ID,SEP_ID,UNK_ID)
print("################################")
tokens = utils.tokenize_sentence(tokenizer,text)
print(tokens)
org_tokens, masked_tokens = utils.mask_token(tokens,0.15)
print(org_tokens)
print(masked_tokens)

sent = sentences[1]
tokens, masked_tokens = utils.tokenize_and_mask_sentence(tokenizer = tokenizer, sentence = sent)
print(tokens)
print(masked_tokens)
"""
'''
vocab_size = 100
seq_len = 10
num_layers = 8
d_model = 512
nhead = 8
embed_dim=512
#num_classes = 2
batch_size = 4
model = BERT(
    vocab_size=vocab_size,
    num_layers=num_layers,
    d_model=d_model,
    nhead=nhead,
)
embed_layer = JointEmbedding(vocab_size=vocab_size,embed_dim=embed_dim)

input_tensor = torch.randint(low=5, high=100, size=(batch_size, seq_len))
indices = torch.arange(input_tensor.size(-1)).expand_as(input_tensor)
idx_SEP = torch.randint(low=1, high=seq_len-1, size = (batch_size,1))
input_tensor = torch.where(indices==idx_SEP, SEP_ID, input_tensor)
idx_PAD = torch.randint(low=seq_len-5, high = seq_len, size = (input_tensor.size(0),1))
input_tensor = torch.where(indices>=idx_PAD, PAD_ID, input_tensor)

print('input_tensor.shape:',input_tensor.shape)
print('input_tensor\n',input_tensor)

encoder_output = model(input_tensor,segment_tensor=None)
print('encoder_output.shape:',encoder_output.shape)
print(encoder_output)

#bert_ckpt = torch.save(model.state_dict(), 'checkpoint/bert.pt')

backbone = BERT(
    vocab_size=vocab_size,
    num_layers=num_layers,
    d_model=d_model,
    nhead=nhead,
)
backbone.load_state_dict(torch.load('checkpoint/bert.pt'))

print("#####################################")
cls_model = BertForCLS(backbone = backbone, num_classes = 2)
cls_probs = cls_model(input_tensor)
print(f'cls_probs.shape: {cls_probs.shape}\n', cls_probs)

print("######################################")
mlm_model = BertForMLM(backbone = backbone)
token_probs = mlm_model(input_tensor)
print(f'token_probs.shape; {token_probs.shape}\n', token_probs)

token_preds = torch.argmax(token_probs, dim = -1)
print(f'token_preds.shape: {token_preds.shape}\n', token_preds)
'''

