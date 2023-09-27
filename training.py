import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer

#from data.datasets import IMDBpretraining, IMDBforCLS, IMDBforMLM
from data import datasets
from special_vars import CLS,PAD,MASK,SEP,UNK,CLS_ID,PAD_ID,MASK_ID,SEP_ID,UNK_ID,MASK_PERCENTAGE
from model.bert import BERT, BertForCLS, BertForMLM, BertForPretraining
from trainer import Trainer
import utils

CSV_FILE = './data/IMDB Dataset.csv'
TRAIN_CSV_FILE = './data/small_imdb_train.csv'
VAL_CSV_FILE = './data/small_imdb_val.csv'
TEST_CSV_FILE = './data/small_imdb_test.csv'
TEXT_COLUMN_NAME = 'review'
LABEL_COLUMN_NAME = 'sentiment'
CLASS_NAMES = ['negative', 'positive']

#raw_imdb_df = utils.dataframe_from_csv(CSV_FILE)
raw_imdb_df = utils.dataframe_from_csv(TRAIN_CSV_FILE)
tokenizer = get_tokenizer('basic_english')
vocab = utils.build_vocab(tokenizer = tokenizer, text_series = raw_imdb_df[TEXT_COLUMN_NAME])
vocab_size = len(vocab)
'''
imdb_pretraining_dataset = datasets.IMDBpretraining(
    csv_file = CSV_FILE,
    text_column_name = TEXT_COLUMN_NAME,
    tokenizer = tokenizer,
    vocab = vocab
)

imdb_pretraining_subset = torch.utils.data.Subset(
    dataset = imdb_pretraining_dataset,
    indices = torch.randint(
        low=0, 
        high = len(imdb_pretraining_dataset),
        size = (20000,),
        generator = torch.Generator().manual_seed(42)
    ).tolist()
)

train_pre_ds, val_pre_ds, test_pre_ds = torch.utils.data.random_split(
    imdb_pretraining_subset, [0.8,0.1,0.1], torch.Generator().manual_seed(42)
)
'''
train_pre_ds = datasets.IMDBpretraining(
    csv_file = TRAIN_CSV_FILE,
    text_column_name = TEXT_COLUMN_NAME,
    tokenizer = tokenizer,
    vocab = vocab
)
val_pre_ds = datasets.IMDBpretraining(
    csv_file = VAL_CSV_FILE,
    text_column_name = TEXT_COLUMN_NAME,
    tokenizer = tokenizer,
    vocab = vocab
)
test_pre_ds = datasets.IMDBpretraining(
    csv_file = TEST_CSV_FILE,
    text_column_name = TEXT_COLUMN_NAME,
    tokenizer = tokenizer,
    vocab = vocab
)
batch_size = 32
train_pre_loader = DataLoader(
    dataset = train_pre_ds,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2
)
val_pre_loader = DataLoader(
    dataset = val_pre_ds,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2
)
test_pre_loader = DataLoader(
    dataset = test_pre_ds,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 2
)

backbone = BERT(
    vocab_size = vocab_size,
    num_layers = 8,
    d_model = 512,
    nhead = 8
)

bert_pretraining = BertForPretraining(
    backbone = backbone
)

trainer = Trainer(num_epochs = 10)
trainer.train(
    model = bert_pretraining,
    train_loader = test_pre_loader,
    val_loader = val_pre_loader
)
