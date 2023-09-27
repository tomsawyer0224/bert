import torch
import torchtext
import pandas as pd
#import random
#import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torchtext.vocab import vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from special_vars import CLS,PAD,MASK,SEP,UNK,CLS_ID,PAD_ID,MASK_ID,SEP_ID,UNK_ID,MASK_PERCENTAGE
import utils

class IMDBpretraining(Dataset):
    '''
    pre-training dataset: nsp_pair (fixed length, tokenized, masked, inserted CLS, PAD), label
    '''
    def __init__(self, csv_file, text_column_name, tokenizer, vocab):
        super().__init__()
        #self.text_column_name = text_column_name
        self.nsp_df = utils.get_nsp_dataframe(
            utils.dataframe_from_csv(csv_file)[text_column_name]
        )
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.optimal_length = min(255, self.find_optimal_length(percentile = 0.75))
    def __len__(self):
        return self.nsp_df.shape[0]
    def __getitem__(self, idx):
        
        row = self.nsp_df.iloc[idx] # first_sentence, second_sentence, is_next

        first_sentence = row['first_sentence']
        second_sentence = row['second_sentence']
        is_next = row['is_next']

        #tokeninze
        first_sequence = utils.tokenize_sentence(self.tokenizer, first_sentence)
        second_sequence = utils.tokenize_sentence(self.tokenizer, second_sentence)
        #mask
        org_first_sequence, masked_first_sequence = utils.mask_sequence(
            sequence = first_sequence, mask_percentage = MASK_PERCENTAGE
        )
        org_second_sequence, masked_second_sequence = utils.mask_sequence(
            sequence = second_sequence, mask_percentage = MASK_PERCENTAGE
        )
        #original nsp sequence
        org_nsp_sequence = utils.insert_special_tokens(
            sequence = org_first_sequence,
            max_len = self.optimal_length,
            insert_CLS = True,
            insert_PAD = True
        )+[SEP]+utils.insert_special_tokens(
            sequence = org_second_sequence,
            max_len = self.optimal_length,
            insert_CLS = False,
            insert_PAD = True
        )
        #processed (masked, inserted CLS, PAD) nsp sequence
        masked_nsp_sequence = utils.insert_special_tokens(
            sequence = masked_first_sequence,
            max_len = self.optimal_length,
            insert_CLS = True,
            insert_PAD = True
        )+[SEP]+utils.insert_special_tokens(
            sequence = masked_second_sequence,
            max_len = self.optimal_length,
            insert_CLS = False,
            insert_PAD = True
        )
        #segment_ids
        segment_ids = [0]*(self.optimal_length + 1) + [1]*self.optimal_length
        return (
            torch.tensor(
                self.vocab.lookup_indices(org_nsp_sequence), dtype = torch.long
            ),
            torch.tensor(
                self.vocab.lookup_indices(masked_nsp_sequence), dtype = torch.long
            ),
            torch.tensor(segment_ids, dtype = torch.long),
            #torch.tensor(is_next, dtype = torch.long)
            torch.tensor(is_next, dtype = torch.float)
        )

    def find_optimal_length(self, percentile = 0.75):
        '''
        get optimal length of first_sentence, second_sentence
        inputs:
            text_series: Pandas Series of texts
        outputs:
            75% precentile of first_sentence, second_sentence
        '''
        length_serie = self.nsp_df['first_sentence'].drop_duplicates().map(
            lambda s: len(s.split())
        )
        optimal_length = round(length_serie.quantile(percentile))
        return optimal_length

class IMDBforCLS(Dataset):
    def __init__(self, csv_file, text_column_name, label_column_name, class_names, tokenizer, vocab):
        super().__init__()
        raw_df = utils.dataframe_from_csv(csv_file)
        raw_df['label_id'] = raw_df[label_column_name].map(
            lambda cls_name: class_names.index(cls_name)
        )
        self.cls_df = raw_df
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.optimal_length = min(511, self.find_optimal_length(percentile = 0.75))
    def find_optimal_length(self, percentile = 0.75):
        length_df = self.cls_df[self.text_column_name].map(
            lambda text: len(self.tokenizer(text))
        )
        optimal_length = round(length_df.quantile(percentile))
        return optimal_length
    def __len__(self):
        return self.cls_df.shape[0]
    def __getitem__(self, idx):
        row = self.cls_df.iloc[idx]
        text = row[self.text_column_name]
        label_id = row['label_id']
        sequence = self.tokenizer(text)
        # insert CLS, PAD
        sequence = utils.insert_special_tokens(
            sequence = sequence,
            max_len = self.optimal_length,
            insert_CLS = True,
            insert_PAD = True
        )
        return (
            torch.tensor(self.vocab.lookup_indices(sequence), dtype = torch.long),
            torch.tensor(label_id, dtype = torch.long)
        )

class IMDBforMLM(Dataset):
    def __init__(self, csv_file, text_column_name, label_column_name, class_names, tokenizer, vocab):
        super().__init__()
        raw_df = utils.dataframe_from_csv(csv_file)
        self.mlm_df = raw_df
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.optimal_length = min(512, self.find_optimal_length(percentile = 0.75))
    def find_optimal_length(self, percentile = 0.75):
        length_df = self.mlm_df[self.text_column_name].map(
            lambda text: len(self.tokenizer(text))
        )
        optimal_length = round(length_df.quantile(percentile))
        return optimal_length
    def __len__(self):
        return self.mlm_df.shape[0]
    def __getitem__(self, idx):
        row = self.mlm_df.iloc[idx]
        text = row[self.text_column_name]
        sequence = self.tokenizer(text)

        org_sequence, masked_sequence = utils.mask_sequence(
            sequence = sequence,
            mask_percentage = MASK_PERCENTAGE
        )
        #padding
        org_sequence = utils.insert_special_tokens(
            sequence = org_sequence, 
            max_len = self.optimal_length, 
            insert_CLS = False, 
            insert_PAD = True)
        masked_sequence = utils.insert_special_tokens(
            sequence = masked_sequence, 
            max_len = self.optimal_length, 
            insert_CLS = False, 
            insert_PAD = True)
        return (
            torch.tensor(self.vocab.lookup_indices(org_sequence), dtype = torch.long),
            torch.tensor(self.vocab.lookup_indices(masked_sequence), dtype = torch.long)
        )

