import torch
import torchtext
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
#from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from special_vars import CLS,PAD,MASK,SEP,UNK,CLS_ID,PAD_ID,MASK_ID,SEP_ID,UNK_ID


def dataframe_from_csv(filepath):
    return pd.read_csv(filepath)
def build_vocab(tokenizer, text_series):
    '''
    inputs:
        tokenizer: tokenizer
        text_series: pandas Series of texts, or can be a list of sentences
    outputs:
        Vocab object
    '''
    text_iter = iter(text_series)
    def yield_token(text_iter):
        for text in text_iter:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(
        yield_token(text_iter),
        specials = [CLS,PAD,MASK,SEP,UNK]
    )
    vocab.set_default_index(vocab[UNK])
    return vocab
def tokenize_sentence(tokenizer, sentence):
    '''
    inputs:
        tokenizer: tokenizer
        sentence: single sentence (string)
    outputs:
        list of tokens (words or subwords): ['','',...]
    '''
    return tokenizer(sentence)
def mask_sequence(sequence, mask_percentage = 0.15):
    '''
    inputs:
        sequence: list of tokens (without special tokens)
        mask_percentage: how much tokens are masked out
    outputs: 
        original sequence, masked sequence
    '''
    sequence_len = len(sequence)
    mask_len = round(sequence_len*mask_percentage)
    mask_indices = random.choices(range(sequence_len), k = mask_len)
    masked_sequence = sequence.copy()
    for i in mask_indices:
        masked_sequence[i] = MASK
    return sequence, masked_sequence

def get_nsp_dataframe(text_series):
    '''
    inputs:
        text_series: pandas series of text (text is a paragraph that includes some sentences)
    outputs:
        nsp dataframe with three columns: first_sentence, second_sentence, is_next
    '''
    first_sentences = []
    second_sentences = []
    is_next = []
    
    for text in tqdm(text_series):
        text = text.strip()
        if '.' in text[:-1]:
            first_sents, second_sents, is_next_lbls = get_nsp_pair(text)
            first_sentences += first_sents
            second_sentences += second_sents
            is_next += is_next_lbls
    return pd.DataFrame({
        'first_sentence' : first_sentences,
        'second_sentence' : second_sentences,
        'is_next' : is_next
    })
def get_nsp_pair(text):
    '''
    inputs:
        text: a paragraph includes some sentences
    ouput:
        (list of first sentence, list of second sentence, isNext = list of 1's or 0's)
    '''
    sentences = text.split('.')
    if len(sentences[-1]) == 0:
        sentences.pop()
    num_sentences = len(sentences)

    if num_sentences == 2:
        true_nsp_pair = sentences
        false_nsp_pair = sentences.copy()
        false_nsp_pair.reverse()
        return (
            true_nsp_pair,
            false_nsp_pair,
            [1,0]
        )

    #get true nsp pairs:
    true_first_sentences = [sentences[i] for i in range(num_sentences-1)]
    true_second_sentences = [sentences[i] for i in range(1, num_sentences)]
    true_is_next = [1 for _ in range(num_sentences-1)]

    #get false nsp pairs:
    #---ensures that pair sentence_a, sentence_b is not next
    sentence_a_indices = np.random.permutation(num_sentences)
    sentence_b_indices = np.random.permutation(num_sentences)
    while np.any(sentence_b_indices==sentence_a_indices) or np.any(sentence_b_indices==sentence_a_indices+1):
        sentence_b_indices = np.random.permutation(num_sentences)
    false_first_sentences = [sentences[i] for i in sentence_a_indices[:-1]]
    false_second_sentences = [sentences[i] for i in sentence_b_indices[:-1]]
    false_is_next = [0 for _ in range(num_sentences-1)]
    
    #concate 2 lists
    first_sentences = true_first_sentences + false_first_sentences
    second_sentences = true_second_sentences + false_second_sentences
    is_next = true_is_next + false_is_next

    return first_sentences, second_sentences, is_next

def insert_special_tokens(sequence, max_len, insert_CLS = True, insert_PAD = True):
    '''
    inserts CLS token at first, PAD tokens at the end of sequence to reach max_len
    inputs:
        sequence: a list of tokens (['I', 'am',...])
        max_len: maximum length to padding to
    outputs:
        list of tokens (with the length depend on inserting behavior)
    '''
    if len(sequence) >= max_len:
        if insert_CLS:
            del sequence[max_len-1:]
        else:
            del sequence[max_len:]
    if insert_CLS:
        sequence.insert(0,CLS)
    sequence_len = len(sequence)
    if len(sequence) < max_len and insert_PAD == True:
        sequence += [PAD]*(max_len-sequence_len)
    return sequence
    
def nsp_accuracy(nsp_target, nsp_outputs, threshold = 0.5):
    '''
    inputs: 
        nsp_target, nsp_outputs: 1-D array-like
    outputs:
        accuracy score
    '''
    acc = accuracy_score(
        nsp_target.cpu(), 
        torch.where(nsp_outputs >= threshold, 1, 0).cpu()
    )
    #return torch.tensor(acc, device = ground_truths.device)
    return acc
def mlm_accuracy(org_sequences, masked_sequences, mlm_outputs):
    '''
    inputs: 
        org_sequences, masked_sequences: tensor of shape (batch_size, seq_len)
        mlm_outputs: tensor of shape (batch_size, seq_len, vocab_size)
    outputs:
        accuracy score
    '''
    pred_sequences = torch.argmax(mlm_outputs, dim = -1)
    masked_indices = org_sequences != masked_sequences

    ground_truths = torch.where(masked_indices, org_sequences, -1).view(-1)
    ground_truths = ground_truths[ground_truths > -1]

    predictions = torch.where(masked_indices, pred_sequences, -1).view(-1)
    predictions = predictions[predictions > -1]

    acc = accuracy_score(ground_truths.cpu(), predictions.cpu())
    return acc

