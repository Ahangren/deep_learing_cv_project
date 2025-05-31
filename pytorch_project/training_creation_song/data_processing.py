import torch
import random
import pandas as pd
import zipfile


def load_data_jay_lyrics(path):
    with open(path,encoding='utf-8') as f:
        corpus_chars=f.read()

    corpus_chars=corpus_chars.replace('\n',' ').replace('\t',' ')
    corpus_chars=corpus_chars[:10000]
    print(corpus_chars[:10])

    idx_to_char=list(set(corpus_chars))
    char_to_idx=dict([(char,i) for i,char in enumerate(idx_to_char)])
    vocab_size=len(char_to_idx)
    print(vocab_size)

    corpus_indices=[char_to_idx[char] for char in corpus_chars]
    sample=corpus_indices[:20]
    print('char',''.join([idx_to_char[idx] for idx in sample]))
    print('indices',sample)
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

corpus_indices,char_to_idx,idx_to_char,vocab_size=load_data_jay_lyrics('./data/jaychou_lyrics.txt')


def data_iter_random(corpus_indices,batch_size,num_steps,device=None):
    num_examples=(len(corpus_indices)-1)//num_steps
    epoch_size=num_examples//batch_size
    example_indices=list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos:pos+num_steps]

    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        i=i*batch_size
        batch_indices=example_indices[i:i+batch_size]
        x=[_data(j*num_steps) for j in batch_indices]
        y=[_data(j*num_steps+1) for j in batch_indices]
        yield torch.tensor(x,dtype=torch.float32,device=device),torch.tensor(y,dtype=torch.float32,device=device)

def data_iter_consecutive(corpus_indices,batch_size,num_steps,device=None):
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices=torch.tensor(corpus_indices,dtype=torch.float32,device=device)
    data_len=len(corpus_indices)
    batch_len=data_len//batch_size
    indices=corpus_indices[0:batch_size*batch_len].view(batch_size,batch_len)
    epoch_size=(batch_len-1)//num_steps
    for i in range(epoch_size):
        i=i*num_steps
        x=indices[:,i:i+num_steps]
        y=indices[:,i+1:i+num_steps+1]
        yield x,y


if __name__ == '__main__':
    myseq=[i for i in range(30)]
    for x,y in load_iter_consecutive(myseq,batch_size=2,num_steps=6):
        print('x: ',x,'\ny: ',y)
