import torch
import torch.nn as nn
import torch.nn.functional as F


class NERTagger(nn.Module):
    def __init__(self, n_vocab, embed_size, hidden_dim, BIOset_size):
        super(NERTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(n_vocab, embed_size, hidden_dim, BIOset_size)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_size, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2BIO = nn.Linear(hidden_dim, BIOset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        BIO_space = self.hidden2BIO(lstm_out.view(len(sentence), -1))
        BIO_scores = F.log_softmax(BIO_space, dim=1)
        return BIO_scores