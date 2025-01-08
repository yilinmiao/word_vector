from collections import Counter

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity


sentences = [
    "i like dog very much!",
    "i like cat a lot!",
]


def preprocess(sentences, window_size=2):
    # word2idx = {}
    # idx2word = {}
    # for sentence in sentences:
    #     for word in sentence.split():
    #         if word not in word2idx:
    #             word2idx[word] = len(word2idx)
    #             idx2word[len(idx2word)] = word
    # return word2idx, idx2word
    vocab = Counter()
    for sentence in sentences:
        vocab.update(sentence.split())
    print(vocab)
    vocab = {word: i for i, word in enumerate(vocab)}
    print(vocab)
    idx2word = {i: word for i, word in enumerate(vocab)}
    print(idx2word)

    data = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            left = max(0, i - window_size)
            right = min(len(words), i + window_size + 1)
            context = words[left:i] + words[i + 1:right]
        # for i in range(window_size, len(words) - window_size):
        #     context = [words[j] for j in range(2*window_size)]
        #     target = words[i]
        if context:
            target = words[i]
            data.append((context, target))
        # print(data)
    return vocab, idx2word, data

class CBOWDataset:
    def __init__(self,data,vocab, max_context_len=4):
        self.data=data
        self.vocab=vocab
        self.max_context_len = max(len(context) for context, _ in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        context,target = self.data[item]
        context_ids = [self.vocab[word] for word in context]

        if len(context_ids) < self.max_context_len:
            context_ids += [self.vocab['<PAD>']] * (self.max_context_len - len(context_ids))
        else:
            context_ids = context_ids[:self.max_context_len]

        target_id = self.vocab[target]
        return torch.tensor(context_ids), torch.tensor(target_id)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        # self.activation_function1 = nn.ReLU()

    def forward(self, x):
        embeds = self.embeddings(x).mean(dim=1)
        out = self.linear(embeds)
        return out

def train(model, dataloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

def get_most_similar_words(model,vocab,idx_word,target_word,top_k=5):
    target_word_id = vocab[target_word]
    target_word_embedding = model.embeddings(torch.tensor(target_word_id)).detach().numpy()
    word_embedding = model.embeddings.weight.detach().numpy()
    # similarity = torch.matmul(model.embeddings.weight,word_embedding)
    # torch.nn.functional.cosine_similarity
    similarities = cosine_similarity([target_word_embedding],word_embedding)[0]
    # _,topk = torch.topk(similarity,k)
    # similar_words = [idx2word[idx.item()] for idx in topk]
    # return similar_words
    most_similar_indices = similarities.argsort()[-top_k - 1:-1][::-1]
    most_similar_words = [(idx_word[idx], similarities[idx]) for idx in most_similar_indices]
    return most_similar_words

if __name__ == '__main__':
    sentences = [
        "i like dog very much!",
        "i like cat a lot!",
    ]

    vocab, idx2word, data = preprocess(sentences)
    print(data)
    data = CBOWDataset(data,vocab)
    dataloader = DataLoader(data, batch_size=2, shuffle=True)

    model = CBOW(len(vocab), 10)

    train(model, dataloader)

    target_word = 'dog'
    most_similar_words = get_most_similar_words(model, vocab, idx2word, target_word, top_k=3)
    print(f"Target Word: {target_word}, Most related words: {most_similar_words}")