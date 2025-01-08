import torch
import torchtext
from torchtext import vocab

gv = torchtext.vocab.GloVe(name='6B', dim=50, cache='.vector_cache')

# print(gv.stoi["tokyo"])

# print(gv.vectors[1363])

# print(gv.itos[1363])

def get_word_vector(word):
    return gv.vectors[gv.stoi[word]]

# print(get_word_vector("tokyo"))

def sim_10(word,n=10):
    all_dists = [(gv.itos[i],torch.dist(word,w)) for i,w in enumerate(gv.vectors)]
    return sorted(all_dists,key=lambda t:t[1])[:n]

print(sim_10(get_word_vector("tokyo")))


# beijing:china,tokyo:?
# w2-w1=w4-w3 china-beijing=?-tokyo

def analogy(a,b,c,n=5,filter_given=True):
    print("\n[%s : %s :: %s : ?]"%(a,b,c))
    #w2-w1+w3=w4
    closet_word = sim_10(get_word_vector(b)-get_word_vector(a)+get_word_vector(c),n=n)
    if filter_given:
        closet_word = [w for w in closet_word if w[0] not in [a,b,c]]
    return closet_word
# print(analogy("china","beijing","tokyo"))