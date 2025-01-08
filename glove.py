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

# [('tokyo', tensor(0.)), ('osaka', tensor(3.2893)), ('seoul', tensor(3.3802)), ('shanghai', tensor(3.6196)), ('japan', tensor(3.6599)), ('japanese', tensor(4.0788)), ('singapore', tensor(4.1160)), ('beijing', tensor(4.2423)), ('taipei', tensor(4.2453)), ('bangkok', tensor(4.2459))]

# beijing:china,tokyo:?
# w2-w1=w4-w3 china-beijing=?-tokyo

def analogy(a,b,c,n=5,filter_given=True):
    print("\n[%s : %s :: %s : ?]"%(a,b,c))
    #w2-w1+w3=w4
    closet_word = sim_10(get_word_vector(b)-get_word_vector(a)+get_word_vector(c),n=n)
    if filter_given:
        closet_word = [w for w in closet_word if w[0] not in [a,b,c]]
    return closet_word
# print(analogy("beijing","china","tokyo"))

# [('japan', tensor(2.7869)), ('japanese', tensor(3.6377)), ('singapore', tensor(3.9106)), ('shanghai', tensor(4.0189))]