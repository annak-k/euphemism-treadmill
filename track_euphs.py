import numpy as np
from sklearn.decomposition import PCA
import pickle
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
plt.switch_backend('agg')

path = "/hal9000/datasets/wordembeddings/historical_sgns_all_english/"
# path = "/hal9000/datasets/wordembeddings/eng-fiction-all_sgns/"

def words2embeddings(words, years):
    embeds = {y:None for y in years}
    for y in years:
        embeds[y] = {w:None for w in words}
        with open(path + str(y) + "-vocab.pkl", "rb") as v_file:
            v = np.array(pickle.load(v_file))
            for w in words:
                index = np.where(np.in1d(v, w))[0] # find indices of specified words
                word_embedding = np.load(path + str(y) + "-w.npy")[index]
                if word_embedding.shape != 0:
                    embeds[y][w] = word_embedding[0]
                else:
                    embeds[y][w] = None
    return embeds

p_r_embeddings = words2embeddings(["polite", "rude"], [1990])[1990]
polite_emb = p_r_embeddings["polite"]
rude_emb = p_r_embeddings["rude"]

taboos = ["graveyard", "sex", "trash", "dead", "toilet", "naked", "old", "fat", "drunk"]
euphemisms = ["cemetery", "coitus", "landfill", "late", "lavatory", "nude", "elderly", "overweight", "intoxicated", "bathroom"]
probes = taboos + euphemisms

years = range(1900, 2000, 10)
data = np.zeros((len(probes),len(years)))

probe_embs = words2embeddings(probes, years)
for i in range(len(probes)):
    data[i] = np.zeros(len(years))
    for j in range(len(years)):
        # data[i][j] = np.subtract(cosine(probe_embs[years[j]][probes[i]], polite_emb), cosine(probe_embs[years[j]][probes[i]], rude_emb))
        data[i][j] = np.divide(cosine(probe_embs[years[j]][probes[i]], polite_emb), cosine(probe_embs[years[j]][probes[i]], rude_emb))

for i in (4,13,18):
    # if probes[i] in taboos:
    #     c = "red"
    # elif probes[i] in euphemisms:
    #     c = "green"
    plt.plot(years, data[i], label=probes[i])
plt.legend()
plt.ylabel("dist(polite) / dist(rude)")
plt.xlabel("years")
plt.savefig("hist_p_r_toilet_ratio.png")
plt.close()

for i in (6,15):
    if probes[i] in taboos:
        c = "red"
    elif probes[i] in euphemisms:
        c = "green"
    plt.plot(years, data[i], label=probes[i], color=c)
plt.legend()
plt.ylabel("dist(polite) - dist(rude)")
plt.xlabel("years")
plt.savefig("hist_p_r_old_ratio.png")
plt.close()
