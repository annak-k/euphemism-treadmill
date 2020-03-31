import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.switch_backend('agg')

path = "C:\\Users\\anna\\histwords"

def load_embeddings(years):
    """ Retrieve the word embeddings for a given set of years,
    and the corresponding vocabulary and parts of speech """

    embeddings  = {}
    vocabs = {}
    majority_pos = {}
    for y in years:
        embs = np.load("{}\\historical_sgns_all_english\\{}-w.npy".format(path, y))
        with open("{}\\historical_sgns_all_english\\{}-vocab.pkl".format(path, y), "rb") as v_file:
            vocab = list(pickle.load(v_file))
            vocabs[y] = vocab
            embeddings[y] = np.zeros((len(vocab), embs[0].shape[0]))
            for i, _ in enumerate(vocab):
                embeddings[y][i] = embs[i]
        
        with open("{}\\pos\\{}-pos.pkl".format(path, y), "rb") as pos_file:
            majority_pos[y] = dict(pickle.load(pos_file))

    print("**word embeddings, vocabulary, and parts of speech loaded**")
    return embeddings, vocabs, majority_pos

def load_taboos():
    """ Load the taboos from a csv file """

    taboos = pd.read_csv("taboos.csv", sep="\t", converters={"preceding contexts": lambda x: x.split(","), 
                                                           "following contexts": lambda x: x.split(",")},
                        header=0)

    return taboos

def get_constrained_nearest_neighbours(taboo_word, taboo_pos, embeddings, vocab, pos, k=10):
    """ Get the k nearest neighbours (by cosine similarity) to a given taboo word,
    looking only at the words which have the same part of speech as the taboo 
    word, and not including the taboo word itself. Return the list of 
    neighbours (by index in embeddings list) in order of decreasing similarity."""

    similarities = cosine_similarity(embeddings, embeddings[vocab.index(taboo_word)].reshape(1, -1)).T[0]
    neighbours = [i for i in np.argsort(similarities) if taboo_word not in vocab[i] and vocab[i] in pos and pos[vocab[i]] == taboo_pos][-k:]
    neighbours = list(reversed(neighbours))
    return neighbours

def main():
    taboos = load_taboos()

    years = range(1900, 2000, 10) # HistWords range 1800-1990
    e, v, pos = load_embeddings(years)
    
    # positive and negative seed words (with some additions)
    pos_words = ["good", "nice", "excellent", "positive", "correct", "superior"]
    pos_words += ["moral", "polite", "respectful"]
    neg_words = ["bad", "nasty", "poor", "negative", "wrong", "inferior"]
    neg_words += ["immoral", "rude", "disrespectful"]

    # get averaged "positive" and "negative" embeddings
    pos_emb = np.mean([e[years[-1]][v[years[-1]].index(w)] for w in pos_words], axis=0).reshape(1, -1)
    neg_emb = np.mean([e[years[-1]][v[years[-1]].index(w)] for w in neg_words], axis=0).reshape(1, -1)

    word_lists = {} # keep track of all the words
    for _, taboo in taboos.iterrows():
        time_series_list = []
        query_words = [taboo["word"]]

        # retrieve potential euphisms at each timestep, 
        # and create a timeseries of polarity
        for year in years:
            for qword in query_words:
                pos_similarity = cosine_similarity(e[year][v[year].index(qword)].reshape(1, -1), pos_emb)[0,0]
                neg_similarity = cosine_similarity(e[year][v[year].index(qword)].reshape(1, -1), neg_emb)[0,0]

                time_series_list.append({"word": qword, "year": year, "polarity": pos_similarity - neg_similarity})

            nearest_neighbours = get_constrained_nearest_neighbours(taboo["word"], taboo["POS"], e[year], v[year], pos[year])
            nearest_neighbour_embeddings = e[year][nearest_neighbours]

            nn_pos_neg = cosine_similarity(nearest_neighbour_embeddings, pos_emb)[0] - cosine_similarity(nearest_neighbour_embeddings, neg_emb)[0]
            new_euphemism = np.argsort(nn_pos_neg)[::-1][0]
            
            time_series_list.append({"word": v[year][nearest_neighbours[new_euphemism]], "year": year, "polarity": nn_pos_neg[new_euphemism]})

            if v[year][nearest_neighbours[new_euphemism]] not in query_words:
                query_words.append(v[year][nearest_neighbours[new_euphemism]])
        
        time_series = pd.DataFrame(time_series_list)
        sns.lineplot(x="year", y="polarity", data=time_series, hue="word")
        plt.title("Tracking {}'s euphemisms".format(taboo["word"]))
        plt.savefig("track {}.png".format(taboo["word"]))
        plt.close()

        word_lists[taboo["word"]] = query_words

    with open("word_lists.pkl", "wb") as outfile:
        pickle.dump(word_lists, outfile)

if __name__ == "__main__":
    main()