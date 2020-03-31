import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.switch_backend('agg')

path = "C:\\Users\\anna\\histwords"

def load_frequencies():
    """ Retrieve the frequencies for all words at each decade """

    frequencies  = {}
    with open("{}\\freqs.pkl".format(path), "rb") as freq_file:
        frequencies = dict(pickle.load(freq_file, encoding="latin-1"))
    return frequencies

def load_taboos():
    """ Load the taboos from a csv file """

    taboos = pd.read_csv("taboos.csv", sep="\t", converters={"preceding contexts": lambda x: x.split(","), 
                                                           "following contexts": lambda x: x.split(",")},
                        header=0)
    return taboos

def main():
    taboos = load_taboos()

    years = range(1900, 2000, 10) # HistWords range 1800-1990
    freqs = load_frequencies(years)

    with open("word_lists.pkl", "rb") as word_file:
        words = dict(pickle.load(word_file))

    for _, taboo in taboos.iterrows():
        time_series_list = []
        query_words = words[taboo["word"]]
        
        for year in years:
            for qword in query_words:
                time_series_list.append({"word": qword, "year": year, "frequency": freqs[qword][year]})
        
        time_series = pd.DataFrame(time_series_list)
        sns.lineplot(x="year", y="frequency", data=time_series, hue="word")
        plt.title("Tracking {}'s euphemism frequencies".format(taboo["word"]))
        plt.savefig("frequency {}.png".format(taboo["word"]))
        plt.close()

if __name__ == "__main__":
    main()