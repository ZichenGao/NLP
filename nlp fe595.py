import numpy as np
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter


def Merge(txt, csv):
    names1 = np.array([])
    purpose1 = np.array([])
    for i in txt:
        df = pd.read_csv(i, delimiter="\t", header=None)
        nrow = len(df)
        for j in range(nrow):
            str = df.iloc[j, 0]
            company_name = str.split(':', 2)
            company_purpose = company_name[2]
            company_name = company_name[1].split('---')[0]

            names1 = np.append(names1,company_name)
            purpose1 = np.append(purpose1,company_purpose)

    names2 = np.array([])
    purpose2 = np.array([])
    for i in csv:
        df = pd.read_csv(i, index_col=False)
        names2 = np.append(names2, df.iloc[:, -2].values)

        purpose2 = np.append(purpose2, df.iloc[:, -1].values)

    name = np.append(names1, names2)
    purpose = np.append(purpose1, purpose2)
    res = pd.DataFrame(data={'Company Name' : name, 'Company Purpose' : purpose})
    return res

def Sentiment_anlysis(df):
    analyzer = SentimentIntensityAnalyzer()
    nrow = len(df)
    Score = np.array([])

    for i in range(nrow):
        Score = np.append(Score, analyzer.polarity_scores(df['Company Purpose'][i])['compound'])

    df['Score'] = Score
    best = np.argmax(Score,axis=0)
    worst = np.argmin(Score,axis=0)

    print("The best business idea is " + df['Company Purpose'][best])
    print("The worst business idea is " + df['Company Purpose'][worst])

    return df

def Most_Common(df):
    word = ""
    for i in range(len(df)):
        word = word + df['Company Purpose'][i] + "\ "

    word_split = word.split()
    frequency = Counter(word_split)
    most_common = frequency.most_common(10)

    return most_common

def main():
    path = os.getcwd()
    path1 = path + '/Data'
    os.chdir(path1)
    #Script 1: Merge data
    # list the data and merge
    txt = ['5.txt']
    csv = ['1.csv','2.csv','3.csv','4.csv']
    MergeData = Merge(txt, csv)

    # output the file
    path2 = path + '/MergedData'
    os.chdir(path2)
    MergeData.to_csv('Merged Data.csv', index=False)

    #Script 2:best and worst
    df = pd.read_csv('Merged Data.csv')
    Sentiment_anlysis(df)
    #OUTput:
    #The best business idea is Visionary zero tolerance flexibility for envisioneer dot-com supply-chains
    #The worst business idea is  Customizable foreground archive for unleash killer functionalities

    #Script 3:Most common
    Common = Most_Common(df)
    print('10 Most Common Words:',Common)
    #output
    #10 Most Common Words: [('for', 250), ('real-time', 11), ('info-mediaries\\', 11), ('re-contextualize', 10), ('niches\\', 10), ('solutions\\', 10), ('global', 10), ('revolutionary', 9), ('channels\\', 9), ('functionalities\\', 9)]


if __name__ == "__main__":
    main()
