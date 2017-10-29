'''This file implements LDA Topic Modeling'''

import csv
from numpy.random import choice

# global data structures (see pseudoLDA.py for explanations)
wordsByLocation = []
topicsByLocation = []
individualWordList = []
wordTopicCounts = []
wordCounts = []
topicList = []
topicWordCounts = []
docList = []
docWordCounts = []

"""
printTopics() -- this method prints each topic in topicList on a new line in the format 
"Topic 1: word1, word2, word3, etc." The words are sorted according to highest incidence 
in the topic.
"""
#TODO: print to file instead of to command line
def printTopics():
    for topic in topicList:
        print("Topic " + str(topicList.index(topic)+1) + ": "),
        print(", ".join(sorted(topic, key=topic.get, reverse=True)))


"""
runLDA(iterations, alpha, beta) -- this method handles the iteration of LDA, calling helper methods
for each iteration and printing at the end.
:param iterations: int -- number of iterations to run
:param alpha: float -- hyperparameter to add to each P(w|t)
:param beta: float -- hyperparameter to add to each P(t|d)
"""
def runLDA(iterations, alpha, beta):
    for i in range(0, iterations):
        for doc in wordsByLocation:
            for word in doc:
                wordProbabilities = calculateProbabilities(doc, word)
                updateDataStructures(word, doc, wordProbabilities)
    printTopics()


class BigData:
    # Location Information
    # 2d array: outer array contains documents which are arrays of words in the order they appear
    wordsByLocation = []
    # 2d array: outer array contains documents which are arrays of topics which exactly match the words
    # in the previous array
    topicsByLocation = []

    # Word Information
    # a list of all unique words 
    individualWordList = []
    # 2d array: outer array contains words (matched to index in individualWordList)
    # inner array has the count of the word across the topics. index is the topic's "number"
    # sam asks: could we use a dictionary instead to combine wordtopiccounts with individualwordlist
    # that maps unique words to [topic1count, topic2count,...]
    wordTopicCounts = []
    # list of numbers corresponding to words individual word list (how many times each word appears)
    # could this be a dictionary instead?
    wordCounts = []

    # Topic Information
    # array of topics, where topics are dictionaries
    # keys are words and the values are counts for that word
    topicList = []
    # array of numbers, where each number is the number of words in the topic (corresponding by index)
    topicWordCounts = []

    # Document Information
    # array of documents, each document is an array
    # each index is a topic
    # each value is number of words in the document that belong to that topic
    docList = []
    # a list of of the number of words in each document
    docWordCounts = []
    
    #consideration: the way the csv is organized could vary. should we standardize it as
    #pat of preprocessing? we are currently using the format given by wikiParse.py
    #working csv
    file = ""
    
    #number of topics we want in the algorithm 
    topicNum = 0
    #constructor
    def __init__(self, file, topicNum):
        self.file = file
        self.topicNum = topicNum
        
    def loadData(self):
        with open(self.file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            wordsColumn = []
            docColumn = []
            curDoc = ""
            curDocIndex = -1
            #load our 2d wordsbylocation array
            #this almost works but it takes the first word out of the second (and maybe
            #the rest) of the arrays for some reason. and has an extra array at the end.
            for row in reader:
                if curDoc == row[1]:
                    wordsByLocation[curDocIndex].append(row[0])
                else:
                    curDoc = row[1]
                    curDocIndex += 1
                    wordsByLocation.append([])
                    wordsByLocation[curDocIndex].append(row[0])
                #have a list representing each column in the doc
                wordsColumn.append(row[0])
                docColumn.append(row[1])       
        #load our unique words array by using set data structure
        self.individualWordList = list(set(wordsColumn)).sort()
        
        
#test function for data loading
def loadTest():
    data = BigData('wiki5Docs.csv', 0)
    data.loadData()
    
''' LDA methods for recalculating the probabilities of each word by topic '''
#TODO: hyperparameters
def calculateProbabilities(docCoord, wordCoord):
    word = wordsByLocation[docCoord][wordCoord]
    newWordProbs = []
    for i in range(len(topicList)):
        # pwt = P(w|t)
        topicDict = topicList[i]
        wordCount = topicDict[word]
        pwt = wordCount / topicWordCounts[i]
        # ptd = P(t|d)
        wordsInTopicInDoc = docList[docCoord][i]
        ptd = wordsInTopicInDoc / docWordCounts[docCoord]
        # ptw = P(t|w)
        ptw = pwt * ptd
        newWordProbs.append(ptw)
        #TODO: once we get hyperparameters involved, will we need to re-regularize here?
        '''
        example:
        
        regularProbabilities = []
        one = sum(newWordProbs)
        for probability in newWordProbs:
            regularProbabilities.append(probability/one)
        return regularProbabilities
        '''
    return newWordProbs


"""
updateDataStructures(word, doc, wordProbabilities) -- this method chooses a new topic assignment for the 
given instance of the word based on its calculated topic probabilities and updates all relevant data 
structures to change its assignment

:param word: int -- word index in location 2D arrays
:param doc: int -- doc index in location 2D arrays
:param wordProbabilities: list -- probabilities of word in each topic
:return: none
"""
def updateDataStructures(word, doc, wordProbabilities):
    wordString = wordsByLocation[doc][word]
    oldTopic = topicsByLocation[doc][word]

    newTopic = choice(range(0, len(wordProbabilities)), p=wordProbabilities)
    topicsByLocation[doc][word] = newTopic

    topicList[oldTopic][wordString] -= 1
    topicList[newTopic][wordString] += 1

    topicWordCounts[oldTopic] -= 1
    topicWordCounts[newTopic] += 1

    docList[doc][oldTopic] -= 1
    docList[doc][newTopic] += 1
