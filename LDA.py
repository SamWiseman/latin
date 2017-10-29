'''This file implements LDA Topic Modeling'''

import csv
import random
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
    # dictionary mapping unique words to the number of times they appear
    wordCounts = {}
    # dictionary mapping unique words to an array indexed by topic of how many times they
    # appear in each topic
    wordTopicCounts = {}
        
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
    numTopics = 0
    #constructor
    def __init__(self, file, numTopics):
        self.file = file
        self.numTopics = numTopics
    
    #reads the csv and loads the appropriate data structures. may be refactored by struct  
    def loadData(self):
        with open(self.file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            wordsColumn = []
            docColumn = []
            curDoc = ""
            curDocIndex = -1
            #load our 2d wordsByLocation array
            for row in reader:
                if curDoc == row[1]:
                    self.wordsByLocation[curDocIndex].append(row[0].lower())
                else:
                    curDoc = row[1]
                    curDocIndex += 1
                    self.wordsByLocation.append([])
                    self.wordsByLocation[curDocIndex].append(row[0].lower())
                #have a list representing each column in the doc
                wordsColumn.append(row[0].lower())
                docColumn.append(row[1])  
                     
        #load word counts into dictionary
        sortedWords = sorted(wordsColumn)
        count = 0
        lastWord = sortedWords[0]
        for word in sortedWords:
            if word == lastWord:
                count += 1
            else:
                self.wordCounts[lastWord] = count
                count = 1
            lastWord = word 
        
        #count words in each document
        docSet = set(docColumn)
        for doc in docSet:
            self.docWordCounts.append(docColumn.count(doc))
        
        #build topicsByLocation by putting a random number in a slot for every word
        for i in range(len(self.wordsByLocation)):
            self.topicsByLocation.append([0] * len(self.wordsByLocation[i]))    
        for i in range(len(self.wordsByLocation)):
            for j in range(len(self.wordsByLocation[i])):
                randTopic = random.randrange(self.numTopics)
                self.topicsByLocation[i][j] = randTopic
          
        #create wordTopicCounts using the information in topicsByLocation       
        for i in range(len(self.wordsByLocation)):
            for j in range(len(self.wordsByLocation[i])):
                word = self.wordsByLocation[i][j]
                assignedTopic = self.topicsByLocation[i][j]
                if word not in self.wordTopicCounts:
                    self.wordTopicCounts[word] = [0] * self.numTopics
                self.wordTopicCounts[word][assignedTopic] += 1
        
#test function for data loading
def loadTest():
    data = BigData('wiki5Docs.csv', 10)
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
