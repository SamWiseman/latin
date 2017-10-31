'''This file implements LDA Topic Modeling'''

import csv
import random
import sys
from numpy.random import choice

"""
runLDA(iterations, file, topics, alpha, beta) -- this method handles the iteration of LDA, calling helper methods
for each iteration and printing at the end.
:param iterations: int -- number of iterations to run
:param file: string -- file name to read data from
:param topics: int -- how many topics to create
:param alpha: float -- hyperparameter to add to each P(w|t)
:param beta: float -- hyperparameter to add to each P(t|d)
"""
def runLDA(iterations, file, topics, alpha=0, beta=0):
    corpus = CorpusData(file, topics)
    corpus.loadData()
    for i in range(0, iterations):
        for doc in range(0, len(corpus.wordsByLocation)):
            for word in range(0, len(corpus.wordsByLocation[doc])):
                wordProbabilities = corpus.calculateProbabilities(doc, word, alpha, beta)
                oldTopic = corpus.topicsByLocation[doc][word]
                print("wordProbs:", wordProbabilities)
                newTopic = choice(range(0, len(wordProbabilities)), p=wordProbabilities)
                corpus.updateDataStructures(word, doc, oldTopic, newTopic)
    corpus.printTopics()

class CorpusData:
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
                #add the word to the current array if word's doc is curDoc
                if curDoc == row[1]:
                    self.wordsByLocation[curDocIndex].append(row[0].lower())
                #add the word to a new doc array if word's doc is not curDoc
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
        
        #count words in each document (docWordCounts)
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
        
        #todo: doclist, topiclist, topicwordcounts
        #create docList
        for i in range(len(self.wordsByLocation)):
            self.docList.append([])
            for j in range(self.numTopics):
                self.docList[i].append(0)
        #i indexes by document in topicsbylocation, then the topic number becomes an index
        for i in range(len(self.topicsByLocation)):
            for topic in self.topicsByLocation[i]: 
                self.docList[i][topic] += 1
                
        #loading topicList and topicWordCounts
        self.topicWordCounts = [0] * self.numTopics
        self.topicList = [{}] * self.numTopics
        for word in self.wordTopicCounts:
            wordTopics = self.wordTopicCounts[word]
            for i in range(self.numTopics):
                self.topicList[i][word] = wordTopics[i]
                
                self.topicWordCounts[i] += wordTopics[i]
                #if wordTopics[i] != 0:
                #    self.topicWordCounts[i] += 1

    """
    printTopics() -- this method prints each topic in topicList on a new line in the format 
    "Topic 1: word1, word2, word3, etc." The words are sorted according to highest incidence 
    in the topic.
    """

    # TODO: print to file instead of to command line
    def printTopics(self):
        for topic in self.topicList:
            print("Topic " + str(self.topicList.index(topic) + 1) + ": "),
            print(", ".join(sorted(topic, key=topic.get, reverse=True)))

    def updateDataStructures(self, word, doc, oldTopic, newTopic):
        wordString = self.wordsByLocation[doc][word]
        self.topicsByLocation[doc][word] = newTopic

        self.topicList[oldTopic][wordString] -= 1
        self.topicList[newTopic][wordString] += 1

        self.topicWordCounts[oldTopic] -= 1
        self.topicWordCounts[newTopic] += 1

        self.docList[doc][oldTopic] -= 1
        self.docList[doc][newTopic] += 1
    
    ''' LDA methods for recalculating the probabilities of each word by topic '''
    #TODO: hyperparameter inclusion in calculations
    def calculateProbabilities(self, docCoord, wordCoord, alpha, beta):
        word = self.wordsByLocation[docCoord][wordCoord]
        newWordProbs = []
        for i in range(len(self.topicList)):
            # pwt = P(w|t)
            topicDict = self.topicList[i]
            wordCount = topicDict[word]
            pwt = wordCount / self.topicWordCounts[i]
            # ptd = P(t|d)
            wordsInTopicInDoc = self.docList[docCoord][i]
            ptd = wordsInTopicInDoc / self.docWordCounts[docCoord]
            # ptw = P(t|w)
            ptw = pwt * ptd
            newWordProbs.append(ptw)
        #TODO: once we get hyperparameters involved, will we need to re-regularize here?
        '''
        regularProbabilities = []
        one = sum(newWordProbs)
        for probability in newWordProbs:
            if one == 0:
                regularProbabilities.append(0)
            else:
                regularProbabilities.append(probability/one)
        return regularProbabilities
        '''
        return newWordProbs

#tiny test function
def main():
    if len(sys.argv) != 4:
        print("Usage: LDA.py iterations filename topics")
    else:
        iterations = int(sys.argv[1])
        filename = sys.argv[2]
        topics = int(sys.argv[3])
        runLDA(iterations, filename, topics)


if __name__ == "__main__":
    main()