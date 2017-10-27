'''This file implements LDA Topic Modeling'''

import csv

#global data structures (see pseudoLDA.py for explanations)
wordLocationList = []
topicsByLocation = []
individualWordList = []
wordTopicCounts = []
wordCounts = []
topicList = []
topicWordCounts = []
docList = []
docWordCounts = []

def BigData():
	#Location Information
	#2d array: outer array contains documents which are arrays of words in the order they appear
	wordLocationList = []
	#2d array: outer array contains documents which are arrays of topics which exactly match the words
	#in the previous array
	topicsByLocation = []

	#Word Information
	#a list of all unique words
	individualWordList = []
	#2d array: outer array contains words (matched to index in individualWordList)
	#inner array has the count of the word across the topics. index is the topic's "number"
	wordTopicCounts = []
	#list of numbers corresponding to words individual word list (how many times each word appears)
	wordCounts = []

	#Topic Information
	#array of topics, where topics are dictionaries
	#keys are words and the values are counts for that word
	topicList = []
	#array of numbers, where each number is the number of words in the topic (corresponding by index)
	topicWordCounts = []

	#Document Information
	#array of documents, each document is a dictionary
	#key is a topic (do we have a way to distinguish these from one another?)
	#value is number of words in the document that belong to that topic
	docList = []
	#a list of of the number of words in each document
	docWordCounts = []

''' LDA methods for recalculating the probabilities of each word by topic '''
def calculateProbabilities(docCoord, wordCoord):
    word = wordLocationList[docCoord][wordCoord]
    newWordProbs = []
    for i in range(len(topicList)):
        #pwt = P(w|t)
        topicDict = topicList[i]
        wordCount = topicDict[word]
        pwt = wordCount/topicWordCounts[i]
        #ptd = P(t|d)
        wordsInTopicInDoc = docList[docCoord][i]
        ptd = wordsInTopicInDoc/docWordCounts[docCoord]
        #ptw = P(t|w)
        ptw = pwt * ptd
        newWordProbs.append(ptw)
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

    wordString = wordLocationList[doc][word]
    oldTopic = topicsByLocation[doc][word]

    newTopic = wordProbabilities.index(max(wordProbabilities))
    topicsByLocation[doc][word] = newTopic

    topicList[oldTopic][wordString] = topicList[oldTopic][wordString] - 1
    topicList[newTopic][wordString] = topicList[newTopic][wordString] + 1

    topicWordCounts[oldTopic] = topicWordCounts[oldTopic] - 1
    topicWordCounts[newTopic] = topicWordCounts[newTopic] + 1

    docList[doc][oldTopic] = docList[doc][oldTopic] - 1
    docList[doc][newTopic] = docList[doc][newTopic] + 1