'''This file implements LDA Topic Modeling'''

import csv
import json
import random
import sys
from collections import OrderedDict, Counter
from numpy.random import choice
import time
import math
from operator import itemgetter
import evaluation

"""
runLDA(iterations, file, topics, alpha, beta) -- this method handles the iteration of LDA, calling helper methods
for each iteration and printing at the end.
:param iterations: int -- number of iterations to run
:param file: string -- file name to read data from
:param topics: int -- how many topics to create
:param alpha: float -- hyperparameter to add to each P(w|t)
:param beta: float -- hyperparameter to add to each P(t|d)
"""
def runLDA(iterations, readfile, encodefile, topics, alpha=0, beta=0):
    corpus = CorpusData(readfile, topics)
    corpus.loadData()
    for i in range(0, iterations):
        #getting start time to measure runtime
        #delete the line below for the final release!
        startTime = time.clock()
        for doc in range(len(corpus.wordLocationArray)):
            for word in range(len(corpus.wordLocationArray[doc])):
                oldTopic = corpus.topicAssignmentByLoc[doc][word]
                corpus.removeWordFromDataStructures(word, doc, oldTopic)
                wordProbabilities = corpus.calculateProbabilities(doc, word, alpha, beta)
                newTopic = choice(range(len(wordProbabilities)), p=wordProbabilities)
                corpus.addWordToDataStructures(word, doc, newTopic)
        #printing the elapsed time (real-time)
        print("Time elapsed for iteration " + str(i) + ": " + str(time.clock() -startTime))
    corpus.encodeData(readfile, encodefile)
    # clean up words from topics that have value 0 (i.e. are not assigned to that topic)
    for topic in corpus.topicWordInstancesDict:
        for key in list(topic.keys()):
            if topic[key] == 0:
                del topic[key]
    corpus.printTopics()
    corpus.outputAsCSV()
    # evaluation.compareDistributions(corpus)
    # evaluation.compareTopicSize(corpus)
    # evaluation.topicSpecificity(corpus)

class CorpusData:
    # Location Information
    # 2d array: outer array contains documents which are arrays of words in the order they appear
    wordLocationArray = []
    # 2d array: outer array contains documents which are arrays of topics which exactly match the words
    # in the previous array
    topicAssignmentByLoc = []

    # Word Information
    # dictionary mapping unique words to the number of times they appear
    uniqueWordDict = {}
    # dictionary mapping unique words to an array indexed by topic of how many times they
    # appear in each topic
    wordDistributionAcrossTopics = {}
        
    # Topic Information
    # array of topics, where topics are dictionaries
    # keys are words and the values are counts for that word
    topicWordInstancesDict = []
    # array of numbers, where each number is the number of words in the topic (corresponding by index)
    topicTotalWordCount = []

    # Document Information
    # array of documents, each document is an array
    # each index is a topic
    # each value is number of words in the document that belong to that topic
    docTopicalWordDist = []
    # a list of of the number of words in each document
    docTotalWordCounts = []
    
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
                    self.wordLocationArray[curDocIndex].append(row[0].lower())
                #add the word to a new doc array if word's doc is not curDoc
                else:
                    curDoc = row[1]
                    curDocIndex += 1
                    self.wordLocationArray.append([])
                    self.wordLocationArray[curDocIndex].append(row[0].lower())
                #have a list representing each column in the doc
                wordsColumn.append(row[0].lower())
                docColumn.append(row[1])  
                     
        #load word counts into dictionary
        self.uniqueWordDict = Counter(wordsColumn)
        
        #removes stopwords (above 90%, below 5%)
        wordDocCounts = dict.fromkeys(self.uniqueWordDict, 0)
        for doc in self.wordLocationArray:
            wordInDoc = []
            for word in doc:
                if word not in wordInDoc:
                    wordInDoc.append(word)
                    wordDocCounts[word] += 1
        
        lowerBound = math.ceil(len(self.wordLocationArray) * 0.05)
        upperBound = math.ceil(len(self.wordLocationArray) * 0.90)
        stopwords = []
        #create an array of stopwords
        for word in wordDocCounts:
            if wordDocCounts[word] <= lowerBound or wordDocCounts[word] >= upperBound:
                stopwords.append(word)
        #remove all stopwords from wordLocationArray and uniqueWordDict
        for docWords in self.wordLocationArray:
            docWords[:] = [w for w in docWords if w not in stopwords]
        for w in stopwords:
            self.uniqueWordDict.pop(w, None)
        
        '''
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
        '''
        #count words in each document (docWordCounts)
        docSet = list(OrderedDict.fromkeys(docColumn))
        for doc in docSet:
            self.docTotalWordCounts.append(docColumn.count(doc))
        
        #build topicsByLocation by going through each topic in a loop
        for i in range(len(self.wordLocationArray)):
            self.topicAssignmentByLoc.append([0] * len(self.wordLocationArray[i]))
        chosenTopic = 0    
        for i in range(len(self.wordLocationArray)):
            for j in range(len(self.wordLocationArray[i])):
                chosenTopic += 1
                chosenTopic %= self.numTopics
                #randTopic = random.randrange(self.numTopics)
                #self.topicsByLocation[i][j] = randTopic
                self.topicAssignmentByLoc[i][j] = chosenTopic
          
        #create wordTopicCounts using the information in topicsByLocation       
        for i in range(len(self.wordLocationArray)):
            for j in range(len(self.wordLocationArray[i])):
                word = self.wordLocationArray[i][j]
                assignedTopic = self.topicAssignmentByLoc[i][j]
                if word not in self.wordDistributionAcrossTopics:
                    self.wordDistributionAcrossTopics[word] = [0] * self.numTopics
                self.wordDistributionAcrossTopics[word][assignedTopic] += 1
        
        #create docList
        for i in range(len(self.wordLocationArray)):
            self.docTopicalWordDist.append([])
            for j in range(self.numTopics):
                self.docTopicalWordDist[i].append(0)
        #i indexes by document in topicsbylocation, then the topic number becomes an index
        for i in range(len(self.topicAssignmentByLoc)):
            for topic in self.topicAssignmentByLoc[i]:
                self.docTopicalWordDist[i][topic] += 1
                
        #loading topicList and topicWordCounts
        self.topicTotalWordCount = [0] * self.numTopics
        self.topicWordInstancesDict = [{} for _ in range(self.numTopics)]
        for word in self.wordDistributionAcrossTopics:
            wordTopics = self.wordDistributionAcrossTopics[word]
            for i in range(self.numTopics):
                self.topicWordInstancesDict[i][word] = wordTopics[i]
                self.topicTotalWordCount[i] += wordTopics[i]
                #if wordTopics[i] != 0:
                #    self.topicWordCounts[i] += 1

    """
    printTopics() -- this method prints each topic in topicList on a new line in the format 
    "Topic 1: word1, word2, word3, etc." The words are sorted according to highest incidence 
    in the topic.
    """

    # TODO: print to file instead of to command line
    def printTopics(self):
        for topic in self.topicWordInstancesDict:
            print("Topic " + str(self.topicWordInstancesDict.index(topic) + 1) + ": "),
            print(", ".join(sorted(topic, key=topic.get, reverse=True)))

    def encodeData(self, readfile, encodefile):
        for doc in self.topicAssignmentByLoc:
            for location in range(len(doc)):
             doc[location] = int(doc[location])
        dumpDict = {'dataset': readfile,
                    'wordsByLocation': self.wordLocationArray,
                    'topicsByLocation': self.topicAssignmentByLoc,
                    'wordCounts': self.uniqueWordDict,
                    'wordTopicCounts': self.wordDistributionAcrossTopics,
                    'topicList': self.topicWordInstancesDict,
                    'topicWordCounts': self.topicTotalWordCount,
                    'docList': self.docTopicalWordDist,
                    'docWordCounts': self.docTotalWordCounts}
        with open(encodefile, 'w') as outfile:
            json.dump(dumpDict, outfile, indent=4)

    def removeWordFromDataStructures(self, word, doc, oldTopic):
        wordString = self.wordLocationArray[doc][word]
        self.topicWordInstancesDict[oldTopic][wordString] -= 1
        self.topicTotalWordCount[oldTopic] -= 1
        self.docTopicalWordDist[doc][oldTopic] -= 1

    def addWordToDataStructures(self, word, doc, newTopic):
        wordString = self.wordLocationArray[doc][word]
        self.topicAssignmentByLoc[doc][word] = newTopic
        self.topicWordInstancesDict[newTopic][wordString] += 1
        self.topicTotalWordCount[newTopic] += 1
        self.docTopicalWordDist[doc][newTopic] += 1
    
    ''' LDA methods for recalculating the probabilities of each word by topic '''
    def calculateProbabilities(self, docCoord, wordCoord, alpha, beta):
        word = self.wordLocationArray[docCoord][wordCoord]
        newWordProbs = []
        for i in range(len(self.topicWordInstancesDict)):
            # pwt = P(w|t)
            topicDict = self.topicWordInstancesDict[i]
            wordCount = topicDict[word]
            pwt = (wordCount + alpha) / (self.topicTotalWordCount[i] + alpha)
            # ptd = P(t|d)
            wordsInTopicInDoc = self.docTopicalWordDist[docCoord][i]
            ptd = (wordsInTopicInDoc + beta) / (self.docTotalWordCounts[docCoord] + beta)
            # ptw = P(t|w)
            ptw = pwt * ptd
            newWordProbs.append(ptw)
        #normalize probabilities
        normalizedProbabilities = []
        rawsum = sum(newWordProbs)
        for probability in newWordProbs:
            if rawsum == 0:
                normalizedProbabilities.append(0.0)
            else:
                normalizedProbabilities.append(probability/rawsum)
        return normalizedProbabilities

    def outputAsCSV(self):
        loadData = []
        largestTopic = max(self.topicTotalWordCount)
        for i in range(largestTopic+2):
            new = []
            for j in range(self.numTopics*3):
                new.append(0)
            loadData.append(new)

        # formatting
        for i in range(self.numTopics):
            loadData[0][3*i] = ''
            loadData[0][3*i+1] = 'Topic' + str(i + 1)
            loadData[0][3*i+2] = ''
            loadData[1][3*i] = 'Word'
            loadData[1][3*i+1] = 'Count'
            loadData[1][3*i+2] = 'Percentage'

        for i in range(self.numTopics):
            topicAsList = []
            for k,v in self.topicWordInstancesDict[i].items():
                percent = (v / self.topicTotalWordCount[i]) * 100
                topicAsList.append([k,v,percent])
            topicAsList.sort(key=itemgetter(1), reverse=True)
            for j in range(len(topicAsList)):
                loadData[j + 2][3 * i] = topicAsList[j][0]
                loadData[j + 2][3 * i + 1] = topicAsList[j][1]
                loadData[j + 2][3 * i + 2] = topicAsList[j][2]

        with open('output.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            for row in loadData:
                filewriter.writerow(row)

def txtToCsv(fileName, splitString):
    fileString = open(fileName, 'r').read().lower()
    if splitString == None:
        print('potaato')
        #TODO: find a way to split the file into an arbitrarily chosen number of documents
    else: 
        docStringsArray = fileString.split(splitString)
        for i in range(len(docStringsArray)):
            #TODO: Handle all escape characters
            docStringsArray[i] = docStringsArray[i].replace("\n", " ")
        with open('input.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            currentDoc = 1
            for docString in docStringsArray:
                wordsArray = docString.split(' ')
                for word in wordsArray:
                    word = word.strip('.,!?"():;\n\t')
                    if word != '':
                        filewriter.writerow([word,str(currentDoc)])
                currentDoc += 1

#tiny test function
def main():
    if len(sys.argv) != 5 and len(sys.argv) != 7:
        print("Usage: LDA.py iterations readfile topics encodefile (optional: alpha=0.8 beta=0.8)")
    elif len(sys.argv) == 7:
        iterations = int(sys.argv[1])
        readFile = sys.argv[2]
        topics = int(sys.argv[3])
        encodeFile = sys.argv[4]
        alpha = float(sys.argv[5])
        beta = float(sys.argv[6])
        if readFile[-3:] == 'txt':
            txtToCsv(readFile, '\n\n\n')
            readFile = 'input.csv'
        runLDA(iterations, readFile, encodeFile, topics, alpha, beta)
    else:
        print(sys.argv[1])

        iterations = int(sys.argv[1])
        readFile = sys.argv[2]
        topics = int(sys.argv[3])
        encodeFile = sys.argv[4]
        if readFile[-3:] == 'txt':
            txtToCsv(readFile, '\n\n\n')
            readFile = 'input.csv'
        runLDA(iterations, readFile, encodeFile, topics, 0.8, 0.8)

if __name__ == "__main__":
    main()