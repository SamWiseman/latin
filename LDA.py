'''This file implements LDA Topic Modeling'''

import csv
import json
import sys
from collections import OrderedDict, Counter
from numpy.random import choice
import time
import math
import evaluation
from operator import itemgetter
import copy

"""
runLDA(iterations, file, topics, alpha, beta) -- this method handles the iteration of LDA, calling helper methods
for each iteration and printing at the end.
:param iterations: int -- number of iterations to run
:param file: string -- file name to read data from
:param topics: int -- how many topics to create
:param alpha: float -- hyperparameter to add to each P(w|t)
:param beta: float -- hyperparameter to add to each P(t|d)
"""


def runLDA(corpus, iterations, alpha, beta):
    for i in range(0, iterations):
        # getting start time to measure runtime
        # delete the line below for the final release!
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


class CorpusData:
    # Location Information
    # 2d array: outer array contains documents which are arrays of words in the order they appear
    wordLocationArray = []
    # 2d array: outer array contains documents which are arrays of topics which exactly match the words
    # in the previous array
    topicAssignmentByLoc = []

    #data structures used for creating the annotated text
    #wordLocArrayStatic is wordLocationArray with stopwords included
    #topicAssignByLocStatic is topicAssignmentByLoc with stopwords included
    wordLocArrayStatic = []
    topicAssignByLocStatic = []

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

    # consideration: the way the csv is organized could vary. should we standardize it as
    # pat of preprocessing? we are currently using the format given by wikiParse.py
    # working csv
    file = ""

    # number of topics we want in the algorithm

    numTopics = 0

    stopwords = []

    # constructor
    def __init__(self, file, numTopics):
        self.file = file
        self.numTopics = numTopics

    # reads the csv and loads the appropriate data structures. may be refactored by struct
    # stopLowerBound and stopUpperBound are floats between 0 and 1
    # they represent what proportion of documents a word can appear in without getting filtered out
    # stopWhitelist and stopBlacklist are two lists of strings
    # strings in stopWhitelist will not be filtered out even if they are outside the document bounds
    # strings in stopBlacklist will be filtered out even if they are inside the document bounds
    def loadData(self, stopLowerBound, stopUpperBound, stopWhitelist, stopBlacklist):
        with open(self.file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            wordsColumn = []
            docColumn = []
            curDoc = ""
            curDocIndex = -1
            # load our 2d wordsByLocation array
            for row in reader:
                # add the word to the current array if word's doc is curDoc
                if curDoc == row[1]:
                    self.wordLocationArray[curDocIndex].append(row[0].lower())
                # add the word to a new doc array if word's doc is not curDoc
                else:
                    curDoc = row[1]
                    curDocIndex += 1
                    self.wordLocationArray.append([])
                    self.wordLocationArray[curDocIndex].append(row[0].lower())
                # have a list representing each column in the doc
                wordsColumn.append(row[0].lower())
                docColumn.append(row[1])

                # load word counts into dictionary
        self.uniqueWordDict = Counter(wordsColumn)
        self.wordLocArrayStatic = copy.deepcopy(self.wordLocationArray)

        # removes stopwords (above 90%, below 5%)
        wordDocCounts = dict.fromkeys(self.uniqueWordDict, 0)
        for doc in self.wordLocationArray:
            wordInDoc = []
            for word in doc:
                if word not in wordInDoc:
                    wordInDoc.append(word)
                    wordDocCounts[word] += 1
        if stopLowerBound == "off":
            stopLowerBound = 0
        if stopUpperBound == "off":
            stopUpperBound = 2
        lowerBound = math.ceil(len(self.wordLocationArray) * stopLowerBound)
        upperBound = math.ceil(len(self.wordLocationArray) * stopUpperBound)
        # create an array of stopwords
        for word in wordDocCounts:
            if wordDocCounts[word] <= lowerBound or wordDocCounts[word] >= upperBound:
                self.stopwords.append(word)
        for bannedWord in stopBlacklist:
            if bannedWord not in self.stopwords:
                self.stopwords.append(bannedWord)
        for allowedWord in stopWhitelist:
            if allowedWord in self.stopwords:
                self.stopwords.remove(allowedWord)

        # remove all stopwords from wordLocationArray and uniqueWordDict

        for docWords in self.wordLocationArray:
            docWords[:] = [w for w in docWords if w not in self.stopwords]
        for w in self.stopwords:
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
        # count words in each document (docWordCounts)
        docSet = list(OrderedDict.fromkeys(docColumn))
        for doc in docSet:
            self.docTotalWordCounts.append(docColumn.count(doc))

        # build topicsByLocation by going through each topic in a loop
        for i in range(len(self.wordLocationArray)):
            self.topicAssignmentByLoc.append([0] * len(self.wordLocationArray[i]))
        chosenTopic = 0
        for i in range(len(self.wordLocationArray)):
            for j in range(len(self.wordLocationArray[i])):
                chosenTopic += 1
                chosenTopic %= self.numTopics
                # randTopic = random.randrange(self.numTopics)
                # self.topicsByLocation[i][j] = randTopic
                self.topicAssignmentByLoc[i][j] = chosenTopic

        # create wordTopicCounts using the information in topicsByLocation
        for i in range(len(self.wordLocationArray)):
            for j in range(len(self.wordLocationArray[i])):
                word = self.wordLocationArray[i][j]
                assignedTopic = self.topicAssignmentByLoc[i][j]
                if word not in self.wordDistributionAcrossTopics:
                    self.wordDistributionAcrossTopics[word] = [0] * self.numTopics
                self.wordDistributionAcrossTopics[word][assignedTopic] += 1

        # create docList
        for i in range(len(self.wordLocationArray)):
            self.docTopicalWordDist.append([])
            for j in range(self.numTopics):
                self.docTopicalWordDist[i].append(0)
        # i indexes by document in topicsbylocation, then the topic number becomes an index
        for i in range(len(self.topicAssignmentByLoc)):
            for topic in self.topicAssignmentByLoc[i]:
                self.docTopicalWordDist[i][topic] += 1

        # loading topicList and topicWordCounts
        self.topicTotalWordCount = [0] * self.numTopics
        self.topicWordInstancesDict = [{} for _ in range(self.numTopics)]
        for word in self.wordDistributionAcrossTopics:
            wordTopics = self.wordDistributionAcrossTopics[word]
            for i in range(self.numTopics):
                self.topicWordInstancesDict[i][word] = wordTopics[i]
                self.topicTotalWordCount[i] += wordTopics[i]
                # if wordTopics[i] != 0:
                #    self.topicWordCounts[i] += 1

    """
    printTopics() -- this method prints each topic in topicList on a new line in the format 
    "Topic 1: word1, word2, word3, etc." The words are sorted according to highest incidence 
    in the topic.
    """

    def printTopics(self):
        for topic in self.topicWordInstancesDict:
            print("Topic " + str(self.topicWordInstancesDict.index(topic) + 1) + ": "),
            print(", ".join(sorted(topic, key=topic.get, reverse=True)))

    def encodeData(self, readfile, topics, iterations, alpha, beta, outputname):
        for doc in self.topicAssignmentByLoc:
            for location in range(len(doc)):
                doc[location] = int(doc[location])
        for doc in self.topicAssignByLocStatic:
            for location in range(len(doc)):
                doc[location] = int(doc[location])
        dumpDict = {'dataset': readfile[:-4],
                    'topics': topics,
                    'iterations': iterations,
                    'alpha': alpha,
                    'beta': beta,
                    'wordsByLocation': self.wordLocationArray,
                    'wordsByLocationWithStopwords': self.wordLocArrayStatic,
                    'topicsByLocation': self.topicAssignmentByLoc,
                    'topicsByLocationWithStopwords': self.topicAssignByLocStatic,
                    'wordCounts': self.uniqueWordDict,
                    'wordTopicCounts': self.wordDistributionAcrossTopics,
                    'topicList': self.topicWordInstancesDict,
                    'topicWordCounts': self.topicTotalWordCount,
                    'docList': self.docTopicalWordDist,
                    'docWordCounts': self.docTotalWordCounts,
                    'stopwords': self.stopwords}
        outputfile = outputname+".json"
        with open(outputfile, 'w') as outfile:
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
        # normalize probabilities
        normalizedProbabilities = []
        rawsum = sum(newWordProbs)
        for probability in newWordProbs:
            if rawsum == 0:
                normalizedProbabilities.append(0.0)
            else:
                normalizedProbabilities.append(probability / rawsum)
        return normalizedProbabilities

    def outputAsCSV(self, outputname):
        loadData = []
        largestTopic = max(self.topicTotalWordCount)
        for i in range(largestTopic + 2):
            new = []
            for j in range(self.numTopics * 3):
                new.append(0)
            loadData.append(new)

        # formatting
        for i in range(self.numTopics):
            loadData[0][3 * i] = ''
            loadData[0][3 * i + 1] = 'Topic' + str(i + 1)
            loadData[0][3 * i + 2] = ''
            loadData[1][3 * i] = 'Word'
            loadData[1][3 * i + 1] = 'Count'
            loadData[1][3 * i + 2] = 'Percentage'

        for i in range(self.numTopics):
            topicAsList = []
            for k, v in self.topicWordInstancesDict[i].items():
                percent = (v / self.topicTotalWordCount[i]) * 100
                topicAsList.append([k, v, percent])
            topicAsList.sort(key=itemgetter(1), reverse=True)
            for j in range(len(topicAsList)):
                loadData[j + 2][3 * i] = topicAsList[j][0]
                loadData[j + 2][3 * i + 1] = topicAsList[j][1]
                loadData[j + 2][3 * i + 2] = topicAsList[j][2]
        outputfile = outputname+".csv"
        with open(outputfile, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            count = 0
            for row in loadData:
                if count < 200:
                    filewriter.writerow(row)
                    count += 1
                else:
                    break

    #create versions of the data structures that include stopwords in order to create the annotated text
    def createAnnoTextDataStructure(self):
        stopwordTopic = -1
        for document in range(len(self.wordLocationArray)):
            docTopicList = []
            counter = 0
            for word in range(len(self.wordLocArrayStatic[document])):
                if len(self.wordLocationArray[document]) > counter:
                    if self.wordLocArrayStatic[document][word] == self.wordLocationArray[document][counter]:
                        docTopicList.append(self.topicAssignmentByLoc[document][counter])
                        counter += 1
                    else:
                        docTopicList.append(stopwordTopic)
                else:
                    docTopicList.append(stopwordTopic)
            self.topicAssignByLocStatic.append(docTopicList)


def txtToCsv(fileName, splitString):
    fileString = open(fileName, 'r').read().lower()
    wordList = fileString.split()
    if splitString[:3] == 'num':
        numDocs = int(splitString[3:])
        docLength = len(wordList) // numDocs
        docStringsArray = getDocsOfLength(docLength, wordList)
    #to have a fixed length document, input "lengthXX" for documents of length XX
    elif splitString[:6] == 'length':
        docLength = int(splitString[6:])
        docStringsArray = getDocsOfLength(docLength, wordList)
    else: 
        docStringsArray = fileString.split(splitString)
    print("Number of documents: " + str(len(docStringsArray)))
    for i in range(len(docStringsArray)):
        #TODO: Handle all escape characters
        docStringsArray[i] = docStringsArray[i].replace("\n", " ")
    csvfilename = fileName[:-4]+".csv"
    with open(csvfilename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        currentDoc = 1
        for docString in docStringsArray:
            wordsArray = docString.split(' ')
            for word in wordsArray:
                word = word.strip('.,!?"():;\n\t')
                if word != '':
                    filewriter.writerow([word,str(currentDoc)])
            currentDoc += 1

def getDocsOfLength(docLen, wordList):
    print("Length of each document: " + str(docLen))
    docStringsArray = []
    while wordList:
        doc = " ".join(str(wordList[i]) for i in range(min(docLen, len(wordList))))
        wordList = wordList[docLen:]
        docStringsArray.append(doc)
    return docStringsArray

def makeChunkString(chunkType, chunkParam):
    chunkString = ''
    if chunkType == 'number of documents':
        chunkString += 'num'
        chunkString += str(chunkParam)
    elif chunkType == 'length of documents':
        chunkString += 'length'
        chunkString += str(chunkParam)
    elif chunkType == 'string':
        chunkString = chunkParam
    else:
        print("Invalid chunkType given.\n")
        exit()
    return chunkString


def main():
    configFile = sys.argv[1]
    configString = open(configFile, 'r').read()
    config = json.loads(configString)
    source = config["required parameters"]["source"]
    iterations = config["required parameters"]["iterations"]
    topics = config["required parameters"]["topics"]
    outputname = config["required parameters"]["output name"]
    upperlimit = config["stopword options"]["upper limit"]
    lowerlimit = config["stopword options"]["upper limit"]
    whitelist = config["stopword options"]["whitelist"]
    blacklist = config["stopword options"]["blacklist"]
    chunkingoptions = config["chunking options"]
    chunkType = ""
    chunkParam = 0
    for option in chunkingoptions.keys():
        if chunkingoptions[option] != "off":
            chunkType = option
            chunkParam = chunkingoptions[option]
            break
    alpha = config["hyperparameters"]["alpha"]
    beta = config["hyperparameters"]["beta"]
    chunkString = makeChunkString(chunkType, chunkParam)
    if source[-3:] == 'txt':
        txtToCsv(source, chunkString)
        source = source[:-4] + ".csv"

    corpus = CorpusData(source, topics)
    corpus.loadData(upperlimit, lowerlimit, whitelist, blacklist)
    runLDA(corpus, iterations, alpha, beta)
    corpus.createAnnoTextDataStructure()
    corpus.encodeData(source, topics, iterations, alpha, beta, outputname)

    # clean up words from topics that have value 0 (i.e. are not assigned to that topic)
    for topic in corpus.topicWordInstancesDict:
        for key in list(topic.keys()):
            if topic[key] == 0:
                del topic[key]
    corpus.printTopics()
    corpus.outputAsCSV(outputname)
    # evaluation.compareDistributions(corpus)
    # evaluation.compareTopicSize(corpus)
    # evaluation.topicSpecificity(corpus)

if __name__ == "__main__":
    main()

