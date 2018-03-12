"""
Usage:      python3 LDA.py

@authors    Estelle Bayer
            Martha Durett
            Brendan Friesen
            Adam Klein
            Bard Swallow
            Sam Wiseman
"""

import csv
import json
import sys
from collections import OrderedDict, Counter
from numpy.random import choice
import time
import math
from operator import itemgetter
import copy

def runLDA(corpus, iterations, alpha, beta):
    """An implementation of Latent Dirichlet Allocation. Probabilistically
        generates "topics" for a given corpus, each of which contains many
        words that are related by their coocurrence in the text. Uses the
        CorpusData data structure containing information about word location
        and outputs a list of the words in each topic to the shell after the
        desired number of iterations.

    Args:
        corpus (CorpusData): A data structure that has already called "loadData"
            on a text.
        iterations (int): The desired number of iterations for the LDA algorithm.
            More iterations lead to more consistent, coherent topics at the cost of
            a longer runtime.
        alpha (float): The first "hyperparameter" or "smoothing constant." Affects
            the P(w|t) calculation. When alpha is higher, documents tend to
            represent a greater variety of topics.
        beta (float): Another hyperparameter, this one affecting the P(t|d)
            calculation. A higher value for beta causes topics to contain a greater
            variety of words.

    """
    printProgressBar(0, iterations, prefix='Progress', suffix='complete', length=50)
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
        estTime = math.ceil((time.clock() - startTime) * (iterations - i) / 60)
        time.sleep(0.1)
        if i == iterations-1:
            printProgressBar(i + 1, iterations, prefix='Progress', suffix='complete', length=50)
        elif (estTime > 0):
            printProgressBar(i + 1, iterations, prefix='Progress', suffix='complete', length=50, estTimeRemaining=estTime)
        else:
            printProgressBar(i + 1, iterations, prefix='Progress', suffix='complete', length=50)


# class that stores words from a text and organizes them in various ways to facilitate LDA
# organization can be by location, by document, by word, and topic
# the methods load in text, encode data, and output topics in various ways
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
    #a list of punctuation
    punctuation = []
    #the locations of the punctuation
    puncLocations = []

    # consideration: the way the csv is organized could vary. should we standardize it as
    # pat of preprocessing? we are currently using the format given by wikiParse.py
    # working csv
    file = ""

    # number of topics we want in the output

    numTopics = 0

    stopwords = []

    def __init__(self, file, numTopics):
        """Constructor for CorpusData.

        Args:
            file (str): The path to a csv file storing the words in a corpus and their documents.
            numTopics (int): The number of topics to be output by this instance of LDA.

        Returns:
            CorpusData: An instance of the corpus data class that only contains a file and number of topics.

        """
        self.file = file
        self.numTopics = numTopics

    # reads the csv and loads the appropriate data structures. may be refactored by struct
    # stopLowerBound and stopUpperBound are floats between 0 and 1
    # they represent what proportion of documents a word can appear in without getting filtered out
    # stopWhitelist and stopBlacklist are two lists of strings
    # strings in stopWhitelist will not be filtered out even if they are outside the document bounds
    # strings in stopBlacklist will be filtered out even if they are inside the document bounds
    def loadData(self, stopLowerBound, stopUpperBound, stopWhitelist, stopBlacklist):
        """Reads the csv and loads the data structures used in LDA.

        Args:
            stopLowerBound (float): The minimum percentage of documents a word must appear in
                to be included in the algorithm.
            stopUpperBound (float): The maximum percentage of documents a word can appear in
                to be included in the algorithm.
            stopWhitelist (list): A list of words that should never be filtered out of the algorithm.
            stopBlacklist (list): A list of words that should always be filtered out of the algorithm.

        """
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

        # removes stopwords
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

        lowerBound = math.ceil(len(self.wordLocationArray) * float(stopLowerBound))
        upperBound = math.ceil(len(self.wordLocationArray) * float(stopUpperBound))

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
        self.stopwords = set(self.stopwords)

        for docWords in self.wordLocationArray:
            docWords[:] = [w for w in docWords if w not in self.stopwords]
        for w in self.stopwords:
            self.uniqueWordDict.pop(w, None)

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

    def printTopics(self):
        """Prints each topic in topicList on a new line in the format "Topic 1: word1, word2,
            word3, ..." The words are sorted from highest to lowest incidence in the topic.

        """
        for topic in self.topicWordInstancesDict:
            print("Topic " + str(self.topicWordInstancesDict.index(topic) + 1) + ": "),
            print(", ".join(sorted(topic, key=topic.get, reverse=True)))

    def encodeData(self, readfile, topics, iterations, alpha, beta, outputname, puncData):
        """Encodes information about the LDA output in a .json file. Stores the
            name of the input file, number of topics, number of iterations,
            hyperparameters, and data structures used to build the topics.

        Args:
            readfile (str): Name of the file given as input to LDA (with file extension).
            topics (int): Number of topics generated by this run of LDA.
            iterations (int): Number of iterations of this run of LDA.
            alpha (float): Alpha constant used in this run of LDA.
            beta (float): Beta constant used in this run of LDA.
            outputname (str): Name of the desired output JSON file (without file extension).
            puncData ([[str]]): Catalogue of tokens in the file that include punctuation or capitalization

        """
        for doc in self.topicAssignByLocStatic:
            for location in range(len(doc)):
                doc[location] = int(doc[location])
        for i in range(len(self.topicWordInstancesDict)):
            self.topicWordInstancesDict[i] = {k:v for k,v in self.topicWordInstancesDict[i].items() if v}

        dumpDict = {'dataset': readfile[:-4],
                    'topics': topics,
                    'iterations': iterations,
                    'alpha': alpha,
                    'beta': beta,
                    'wordsByLocationWithStopwords': self.wordLocArrayStatic,
                    'topicsByLocationWithStopwords': self.topicAssignByLocStatic,
                    'topicWordInstancesDict': self.topicWordInstancesDict,
                    'stopwords': list(self.stopwords),
                    'puncAndCap': puncData[0],
                    'puncCapLocations': puncData[1],
                    'newlineLocations': puncData[2]}
                    # 'puncAndCap': puncData[0], <--potential restructure
                    # 'newlineLocations': puncData[1]} <--potential restructure
        outputfile = outputname+".json"
        with open(outputfile, 'w') as outfile:
            json.dump(dumpDict, outfile, indent=4)

    def removeWordFromDataStructures(self, word, doc, oldTopic):
        """Removes an instance of a word from a topic and updates the
            appropriate data structures accordingly.

        Args:
            word (int): The index of the given word in its home document.
            doc (int): The index of the given document in wordLocationArray.
            oldTopic (int): The index of the topic from which the word in
                question is being removed.

        """
        wordString = self.wordLocationArray[doc][word]
        self.topicWordInstancesDict[oldTopic][wordString] -= 1
        self.topicTotalWordCount[oldTopic] -= 1
        self.docTopicalWordDist[doc][oldTopic] -= 1

    def addWordToDataStructures(self, word, doc, newTopic):
        """Adds an instance of a word to a topic and updates the approrpriate
            data structures accordingly.

        Args:
            word (int): The index of the given word in its home document.
            doc (int): The index of the given document in wordLocationArray.
            newTopic (int): The index of the topic to which the word in
                question is being added.
        """
        wordString = self.wordLocationArray[doc][word]
        self.topicAssignmentByLoc[doc][word] = newTopic
        self.topicWordInstancesDict[newTopic][wordString] += 1
        self.topicTotalWordCount[newTopic] += 1
        self.docTopicalWordDist[doc][newTopic] += 1

    def calculateProbabilities(self, docCoord, wordCoord, alpha, beta):
        """Given an instance of a word and two smoothing constants, returns
            a list of probabilities that a word will be assigned to each topic.

        Args:
            docCoord (int): The index of the document in question.
            wordCoord (int): The index of the desired word in that document.
            alpha (float): A constant default value for the P(w|t) calculation.
            beta (float): A constant default value for the P(t|d) calculation.

        Returns:
            [float]: A list of normalized probabilities that the given word
            will appear in each topic. Each index in this list corresponds to a
            topic, and the probability associated with the topic is used in
            runLDA to determine to which topic a word should be assigned.

        """
        word = self.wordLocationArray[docCoord][wordCoord]
        newWordProbs = []
        for i in range(len(self.topicWordInstancesDict)):
            # pwt = P(w|t)
            topicDict = self.topicWordInstancesDict[i]
            wordCount = topicDict[word]
            pwt = (wordCount + beta) / (self.topicTotalWordCount[i] + beta)
            # ptd = P(t|d)
            wordsInTopicInDoc = self.docTopicalWordDist[docCoord][i]
            ptd = (wordsInTopicInDoc + alpha) / (self.docTotalWordCounts[docCoord] + alpha)
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
        """Creates a .csv file containing readable results from the LDA run.
            Each topic has three columns: Word, Count (number of times the
            word appears in that topic) and Percentage (percentage of that
            topic that is the given word).

        Args:
            outputname (str): The desired name (with no file extension) of the
                .csv output file.

        """
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
        outputfile = outputname + ".csv"
        with open(outputfile, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            count = 0
            for row in loadData:
                if count < 200:
                    filewriter.writerow(row)
                    count += 1
                else:
                    break

    def createAnnoTextDataStructure(self):
        """Creates "static" versions of several data structures such that they
            contain stop words. Used in the creation of the annotated text.

        """
        stopwordTopic = -1
        for document in range(len(self.wordLocationArray)):
            docTopicList = []
            counter = 0
            for word in range(len(self.wordLocArrayStatic[document])):
                if len(self.wordLocationArray[document]) > counter:
                    if self.wordLocArrayStatic[document][word] == self.wordLocationArray[document][counter]:
                        docTopicList.append(self.topicAssignmentByLoc[document][counter])
                        # self.topicAssignByLocStatic.append(int(self.topicAssignmentByLoc[document][counter])) <--potential restructure
                        counter += 1
                    else:
                        docTopicList.append(stopwordTopic)
                        #self.topicAssignByLocStatic(int(stopwordTopic)) <--potential restructure
                else:
                    docTopicList.append(stopwordTopic)
            self.topicAssignByLocStatic.append(docTopicList)

def grabPuncAndCap(fileName):
    """Given .txt file, iterates through to create data structures containing all words
            that are attached to punctuation or contain capitalization. Also stores the
            location of the new line characters. Purpose: for loading in a user-friendly
            version of the text in the UI Annotated Text section.

        Args:
            fileName (str): The input file given by the user, if the input file is a .txt

        Returns:
            puncAndCap ([str]): A list of words, in the order they appear and as they appear
            in the .txt, that are attached to punctuation or contain capitalization
            puncCapLocations ([str]): A list of indexes that corresponds with the words in
            puncAndCap. The indexes indicate where in the full text each word in puncAndCap
            belongs.
            newlineLocations ([str]): A list of indexes. Each index indicates where a new line
            character belongs in the Annotated Text.

        """
    fileString = open(fileName, 'r').read().split()
    unsplitFile = open(fileName, 'r').read()
    newlineLocations = []
    count = 0
    trackToken = ''
    ##remove starting white space from file
    removeStartWhitespace = False
    while not removeStartWhitespace:
        if unsplitFile[0] == ' ' or unsplitFile[0] == '\t' or unsplitFile[0] == '\n':
            unsplitFile = unsplitFile[1:]
        else:
            removeStartWhitespace = True
    ##get locations of new line characters
    for i in range(len(unsplitFile)):
        if unsplitFile[i] == "\n":
            if trackToken != '':
                newlineLocations.append(count)
                trackToken = ''
                count += 1
            else:
                newlineLocations.append(newlineLocations[len(newlineLocations)-1])
        elif unsplitFile[i] == '\t' or unsplitFile[i] == ' ':
            if unsplitFile[i+1] != '\t' and unsplitFile[i+1] != ' ' and unsplitFile[i+1] != "\n":
                if trackToken != '':
                    trackToken = ''
                    count += 1
        else:
            trackToken += unsplitFile[i]

    ##get punctuation and capitalization info
    puncAndCap = []
    puncCapLocations = []
    count = 0
    for token in fileString:
        allPunc = False
        if '.' in token or ',' in token or '!' in token or '?' in token or '"' in token or '(' in token or ')' in token or ':' in token or ';' in token or '“' in token or '”' in token or '‘' in token or '’' in token or "'" in token or any(ltr for ltr in token if ltr.isupper()):
            allPunc = True
            for char in token:
                if char != "." and char != "," and char != "!" and char != "?" and char != '"' and char != "(" and char != ")" and char != ":" and char != ';' and char != '“' and char != '”' and char != '‘' and char != '’' and char != "'":
                    allPunc = False
            puncAndCap.append(token)
            if allPunc:
                puncCapLocations.append(count - 0.5)
            else:
                puncCapLocations.append(count)
        if not allPunc:
            count += 1
    return puncAndCap, puncCapLocations, newlineLocations
    #return fileString, newlineLocations <--potential restructure


def txtToCsv(fileName, splitString):
    """Given a .txt file containing the corpus, creates a .csv file that chunks the
        corpus based on splitString. The .csv file has two columns: the first contains
        words in the order they appear in the corpus, and the second contains the
        document (denoted by an integer starting from 1) that contains the word.

    Args:
        fileName (str): The name of the .txt file being read and converted.
        splitString (str): The string generated by makeChunkString that
            gives instructions on how to split up the txt file into documents.

    """
    fileString = open(fileName, 'r').read().lower()
    wordList = fileString.split()
    if splitString[:3] == 'num':
        numDocs = int(splitString[3:])
        docLength = len(wordList) // numDocs
        docStringsArray = getDocsOfLength(docLength, wordList, True)
    #to have a fixed length document, input "lengthXX" for documents of length XX
    elif splitString[:6] == 'length':
        docLength = int(splitString[6:])
        docStringsArray = getDocsOfLength(docLength, wordList, False)
    else: 
        docStringsArray = fileString.split(splitString.lower())
        temp = [docStringsArray[0]]
        for i in range(1, len(docStringsArray)):
            temp.append(splitString + docStringsArray[i])

        docStringsArray = temp
    print("Number of documents: " + str(len(docStringsArray)))
    for i in range(len(docStringsArray)):
        docStringsArray[i] = docStringsArray[i].replace("\n", " ")
    csvfilename = fileName[:-4]+".csv"
    with open(csvfilename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        currentDoc = 1
        for docString in docStringsArray:
            wordsArray = docString.split(' ')
            for word in wordsArray:
                word = word.strip('.,!?"“”‘’():;\n\t\'')
                if word != '':
                    filewriter.writerow([word,str(currentDoc)])
            currentDoc += 1

def getDocsOfLength(docLen, wordList, numCap):
    """Divides a list of words up by documents with a desired length. If the words
        don't divide evenly into this number, the last document will either be
        longer or shorter than the rest depending on the circumstances.

    Args:
        docLen (int): The number of words that should appear in each document.
        wordList ([str]): A list containing all of the words in a corpus split
            by whitespace.
        numCap (bool): True if a desired number of documents was specified and
            an additional "stub document" would be a problem. False otherwise.

    Returns:
        docStringsArray ([[str]]): A list of documents, each containing (docLen)
            words in the order they appear in the corpus. If numCap was True and
            wordList doesn't divide evenly by docLen, the last element of this
            list will be longer than the rest. Otherwise, whether there is a
            longer final document or a shorter final document depends on whether
            len(wordList) % docLen is less than or greater than docLen / 2
            respectively.

    """
    print("Length of each document: " + str(docLen))
    docStringsArray = []
    while wordList:
        doc = " ".join(str(wordList[i]) for i in range(min(docLen, len(wordList))))
        wordList = wordList[docLen:]
        docStringsArray.append(doc)
    lastDocLen = len(docStringsArray[-1].split())
    #if we have a fixed number of documents, we want to stick a "stub" document onto the last one
    #otherwise, we will do it if it is under half the desired document length
    if (numCap and lastDocLen < docLen) or \
    (not numCap and lastDocLen < docLen // 2):
        stubDoc = docStringsArray.pop()
        appendString = " " + stubDoc
        docStringsArray[-1] += appendString
    return docStringsArray

def makeChunkString(chunkType, chunkParam):
    """Creates a string that instructs txtToCsv how to split up documents.

    Args:
        chunkType (str): "number of documents" or "length of documents" depending on
            whether documents should be into a given number or by a given length
            (by number of words), or 'string' if a document should be split on a
            given string.
        chunkParam (int): Depending on chunkType, represents the desired number of
            documents or the desired length of documents in number of words.

    Returns:
        str: A string such as "num200" "length7150" or "\n\n\n" that instructs txtToCsv
            how to split up a text file into documents.

    """
    chunkString = ''
    if chunkType == 'number of documents':
        chunkString += 'num'
        chunkString += str(chunkParam)
    elif chunkType == 'length of documents':
        chunkString += 'length'
        chunkString += str(chunkParam)
    elif chunkType == 'split string':
        chunkString = chunkParam
    elif chunkType == 'using csv':
        pass
    else:
        print("Invalid chunkType given.\n")
        exit()
    return chunkString

# progressbar by user Greenstick on stackoverflow modified to include estimated time remaining
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', estTimeRemaining=0):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        estTimeRemaining - Optional : custom parameter added to original version of function
        :param timeremaining:
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if estTimeRemaining == 1:
        sys.stdout.write('\r%s |%s| %s%% %s %s %s' % (prefix, bar, percent, (suffix + '; Estimated time remaining: '), estTimeRemaining, 'minute'))
    elif estTimeRemaining > 0:
        sys.stdout.write('\r%s |%s| %s%% %s %s %s' % (prefix, bar, percent, (suffix + '; Estimated time remaining: '), estTimeRemaining, 'minutes'))
    else:
        sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
def main():
    """ Uses config.json to be run straight from the shell with no
        arguments: "python3 LDA.py [name of config file].json". Calls virtually every other
        function in the file. Topics are generated according to the settings
        in config.json and relevant data is placed in [outputname].json.
        Note: The config file must be in the same directory as LDA.py and must be
        formatted properly.

    """
    if len(sys.argv) < 2:
        print("Usage: python3 LDA.py [config file name].json")
        exit()
    configFile = sys.argv[1]
    configString = open(configFile, 'r').read()
    config = json.loads(configString)
    source = config["required parameters"]["source"]
    iterations = config["required parameters"]["iterations"]
    topics = config["required parameters"]["topics"]
    outputname = config["required parameters"]["output name"]
    upperlimit = config["stopword options"]["upper limit"]
    lowerlimit = config["stopword options"]["lower limit"]
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
        puncData = grabPuncAndCap(source)
        txtToCsv(source, chunkString)
        source = source[:-4] + ".csv"
    else:
        puncData = [[], [], []]

    corpus = CorpusData(source, topics)
    corpus.loadData(lowerlimit, upperlimit, whitelist, blacklist)
    runLDA(corpus, iterations, alpha, beta)
    corpus.createAnnoTextDataStructure()
    corpus.encodeData(source, topics, iterations, alpha, beta, outputname, puncData)

    # clean up words from topics that have value 0 (i.e. are not assigned to that topic)
    for topic in corpus.topicWordInstancesDict:
        for key in list(topic.keys()):
            if topic[key] == 0:
                del topic[key]
    corpus.outputAsCSV(outputname)

if __name__ == "__main__":
    main()

