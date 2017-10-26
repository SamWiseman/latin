'''
DATA STRUCTURES

Location Information (2D arrays, inner array for words, outer array for docs)
    wordsByLocation = [[word, word, word], [word,word,word]]
    topicsByLocation = [[tag, tag, tag], [tag, tag, tag]]

Word Information (indices line up)
    individualWordList = [word, word, word] (not updated after init)
    wordTopicCounts = [[count,count], [count,count]] (index of count in inner array corresponds to topic)
    wordCounts = [int, int, int] (not updated after init)

Topic Information (indices line up)
    topicList = [{word:count, word:count}, {word:count,word:count}]
    topicWordCounts = [#wordsInTopic, #wordsInTopic]

Document Information (indices line up)
    docList = [[#docWords, #docWords], [#docWords, #docWords]] (outer array is list of docs, inner array is #docwords in each topic)
    docWordCounts = [#wordsInDoc, #wordsInDoc] (not updated after init)



METHODS

init
    Make the above data structures from input.

runLDA
    for i in range(0, iterations):
        for doc in wordLocationList:
            for word in doc:
                wordProbabilities = calculateProbabilities(word)
                updateDataStructures(word, doc, wordProbabilities)
    printTopics()

calculateProbabilities(word)
    newWordProbabilities = []
    for topic in topicList:
        use p(w|t) and p(t|d) to calculate p(t|w) (this is where hyperparameters happen)
        put probability in newWordProbabilities at index topic
    return newWordProbabilities

updateDataStructures(word, doc, wordProbabilities)
    update all data structures based on new probabilities
    get word's current topic using tagsByLocation
    topicList(decrement old location, increment new location)
    topicWordCounts(decrement/increment)
    docList(find doc we were just in, decrement/increment docwords in topic if needed, decrement/increment docwords in other doc if needed)
    tagsByLocation

printTopics()
    for topic in topicList:
        print(topic + "\n")

'''


#DUMMY DATA STRUCTURES
#Location Information (2D arrays, inner array for words, outer array for docs)
wordsByLocation = [["word", "word"]]
topicsByLocation = [[0,1]]              #topics will be ints since they're essentially indices in topicList

#Word Information (indices line up)
individualWordList = ["word", "word"]   # (not updated after init)
wordTopicCounts = [[0, 1]]              # (index of count in inner array corresponds to topic)
wordCounts = [0, 1]                     # (not updated after init)

#Topic Information (indices line up)
topicList = [{"word1":0, "word2":1}]
topicWordCounts = [1, 0]

#Document Information (indices line up)
docList = [[0,1],[1,0]]                 # (number of docwords in inner array corresponds to topic)
docWordCounts = []                      # (not updated after init)






"""
updateDataStructures(word, doc, wordProbabilities) -- this method chooses a new topic assignment for the 
given instance of the word based on its calculated topic probabilities and updates all relevant data 
structures to change its assignment

:param word: int -- word index in location 2D arrays
:param doc: int -- doc index in location 2D arrays
:param wordProbabilities: list -- probabilities of word in each topic
:return:
"""
def updateDataStructures(word, doc, wordProbabilities):

    wordString = wordsByLocation[doc][word]
    oldTopic = topicsByLocation[doc][word]

    newTopic = wordProbabilities.index(max(wordProbabilities))
    topicsByLocation[doc][word] = newTopic

    topicList[oldTopic][wordString] = topicList[oldTopic][wordString] - 1
    topicList[newTopic][wordString] = topicList[newTopic][wordString] + 1

    topicWordCounts[oldTopic] = topicWordCounts[oldTopic] - 1
    topicWordCounts[newTopic] = topicWordCounts[newTopic] + 1

    docList[doc][oldTopic] = docList[doc][oldTopic] - 1
    docList[doc][newTopic] = docList[doc][newTopic] + 1

    return