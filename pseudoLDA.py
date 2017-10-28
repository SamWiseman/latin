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
    docList(find doc we were just in, decrement/increment docWords in topic if needed, decrement/increment docWords in other doc if needed)
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
docList = [[0,1],[1,0]]                 # (number of docWords in inner array corresponds to topic)
docWordCounts = []                      # (not updated after init)