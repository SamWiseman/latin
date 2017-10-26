'''
DATA STRUCTURES

Location Information (2D arrays, inner array for words, outer array for docs)
    wordLocationList = [[word, word, word], [word,word,word]]
    tagsByLocation = [[tag, tag, tag], [tag, tag, tag]]

Word Information (indices line up)
    individualWordList = [word, word, word] (not updated after init)
    wordTopicCounts = [[count,count], [count,count]] (index of count in inner array corresponds to topic)
    wordCounts = [int, int, int] (not updated after init)

Topic Information (indices line up)
    topicList = [{word:count, word:count}, {word:count,word:count}]
    topicWordCounts = [#wordsInTopic, #wordsInTopic]

Document Information (indices line up)
    docList = [{topic:#docWords, topic:#docWords}, {topic:#docWords, topic:#docWords}]
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