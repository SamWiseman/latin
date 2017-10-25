'''
DATA STRUCTURES

Location Information (2D arrays, inner array for words, outer array for docs)
    wordLocationList = [[word, word, word], [word,word,word]]
    tagsByLocation = [[tag, tag, tag], [tag, tag, tag]]

Word Information (indices line up)
    individualWordList = [word, word, word] (not updated after init)
    wordTopicCounts = [{topic:count,topic:count}, {topic:count,topic:count]}
    wordCounts = [int, int, int] (not updated after init)
    wordProbabilities = {word:[topicProb, topicProb], word:[topicProb, topicProb]} (random init)

Topic Information (indices line up)
    topicList = [{word:count, word:count}, {word:count,word:count}]
    topicWordCounts = [#wordsInTopic, #wordsInTopic]

Document Information (indices line up)
    docList = [{topic:#docWords, topic:#docWords}, {topic:#docWords, topic:#docWords}]
    docWordCounts = [#wordsInDoc, #wordsInDoc]



METHODS

init
    Make the above data structures from input.

runLDA
    for i in range(0, iterations):
        for doc in corpus:
            for word in doc:
                wordProbabilities[word] = calculateProbabilities(word)
                updateDataStructures(word)

calculateProbabilities(word)
    for topic in topicList:
        use p(w|t) and p(t|d) to calculate p(t|w) (this is where hyperparameters happen)
        update wordProbabilities[word][topic]

updateDataStructures(word)
    probabilities = wordProbabilities[word]
    update all data structures based on new probabilities

'''