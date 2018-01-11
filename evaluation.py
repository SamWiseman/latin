import numpy as np

#Determine the variance in topic size for the whole document
#(Are topics all similarly sized?)
def compareTopicSize(LDAModel):
    with open('eval.txt', 'a') as eval:
        print(LDAModel.topicTotalWordCount)
        for i in range(len(LDAModel.topicTotalWordCount)):
            #print the variance of the word distribution into topics list
            eval.write("Topic size variance: " + np.var(LDAModel.topicTotalWordCount) + "\n")
            eval.write("\n")

#Compare the distribution of topics in the entire corpus to that of indivdual documents
#(Do the documents have distinctive distributions, or just mirror the whole?)
def compareDistributions(LDAModel):
    with open('eval.txt', 'a') as eval:
        eval.write("Whole corpus distribution: "+ list(map(lambda x: x/sum(LDAModel.topicTotalWordCount), LDAModel.topicTotalWordCount)))
        i = 0
        #print topic distributions for each document as percentage lists
        for document in LDAModel.docTopicalWordDist:
            document = list(map(lambda x: x/sum(document), document))
            eval.write("Document " + i + " distribution: " + document + "\n")
            i += 1
        eval.write("\n")

#Compare a topic's size to its prevalence
#(More specific topics should appear in larger proportions of a few documents)
def topicSpecificity(LDAModel):
    topics = []
    #generate an
    for i in range(len(LDAModel.topicList)):
        topics[i] = []
    for j in range(len(LDAModel.docTopicalWordDist)):
        document = LDAModel.docTopicalWordDist[j]
        for k in range(len(document)):
            if document[k] > 0:
                topics[k][j] = 1

    topicPercents = []
    for l in range(len(topics)):
        topicPercents[l] = sum(topics)/len(topics)

    avgPercentOfDoc = []
    for m in range(len(LDAModel.topicList)):
        docPercents = []
        for document in LDAModel.docTopicalWordDist:
            document = list(map(lambda x: x / sum(document), document))
            docPercents.append(document[m])
        avgPercentOfDoc[m] = np.mean(docPercents)



    with open('eval.txt', 'a') as eval:
        for t in range(len(LDAModel.topicList)):
            eval.write("Topic "+str(t)+ ": "+ topicPercents[t] * avgPercentOfDoc[t]+ "\n")
        eval.write("\n")


