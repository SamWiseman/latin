import numpy as np

#
def compareTopicSize(LDAModel):
    with open('eval.txt', 'a') as eval:
        print(LDAModel.docTopicalWordDist)
        for i in range(len(LDAModel.docTopicalWordDist)):
            print(np.var(list(map(lambda x: x/sum(LDAModel.docTopicalWordDist[i])), LDAModel.docTopicalWordDist)))
            eval.write("Topic size variance: " + np.var(list(map(lambda x: x/sum(LDAModel.docTopicalWordDist[i]), LDAModel.docTopicalWordDist[i]))) + "\n")
            eval.write("\n")


def compareDistributions(LDAModel):
    wholeCorpList = []
    for document in LDAModel.docTopicalWordDist:
        for i in range(len(document)):
            wholeCorpList[i] = wholeCorpList[i] + document[i]
    totalWordsTopic = sum(wholeCorpList)
    wholeCorpList = list(map(lambda x: x/totalWordsTopic, wholeCorpList))
    with open('eval.txt', 'a') as eval:
        eval.write("Whole corpus distribution: "+ str(wholeCorpList))
        i = 0
        for document in LDAModel.docTopicalWordDist:
            document = list(map(lambda x: x/sum(document), document))
            eval.write("Document " + i + " distribution: " + str(document) + "\n")
            i += 1
        eval.write("\n")


def topicSpecificity(LDAModel):
    topics = []
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


