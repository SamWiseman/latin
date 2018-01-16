import numpy as np

#Determine the variance in topic size for the whole document
#(Are topics all similarly sized i.e. contain similar total quantities of words?)
def compareTopicSize(LDAModel):
    with open('eval.txt', 'a') as eval:
        #print the variance of the word distribution into topics list
        # eval.write("Topic size variance: " + np.var(LDAModel.topicTotalWordCount) + "\n")
        print("Topic size variance: " , np.var(LDAModel.topicTotalWordCount) , "\n")

#Compare the distribution of topics in the entire corpus to that of indivdual documents
#(Do the documents have distinctive distributions, or just mirror the whole?)
def compareDistributions(LDAModel):
    with open('eval.txt', 'a') as eval:
        # eval.write("Whole corpus distribution: "+ list(map(lambda x: x/sum(LDAModel.topicTotalWordCount), LDAModel.topicTotalWordCount)))
        print("Whole corpus distribution: ", list(map(lambda x: x/sum(LDAModel.topicTotalWordCount), LDAModel.topicTotalWordCount)))
        i = 0
        #print topic distributions for each document as percentage lists
        for document in LDAModel.docTopicalWordDist:
            if sum(document) != 0:
                document = list(map(lambda x: x/sum(document), document))
            # eval.write("Document " + i + " distribution: " + document + "\n")
            print("Document " ,i , " distribution: " , document , "\n")
            i += 1
        eval.write("\n")

#Compare a topic's size to its prevalence
#(More specific topics should appear in larger proportions of a few documents)
def topicSpecificity(LDAModel):
    #tracks what percent of documents contain the ith topic
    topicPercents = []
    #check to see whether a given topic appears in a document
    for j in range(len(LDAModel.docTopicalWordDist)):
        document = LDAModel.docTopicalWordDist[j]
        #check each doc's topic distribution for given ropic
        for k in range(len(document)):
            #if kth topic appears in document, increment kth topic document presence count
            if document[k] > 0:
                topicPercents[k] += 1


    topicPercents = list(map(lambda x: x/len(LDAModel.docTopicalWordDist), topicPercents))


    avgPercentOfDoc = []
    for m in range(len(LDAModel.topicTotalWordCount)):
        docPercents = []
        for document in LDAModel.docTopicalWordDist:
            document = list(map(lambda x: x / sum(document), document))
            docPercents.append(document[m])
        avgPercentOfDoc[m] = np.mean(docPercents)



    with open('eval.txt', 'a') as eval:
        for t in range(len(LDAModel.topicTotalWordCount)):
            # eval.write("Topic "+str(t)+ ": "+ topicPercents[t] , avgPercentOfDoc[t]+ "\n")
            print("Topic " , t , ": " , "present in" , topicPercents[t] , "% of documents" , avgPercentOfDoc[t], "\n")
        eval.write("\n")


