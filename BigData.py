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
    docWordCounts = [#wordsInDoc, #wordsInDoc]
'''

import csv
def BigData():
	#Location Information
	#2d array: outer array contains documents which are arrays of words in the order they appear
	wordLocationList = []
	#2d array: outer array contains documents which are arrays of topics which exactly match the words
	#in the previous array
	tagsByLocation = []

	#Word Information
	#a list of all unique words
	individualWordList = []
	#2d array: outer array contains words (matched to index in individualWordList)
	#inner array has the count of the word across the topics. index is the topic's "number" 
	wordTopicCounts = []
	#list of numbers corresponding to words individual word list (how many times each word appears)
	wordCounts = []

	#Topic Information
	#array of topics, where topics are dictionaries 
	#keys are words and the values are counts for that word
	topicList = []
	#array of numbers, where each number is the number of words in the topic (corresponding by index)
	topicWordCounts = []

	#Document Information 
	#array of documents, each document is a dictionary
	#key is a topic (do we have a way to distinguish these from one another?)
	#value is number of words in the document that belong to that topic
	docList = []
	#a list of of the number of words in each document
	docWordCounts = []
