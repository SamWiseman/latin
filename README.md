# latin
Topic Modeling of Latin Texts Your Health

Config Instructions

Required Parameters
 Source: a string representing the file name from which you want to draw your data
 Iterations: an integer representing the number of iterations you want to run
 Topics: an integer representing the number of topics you want to sort words into
 Output Name: a string representing the file name you want this model's output files to have

Stopword Options
 Upper Limit: either a decimal value between 0 and 1 describing the percentage (or greater) of documents a word must
 appear in if it is to be tagged as a stopword OR “off” if you do not wish to use this feature
 Lower Limit: either a decimal value between 0 and 1 describing the percentage (or fewer) of documents a word must
 appear in if it is to be tagged as a stopword OR “off” if you do not wish to use this feature
 Whitelist: either a list of words to whitelist OR “off” if you do not wish to use this feature
 Blacklist: either a list of words to blacklist OR “off” if you do not wish to use this feature

Chunking Options
Note: only one of these methods may be used at one time; if a config file refers to more than one,
only the first will be used.
 Number of Documents: an integer representing the number of documents a text should be divided into OR “off” if you
 are using a different chunking method
 Length of Documents: an integer representing the number of words to make each document OR “off” if you are using a
 different chunking method
 Split String: a string representing the sequence of characters that separates documents from one another OR “off” if
 you are using a different chunking method

Hyperparameters
 Alpha: a decimal value between 0 and 1 representing how similar topics should be to each other in topic makeup
 Beta: a decimal value between 0 and 1 representing how similar topics should be to each other in word makeup