# latin
Topic Modeling of Latin Texts Your Health

## System Requirements
Requires prior installation of python3 and numpy. 

## Config Instructions

### Required Parameters
 **Source**: a string representing the file name from which you want to draw your data

 **Iterations**: an integer representing the number of iterations you want to run

 **Topics**: an integer representing the number of topics you want to sort words into

 **Output Name**: a string representing the file name you want this model's output files to have

### Stopword Options
 **Upper Limit**: either a decimal value between 0 and 1 describing the percentage (or greater) of documents a word must appear in if it is to be tagged as a stopword OR “off” if you do not wish to use this feature

 **Lower Limit**: either a decimal value between 0 and 1 describing the percentage (or fewer) of documents a word must
 appear in if it is to be tagged as a stopword OR “off” if you do not wish to use this feature

 **Whitelist**: either a list of words to whitelist OR “off” if you do not wish to use this feature

 **Blacklist**: either a list of words to blacklist OR “off” if you do not wish to use this feature

**Default stopwords**: This option is included in the config file creation form. Check these boxes to blacklist common stopwords in English and/or Latin as designated by the Perseus Digital Library. This is recommended for optimal output.

### Chunking Options
Note: only one of these methods may be used at one time; if a config file refers to more than one,
only the first will be used.

 **Number of Documents**: an integer representing the number of documents a text should be divided into OR “off” if you
 are using a different chunking method

 **Length of Documents**: an integer representing the number of words to make each document OR “off” if you are using a
 different chunking method

 **Split String**: a string representing the sequence of characters that separates documents from one another OR “off” if you are using a different chunking method

 **Using CSV**: "on" if you are using a CSV and don't need to chunk your document OR "off" if you are using a different chunking method

### Hyperparameters
 **Alpha**: a decimal value between 0 and 1 representing how similar documents should be to each other in topic makeup

 **Beta**: a decimal value between 0 and 1 representing how similar topics should be to each other in word makeup

## Usage
Requires that python3 be installed. Folder must contain the .txt or .csv input file containing the corpus as well as the .json file containing config information. 

    python3 LDA.py config.json

## Output
**filename.csv**: If a .txt file is designated as the source text, a .csv file will be created that matches each word to a document based on the user's chunking preferences. If the user runs the algorithm again and doesn't wish to change their chunking option, they can save time by designating this .csv as the source instead of their original .txt file.

**outputname.csv**: A file containing the model legible without the use of the visualization tool. In this CSV, each topic has three columns: Word, Count (number of times the
            word appears in that topic) and Percentage (percentage of that topic that is the given word).
**.json**: Contains the information that the app's electron-bede companion uses to display visualizations of the algorithm's output.