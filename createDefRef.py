##short python script to create a Latin-to-English definition reference sheet
##takes a CSV of Latin words as input and outputs a CSV with corresponding definitions

import csv
import xlrd

workbook = xlrd.open_workbook('latinDefinitions.xlsx', on_demand = True)
worksheet = workbook.sheet_by_name('Sheet1')

#dictionaryDict holds the location of the definitions in the latinDefinitions spreadsheet for each word

def makeDefRefs(numTopics):
    dictionaryDict = {}
    for i in range(worksheet.nrows()):
        dictionaryDict[worksheet.cell(0, i+2).value] = i+2

    defsForCSV = []
    with open('output.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i in range(numTopics):
            for row in reader:
                #get location in spreadsheet
                defLocation = dictionaryDict[row[3*i]]
                definition = worksheet.cell(0, defLocation)
                defsForCSV.append([row[3*i], definition])

    with open('defRefs.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in defsForCSV:
            filewriter.writerow(row)