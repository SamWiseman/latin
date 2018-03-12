##short python script to create a Latin-to-English definition reference sheet
##takes a CSV of Latin words as input and outputs a CSV with corresponding definitions

import csv
import xlrd

workbook = xlrd.open_workbook('latinDefinitions.xlsx', on_demand = True)
worksheet = workbook.sheet_by_name('Sheet1')
#print(worksheet.cell(1, 0))

#dictionaryDict holds the location of the definitions in the latinDefinitions spreadsheet for each word

def makeDefRefs(numTopics):
    dictionaryDict = {}
    numRows = worksheet.nrows
    for i in range(numRows-1):
        dictionaryDict[worksheet.cell(i+1, 0).value] = i+1

    defsForCSV = []
    with open('output.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i in range(numTopics):
            count = 0
            for row in reader:
                if count < 2:
                    count += 1
                #get location in spreadsheet
                else:
                    key = row[3*i]
                    key = key.split(',')
                    key = key[0]
                    print(key)
                    defLocation = dictionaryDict[key]
                    definition = worksheet.cell(defLocation, 1)
                    defsForCSV.append([key, definition])
            count = 0

    with open('defRefs.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in defsForCSV:
            filewriter.writerow(row)

makeDefRefs(2)