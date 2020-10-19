
import csv
import json

bpm_dict = {}

with open('SelectedLoops.csv') as csvfile:
    csvreader = csv.reader(csvfile,dialect='excel')
    next(csvreader)
    for row in csvreader:
        if 110 <= int(row[7]) <= 150:
            bpm_dict[row[0]] = {"bpm":row[7], "confidence":float(row[6])}


json.dump(bpm_dict, open("loops_bpm_to_timestretch.json", 'w'))

