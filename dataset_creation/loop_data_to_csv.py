import json
import csv

conf_comparison =  json.load(open('conf_comparison.json', 'rb'))

with open('confidenceBPM.csv', 'w', newline='') as csvfile:
    fieldnames = ['id','ann_bpm', 'ann_confidence','perc_bpm', 'perc_confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for loop in conf_comparison:
        writer.writerow({'id': loop, 'ann_bpm': conf_comparison[loop][0]['annotated_bpm']['bpm'], 'ann_confidence':conf_comparison[loop][0]['annotated_bpm']['confidence'], 'perc_bpm' : conf_comparison[loop][1]['percival']['bpm'], 'perc_confidence': conf_comparison[loop][1]['percival']['confidence']})
    








