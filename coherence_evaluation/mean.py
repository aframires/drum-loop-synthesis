import os
import json

file_reads = [  "results/multi_noenv_stats_comparison_feature_values.json",
                "results/multi_stats_comparison_feature_values.json",
                "results/stft_stats_comparison_feature_values.json",
                "results/wavstft_stats_comparison_feature_values.json"]

for file_read in file_reads:
    analysis = json.load(open(file_read, 'rb'))
    print(file_read)
    for key in analysis:
        print(key)
        total = 0
        count = 0
        for result in analysis[key]:
            total = total + analysis[key][result]
            count = count + 1
        print("{:.2%}".format(total/count))

print()
print()

for file_read in file_reads:
    analysis = json.load(open(file_read, 'rb'))
    features = ["boominess",
                "brightness",
                "depth",
                "hardness",
                "roughness",
                "sharpness",
                "warmth"]
    print(file_read.split("_")[0])
    for feature in features:
        print(feature,end=" & ")
        print("{:.2%}".format(analysis["ratio high > low (type1)"][feature]), end= " & ")
        print("{:.2%}".format(analysis["ratio high > mid (type2)"][feature]), end= " & ")
        print("{:.2%}".format(analysis["ratio mid > low (type3)"][feature]), end= " ")
        print("\\\\")
    print()
