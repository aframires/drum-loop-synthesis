import json
import sqlite3


def compute_confidence_measure(estimated_bpm,
                       duration_samples,
                       start_effective_duration,
                       end_effective_duration,
                       sample_rate=44100, beat_range=range(1, 128), k=0.5):
    if estimated_bpm == 0:
        # This condition is to skip computing other steps if estimated bpm is 0, we already know that the
        # output will be 0
        return 0

    durations_to_check = [
        duration_samples,
        duration_samples - start_effective_duration,
        end_effective_duration,
        end_effective_duration - start_effective_duration
    ]

    beat_duration = (60.0 * sample_rate)/estimated_bpm
    L = [beat_duration * n for n in beat_range]
    thr_lambda = k * beat_duration
    confidences = list()
    for duration in durations_to_check:
        delta_l = min([abs(l - duration) for l in L])
        if delta_l > thr_lambda:
            confidences.append(0.0)
        else:
            confidences.append(1.0 - float(delta_l) / thr_lambda)
    return max(confidences)

#load durations
#load percival results
#load from db to get the ann_bpm

percival_bpms =  json.load(open('all_drum_loops/analysis_rhythm_percival14.json', 'rb'))
durations =  json.load(open('all_drum_loops/analysis_durations.json', 'rb')) 

db_file="../loopermanscrapy/loops.db"
conn = sqlite3.connect(db_file)
cur = conn.cursor()

# Iterate over all instances in all datasets and for all methods
print('Computing confidence measure values for all sounds in all datasets and for all methods...')
n_annotated = 0

conf_comparison = json.load(open("./conf_comparison.json"))

for key in percival_bpms:
    #try:
    
    conf_measure = compute_confidence_measure(
        int(round(percival_bpms[key]["Percival14"]["bpm"])),
        durations[key]['durations']['length_samples'],
        durations[key]['durations']['start_effective_duration'],
        durations[key]['durations']['end_effective_duration']
    )
    conf_comparison[key] = [conf_comparison[key], {'percival': {'bpm' :  int(round(percival_bpms[key]["Percival14"]["bpm"])), 'confidence' : conf_measure}}]
    n_annotated += 1
    if n_annotated % 50 == 0:
        print(n_annotated)
    #except Exception as e:
        #print(e)
        #continue

"""
for key in percival_bpms:
    try:
        cur.execute("SELECT BPM FROM LOOPS WHERE OUTPUT_FILE LIKE '%s'" % key.replace(".wav",".mp3"))
        bpm = cur.fetchone()[0]


        conf_measure = compute_confidence_measure(
            int(round(int(bpm))),
            durations[key]['durations']['length_samples'],
            durations[key]['durations']['start_effective_duration'],
            durations[key]['durations']['end_effective_duration']
        )
        conf_comparison.setdefault(key,[]).append({'annotated_bpm': {'bpm' :  bpm, 'confidence' : conf_measure}})
        n_annotated += 1
        if n_annotated % 50 == 0:
            print(n_annotated)
    except KeyError:
        continue
"""

conn.close()
json.dump(conf_comparison, open("conf_comparison.json", 'w'))
print("Annotated " + str(n_annotated) + " loops")
