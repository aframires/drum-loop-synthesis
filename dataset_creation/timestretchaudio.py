import pyrubberband
import subprocess
import json

loops =  json.load(open('loops_bpm_to_timestretch.json', 'rb'))

for loop in loops:
    try:
        subprocess.run("rubberband --tempo " + loops[loop]['bpm'] + ":130 ./all_drum_loops/wav/"+loop+" ./all_drum_loops/ts/"+loop,shell=True)

    except Exception as e:
        print(e)
        print("rubberband --tempo " + loops[loop]['bpm'] + ":130 ./all_drum_loops/wav/"+loop+" ./all_drum_loops/ts/"+loop)
