from shutil import copyfile
import sqlite3
import json
import essentia.standard as estd



f = open("./drum_loops_all_filenames.txt","r")
loops = f.readlines()
f.close()



def copy_loops_to_folder():

    LOOPS_PATH = "../loopermanscrapy/looperman_audios/"
    SELECTED_LOOPS_PATH = "./all_drum_loops/"

    i = 0
    for loop in loops:
        i+=1
        if i % 50 == 0:
            print(i)
        file_to_move = loop.rstrip()
        try:
            copyfile(LOOPS_PATH + file_to_move, SELECTED_LOOPS_PATH + file_to_move)
        except:
            print(file_to_move)

def analyze_loops(db_file="../loopermanscrapy/loops.db"):
    
    loop_data = {}

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    sample_rate = 44100
    for loop in loops:
        cur.execute("SELECT BPM FROM LOOPS WHERE OUTPUT_FILE LIKE '%s'" % loop.rstrip())
        bpm = cur.fetchone()[0]
        
        audio_file = estd.EasyLoader(filename="./drum_loops/"+loop.rstrip(), sampleRate=sample_rate)
        audio = audio_file.compute()

        beat_duration = (60.0 * sample_rate)/ int(bpm)
        L = [beat_duration * n for n in range(1, 128)]
        thr_lambda = 0.5 * beat_duration
        la = audio.shape[0]
        delta = min([abs(l - la) for l in L])
        if delta > thr_lambda:
            ann_confidence = 0.0
        else:
            ann_confidence = (1.0 - float(delta) / thr_lambda)
        
        duration = la/sample_rate


        loop_data[loop.rstrip()] = {"bpm_annotated":bpm, "duration":duration, "annotated_confidence":ann_confidence, "bpm_percival":bpm_percival, "confidence": confidence}


    conn.close()
    json.dump(loop_data, open("loop_data.json", 'w'))

def create_metadata_file(db_file="../loopermanscrapy/loops.db"):

    SELECTED_LOOPS_PATH = "./all_drum_loops/wav/"

    loop_data = {}

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    sample_rate = 44100
    for loop in loops:
        cur.execute("SELECT BPM FROM LOOPS WHERE OUTPUT_FILE LIKE '%s'" % loop.rstrip())
        bpm = cur.fetchone()[0]
        
        #audio_file = estd.EasyLoader(filename="./all_drum_loops/"+loop.rstrip(), sampleRate=sample_rate)
        #audio = audio_file.compute()
        #try:
        #    beat_duration = (60.0 * sample_rate)/ int(bpm)
        #except ZeroDivisionError:
        #    beat_duration = 0
        #L = [beat_duration * n for n in range(1, 128)]
        #thr_lambda = 0.5 * beat_duration
        #la = audio.shape[0]
        #delta = min([abs(l - la) for l in L])
        #if delta > thr_lambda:
        #    ann_confidence = 0.0
        #else:
        #    ann_confidence = (1.0 - float(delta) / thr_lambda)
        
        #duration = la/sample_rate
        

        loop_data[loop.rstrip()] = {"bpm_annotated":bpm, "wav_sound_path": SELECTED_LOOPS_PATH + loop.rstrip(), "id":loop.rstrip() }


    conn.close()
    json.dump(loop_data, open("metadata.json", 'w'))


create_metadata_file()