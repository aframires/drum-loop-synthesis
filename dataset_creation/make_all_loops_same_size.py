import numpy as np
import essentia.standard as es
import soundfile as sf


import os
import json
import math
import subprocess
import random



sampleRate = 16000
bar_size = 29538
BASE_PATH = './all_drum_loops/ts/'
CHOPPED_PATH = './all_drum_loops/ts-d-c/'
EQ_PATH = './all_drum_loops/ts-d-c/eq/'
EQ_NEW_PATH = './all_drum_loops/ts-d-c/eq_new/'

def create_histogram_durations():
    histogram_durations = {}

    for loop in os.listdir(BASE_PATH):
        audio_file=es.MonoLoader(filename=BASE_PATH+loop,sampleRate=sampleRate)
        audio = audio_file.compute()
        if len(audio) in histogram_durations.keys():
            histogram_durations[len(audio)] = histogram_durations[len(audio)] + 1
        else:
            histogram_durations[len(audio)] = 1

    json.dump(histogram_durations, open("histogram_durations.json", 'w'))

def chop_files_to_1_bar():
    count = 0
    for loop in os.listdir(BASE_PATH):
        audio_file=es.MonoLoader(filename=BASE_PATH+loop,sampleRate=sampleRate)
        audio = audio_file.compute()
        count +=1
        if count % 50 == 0:
            print(count)
        if len(audio) <= bar_size:
            if len(audio) > bar_size/2:
                audio_to_save = np.zeros(bar_size)
                audio_to_save[:len(audio)] = audio
                sf.write(CHOPPED_PATH + loop, audio_to_save,16000)
        else:
            chops = range(math.ceil(len(audio)/bar_size))
            for chop in chops:
                if chop == chops[-1]:
                    if len(audio[chop*bar_size:]) > bar_size/2:
                        audio_to_save = np.zeros(bar_size)
                        audio_to_save[:len(audio[chop*bar_size:])] = audio[chop*bar_size:]
                        sf.write(CHOPPED_PATH + loop, audio_to_save,16000)
                else:
                    audio_to_save = audio[chop*bar_size:(chop+1)*bar_size]
                    sf.write(CHOPPED_PATH + str(chop+1) + loop, audio_to_save,16000)


def transcribe_loops():
    from ADTLib import ADT

    TS_D_C_PATH = './all_drum_loops/ts-d-c/'
    loops = os.listdir(TS_D_C_PATH)
    ANALYSIS_PATH = TS_D_C_PATH + 'analysis/'
    BACKUP_ANALYSIS_PATH = ANALYSIS_PATH + 'backup/'
    loops_full = [TS_D_C_PATH + s for s in loops]
    for loop in loops_full:
        [Onsets, ActivationFunctions] = ADT([loop], output_act='yes', tab='no', save_dir=ANALYSIS_PATH)
        np.savetxt(BACKUP_ANALYSIS_PATH + os.path.split(loop)[1] + "onsets.txt", Onsets)
        np.savetxt(BACKUP_ANALYSIS_PATH + os.path.split(loop)[1] + "actFun.txt", ActivationFunctions)


def transcribe_loops_cmd():
    TS_D_C_PATH = './all_drum_loops/ts-d-c/'
    loops = os.listdir(TS_D_C_PATH)
    for i in range(0, len(loops), 20):
        loops_to_proc = [TS_D_C_PATH + s for s in loops[i:i+20]]
        subprocess.run("ADT -od "+ TS_D_C_PATH +"analysis/ -o yes -ot no " + " ".join(loops_to_proc),shell=True)

def timestretch_loops():
    loops =  json.load(open('loops_bpm_to_timestretch.json', 'rb'))

    for loop in loops:
        try:
            subprocess.run("rubberband --tempo " + loops[loop]['bpm'] + ":130 ./all_drum_loops/wav/"+loop+" ./all_drum_loops/ts/"+loop,shell=True)

        except Exception as e:
            print(e)
            print("rubberband --tempo " + loops[loop]['bpm'] + ":130 ./all_drum_loops/wav/"+loop+" ./all_drum_loops/ts/"+loop)

def filter_loops():

    loops = os.listdir(CHOPPED_PATH)
    proc_loops = os.listdir(EQ_NEW_PATH)

    lp_filter = es.LowPass(cutoffFrequency=90,sampleRate=sampleRate)
    bp_filter = es.BandPass(bandwidth=100 ,cutoffFrequency=280,sampleRate=sampleRate)
    hp_filter = es.HighPass(cutoffFrequency=9000,sampleRate=sampleRate)
    i=0
    for loop in loops:
        i=i+1
        if i % 50 == 0:
            print(str(i))
        if ".wav" in loop:
            if ("bpf_" + loop) not in proc_loops:
                audio_file=es.MonoLoader(filename=CHOPPED_PATH+loop,sampleRate=sampleRate)
                #lpf_audio = lp_filter(audio_file())
                bpf_audio = bp_filter(audio_file())
                #hpf_audio = hp_filter(audio_file())
                #sf.write(EQ_PATH + "lpf_" + loop, lpf_audio, sampleRate)
                sf.write(EQ_NEW_PATH + "bpf_" + loop, bpf_audio, sampleRate)
                #sf.write(EQ_PATH + "hpf_" + loop, hpf_audio, sampleRate)

def split_dataset():
    files = {}
    number_of_loops = 0
    for loop in os.listdir(BASE_PATH):
        audio_file=es.MonoLoader(filename=BASE_PATH+loop,sampleRate=sampleRate)
        audio = audio_file.compute()

        if len(audio) <= bar_size:
            size = 1
            files[loop] = { "size":size,
                            "resulting_files":[loop]}
        else:
            filenames = []
            chops = range(math.ceil(len(audio)/bar_size))
            for chop in chops:
                filenames.append(str(chop+1) + loop)
            size = chops[-1]
            files[loop] = { "size":size,
                            "resulting_files":filenames}
        number_of_loops = number_of_loops + size

    with open('loop_sizes.txt', 'w') as filehandle:
        json.dump(files, filehandle)

    count = 0
    loops_test = []
    loops_train = []
    loops = random.sample(list(files.keys()),len(list(files.keys())))
    i = 0
    while count < (0.1 * number_of_loops):
        count = count + files[loops[i]]["size"]
        print(count)
        loops_test.extend(files[loops[i]]["resulting_files"])
        files.pop(loops[i])
        i=i+1
    
    for j in range(i,len(loops)):
        loops_train.extend(files[loops[j]]["resulting_files"])

    
    with open('train_split.json', 'w') as filehandle:
        json.dump(loops_train, filehandle)

    with open('test_split.json', 'w') as filehandle:
        json.dump(loops_test, filehandle)


def filter_loops_eval():
    loops_paths = [ "icassp2021_outputs/outputs_stft_coherence/",
                    "icassp2021_outputs/outputs_wavstft_coherence/"]
    
    lp_filter = es.LowPass(cutoffFrequency=90,sampleRate=sampleRate)
    bp_filter = es.BandPass(bandwidth=100 ,cutoffFrequency=280,sampleRate=sampleRate)
    hp_filter = es.HighPass(cutoffFrequency=9000,sampleRate=sampleRate)
    for path in loops_paths:
        loops = os.listdir(path)
        for loop in loops:
            if ".wav" in loop:
                audio_file=es.MonoLoader(filename=path+loop,sampleRate=sampleRate)
                if "lpf" in loop:
                    lpf_audio = lp_filter(audio_file())
                    sf.write(path + "eq/" + loop, lpf_audio, sampleRate)
                if "bpf" in loop:
                    bpf_audio = bp_filter(audio_file())
                    sf.write(path + "eq/" + loop, bpf_audio, sampleRate)
                if "hpf" in loop:
                    hpf_audio = hp_filter(audio_file())
                    sf.write(path + "eq/" + loop, hpf_audio, sampleRate)

        
filter_loops_eval()


 
    