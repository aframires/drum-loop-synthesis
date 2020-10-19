import csv
import os
import subprocess
import math
import json
import glob

import numpy as np
import essentia.standard as es
import soundfile as sf
import h5py
import madmom


FSLD_PATH = "/mnt/f/code/Research/freesound-loop-annotator/"
FLSD_AUDIO_PATH = os.path.join(FSLD_PATH,"static/FSL10K/audio/wav/")
FLSD_ANNOTATIONS_FILE = os.path.join(FSLD_PATH,"data_analysis/annotation_dict.json")
EXCEL_PATH = "/mnt/f/code/Research/NeuroLoops/Balanced_Loops/csv/"
CHOPPED_AUDIO_DIR = "/mnt/f/code/Research/NeuroLoops/Balanced_Loops/chooped_audios/"
TS_AUDIO_DIR = "/mnt/f/code/Research/NeuroLoops/Balanced_Loops/ts_audios/"
FLSD_TS_AUDIO_DIR = "/mnt/f/code/Research/NeuroLoops/Freesound_Loops/ts_audios/"
FLSD_CHOPPED_AUDIO_DIR = "/mnt/f/code/Research/NeuroLoops/Freesound_Loops/chopped_audios/"
SOURCE_AUDIO_DIR = "/mnt/f/code/Research/NeuroLoops/all_drum_loops/wav/"
SPEC_PATH = "/mnt/f/code/Research/NeuroLoops/Balanced_Loops/specs/"
FLSD_SPEC_PATH = "/mnt/f/code/Research/NeuroLoops/Freesound_Loops/specs/"
PATH_TO_METADATA = os.path.join(FSLD_PATH, 'static/FSL10K/fs_analysis/')

sampleRate = 44100
final_bpm = 130
four_bar_length = int( (60/130) * 4 * 44100)

def check_drums_only(ann):
    if ann["discard"] == "False":
        if  ann["instrumentation_percussion"] == "True" and \
            ann["instrumentation_vocal"] == "False" and \
            ann["instrumentation_bass"] == "False" and \
            ann["instrumentation_melody"] == "False" and \
            ann["instrumentation_chords"] == "False" and \
            ann["instrumentation_fx"] == "False":

            return True
        else:
            return False
    else:
        return False



def get_drum_annotations():
    with open(FLSD_ANNOTATIONS_FILE, "r") as ann_file:
        annotations = json.load(ann_file)
    
    annotations_drums = {}

    for sound in annotations:
        if len(annotations[sound]) == 1:
            if check_drums_only(annotations[sound][0]):
                annotations_drums[sound] = annotations[sound]
        if len(annotations[sound]) == 2:
            if check_drums_only(annotations[sound][0]):
                if check_drums_only(annotations[sound][1]):
                    annotations_drums[sound] = annotations[sound]

    return annotations_drums

def ts_fsld(ann):
    for sound in ann:
        tempo = get_tempo(ann[sound])
        if tempo != 0:
            filename = get_filename_from_id(sound)
            timestretch_file(str(tempo),filename,None,mode='FREESOUND')

def get_tempo(ann):
    try:
        tempo = int(ann[0]['bpm'])
        if 120 <= tempo <= 140:
            return tempo
        elif 60 <= tempo <= 70:
            return tempo*2
        else:
            return 0
    except:
        return 0

def get_filename_from_id(sound_id):
    metadata = json.load(open(PATH_TO_METADATA + sound_id + '.json', 'rb'))
    ac_analysis_filename = metadata["preview_url"]
    base_name = ac_analysis_filename[ac_analysis_filename.rfind("/"):ac_analysis_filename.find("-hq")]
    audio_file = glob.glob(FLSD_AUDIO_PATH + base_name + '*')[0]
    audio_file = os.path.basename(audio_file)
    return audio_file

def chop_folder(folder):
    count=0
    for loop in os.listdir(folder):
        try:
            loop_path = folder + loop
            audio_file=es.MonoLoader(filename=loop_path,sampleRate=sampleRate)
            audio = audio_file.compute()
            count +=1

            if count % 50 == 0:
                print(count)

            chops = range(math.ceil(len(audio)/four_bar_length))
            for chop in chops:
                if chop == chops[-1]:
                    audio_to_save = np.zeros(four_bar_length)
                    audio_to_save[:len(audio[chop*four_bar_length:])] = audio[chop*four_bar_length:]
                    sf.write(FLSD_CHOPPED_AUDIO_DIR + str(chop+1) + loop, audio_to_save,sampleRate)
                else:
                    audio_to_save = audio[chop*four_bar_length:(chop+1)*four_bar_length]
                    sf.write(FLSD_CHOPPED_AUDIO_DIR + str(chop+1) + loop, audio_to_save,sampleRate)
        except:
            print(loop + " failed!")

def chop_excel(csv_filename):
    count = 0
    folder = get_folder_name(csv_filename)
    with open(EXCEL_PATH + csv_filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        for row in csvreader:
            loop_info = get_loop_info(row)
            loop_path = TS_AUDIO_DIR + folder + loop_info['filename']
            audio_file=es.MonoLoader(filename=loop_path,sampleRate=sampleRate)
            audio = audio_file.compute()
            count +=1

            if count % 50 == 0:
                print(count)

            if csv_filename.find("_1stBar") != -1:
                audio_to_save = np.zeros(four_bar_length)
                if len(audio) <= four_bar_length:
                    audio_to_save[:len(audio)] = audio
                else:
                    audio_to_save = audio[0:four_bar_length]
                sf.write(CHOPPED_AUDIO_DIR + folder + loop_info['filename'], audio_to_save,sampleRate)

            if csv_filename.find("_Chop") != -1:
                chops = range(math.ceil(len(audio)/four_bar_length))
                for chop in chops:
                    if chop == chops[-1]:
                        audio_to_save = np.zeros(four_bar_length)
                        audio_to_save[:len(audio[chop*four_bar_length:])] = audio[chop*four_bar_length:]
                        sf.write(CHOPPED_AUDIO_DIR + folder + str(chop+1) + loop_info['filename'], audio_to_save,sampleRate)
                    else:
                        audio_to_save = audio[chop*four_bar_length:(chop+1)*four_bar_length]
                        sf.write(CHOPPED_AUDIO_DIR + folder + str(chop+1) + loop_info['filename'], audio_to_save,sampleRate)




def timestretch_excel(csv_filename):
    count = 0
    folder = get_folder_name(csv_filename)
    with open(EXCEL_PATH + csv_filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=";")
        for row in csvreader:
            if count % 50 == 0:
                print(count)

            loop_info = get_loop_info(row)
            timestretch_file(loop_info['bpm'],loop_info['filename'], folder)
            count += 1



def get_folder_name(csv_filename):
    return csv_filename.replace('.csv','') + '/'

def timestretch_file(orig_tempo, file_name, subfolder, mode='LOOPERMAN'):
    if mode=='LOOPERMAN':
        try:
            command = "rubberband --tempo " + orig_tempo + ":" + str(final_bpm) + " -c 6 " + SOURCE_AUDIO_DIR+file_name + " " + TS_AUDIO_DIR+subfolder+file_name
            subprocess.run(command, shell=True)

        except Exception as e:
            print(e)
            print(command)
    else:
        command = "rubberband --tempo " + orig_tempo + ":" + str(final_bpm) + " -c 6 " + FLSD_AUDIO_PATH + file_name + " " + FLSD_TS_AUDIO_DIR+file_name
        print(command)
        try:
            subprocess.run(command, shell=True,timeout=30)

        except Exception as e:
            print(e)
            print(command)        

def get_loop_info(row):
    loop_info = {
        'filename'  : row[5] + '.wav',
        'bpm'       : row[7],
        'num_4bars' : row[12]
    }
    return loop_info

def create_spec(audio_file):
    return madmom.audio.spectrogram.Spectrogram(audio_file, frame_size=2048, hop_size=512, fft_size=2048,num_channels=1)

def timestretch_and_chop_excel():
    for excel_file in os.listdir(EXCEL_PATH):
        timestretch_excel(excel_file)
        chop_excel(excel_file)

def spectograms(src_folder,dest_folder):
    for root, dirs, files in os.walk(src_folder):
        for name in files:
            if name.endswith(".wav"):
                spec = create_spec(os.path.join(root, name))
                with h5py.File(dest_folder + name +'.h5', 'w') as hf:
                    hf.create_dataset(name, data=spec)

def fsld():
    ann = get_drum_annotations()
    ts_fsld(ann)


spectograms(FLSD_CHOPPED_AUDIO_DIR,FLSD_SPEC_PATH)