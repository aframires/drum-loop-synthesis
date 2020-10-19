from ac_utils.sound import load_audio_file
import numpy as np
import essentia.standard as estd
import essentia
import settings
import subprocess

SOUND_FILE_KEY = 'file_path'


def algorithm_rhythm_essentia_basic(sound):
    """
    Estimates bpm of given audio file using Zapata14 and Degara12.
    * Zapata14: Jose R Zapata, Matthew E P Davies, and Emilia Gomez. Multi-Feature Beat Tracking. IEEE/ACM
    Transactions on Audio, Speech, and Language Processing, 22(4):816-825, 2014.
    * Degara12: Norberto Degara,Enrique Argones Rua,Antonio Pena, Soledad Torres-Guijarro, Matthew EP Davies, and Mark D
    Plumbley. Reliability-Informed Beat Track- ing of Musical Signals. IEEE Transactions on Audio, Speech, and
    Language Processing, 20(1):290-301, 2012.
    :param sound: sound dictionary from dataset
    :return: dictionary with results per different methods
    """
    results = dict()
    audio = load_audio_file(file_path=sound[SOUND_FILE_KEY], sample_rate=44100)

    # Method RhythmExtractor2013 - multifeature
    rhythm_extractor_2013 = estd.RhythmExtractor2013()
    bpm, ticks, confidence, _, bpm_intervals = rhythm_extractor_2013(audio)
    results['Zapata14'] = {'bpm': bpm, 'confidence': float(confidence)}

    # Method RhythmExtractor2013 - degara
    rhythm_extractor_2013 = estd.RhythmExtractor2013(method='degara')
    bpm, ticks, confidence, _, bpm_intervals = rhythm_extractor_2013(audio)
    results['Degara12'] = {'bpm': bpm}
    return results

def algorithm_rhythm_percival_essentia(sound):
    results = dict()
    audio = load_audio_file(file_path=sound[SOUND_FILE_KEY], sample_rate=44100)
    tempo_estimator = estd.PercivalBpmEstimator()
    bpm = tempo_estimator(audio)
    results['Percival14_essentia'] = {'bpm': bpm}

    return results

def algorithm_rhythm_percival14(sound):
    """
    Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation
    with pulses. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765-1776.
    :param sound: sound dictionary from dataset
    :return: dictionary with results
    """
    results = dict()
    sample_rate = 44100
    audio = load_audio_file(file_path=sound[SOUND_FILE_KEY], sample_rate=sample_rate)

    # Convert to mono and add silence at the beginning if sound is shorter than 6s (otherwise algorithm fails)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    min_duration = 6  # In seconds
    minimum_size = min_duration * sample_rate
    if audio.shape[0] < minimum_size:
        audio = np.append(np.zeros(minimum_size - audio.shape[0]), audio)
    try:
        from algorithms.Percival14.defs_class import Defs
        from algorithms.Percival14.onset_strength import onset_strength_signal
        from algorithms.Percival14.beat_period_detection import beat_period_detection
        from algorithms.Percival14.accumulator_overall import accumulator_overall
        defs = Defs()
        oss_sr, oss_data = onset_strength_signal(defs, sample_rate, audio, plot=False)
        tempo_lags = beat_period_detection(defs, oss_sr, oss_data, plot=False)
        bpm = accumulator_overall(defs, tempo_lags, oss_sr)
        results['Percival14'] = {'bpm': bpm}
    except ValueError:
        pass
    return results

def algorithm_rhythm_percival14_mod(sound):
    """
    Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation
    with pulses. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765-1776.
    :param sound: sound dictionary from dataset
    :return: dictionary with results
    """
    results = dict()
    sample_rate = 44100
    audio = load_audio_file(file_path=sound[SOUND_FILE_KEY], sample_rate=sample_rate)

    # Convert to mono and add silence at the beginning if sound is shorter than 6s (otherwise algorithm fails)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    try:
        from algorithms.Percival14Mod.defs_class import Defs
        from algorithms.Percival14Mod.onset_strength import onset_strength_signal
        from algorithms.Percival14Mod.beat_period_detection import beat_period_detection
        from algorithms.Percival14Mod.accumulator_overall import accumulator_overall
        defs = Defs()
        oss_sr, oss_data = onset_strength_signal(defs, sample_rate, audio, plot=False)
        tempo_lags = beat_period_detection(defs, oss_sr, oss_data, plot=False)
        bpm = accumulator_overall(defs, tempo_lags, oss_sr)
        results['Percival14Mod'] = {'bpm': bpm}
        #tempo_lags = beat_period_detection(defs, oss_sr, oss_data, plot=False, limit_pulse_trains=True)
        #bpm = accumulator_overall(defs, tempo_lags, oss_sr)
        #results['Percival14ModLimitPulseTrains'] = {'bpm': bpm}
        #tempo_lags = beat_period_detection(defs, oss_sr, oss_data, plot=False, skip_calc_pulse_trains=True)
        #bpm = accumulator_overall(defs, tempo_lags, oss_sr)
        #results['Percival14ModSkipPulseTrains'] = {'bpm': bpm}
    except ValueError:
        pass
    return results


def algorithm_rhythm_gkiokas12(sound):
    """
    Gkiokas, A., Katsouros, V., Carayannis, G., & Stafylakis, T. (2012). Music Tempo Estimation and Beat Tracking By
    Applying Source Separation and Metrical Relations. In International Conference on Acoustics, Speech and Signal
    Processing (ICASSP) (Vol. 7, pp. 421-424).
    :param sound: sound dictionary from dataset
    :return: dictionary with results
    """
    results = dict()
    command = ['octave', '--eval',
               'cd %s/algorithms/Gkiokas12; extractTempo \'%s\'' % (settings.PROJECT_PATH, sound[SOUND_FILE_KEY])]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err:
        print('\n' + str(err) + '\n')
    try:
        bpm = float(out.split("=  ")[1].split('\n')[0])
        results['Gkiokasq12'] = {'bpm': bpm}
    except IndexError:
        pass

    return results


def algorithm_rhythm_madmom(sound):
    """
    Accurate Tempo Estimation based on Recurrent Neural Networks and Resonating Comb Filters. Sebastian Bock, Florian
    Krebs and Gerhard Widmer. Proceedings of the 16th International Society for Music Information Retrieval Conference
    (ISMIR), 2015
    :param sound: sound dictionary from dataset
    :return: dictionary with results
    """
    results = dict()
    command = ['TempoDetector', '--mirex', '--method', 'comb', 'single']
    p = subprocess.Popen(command + [sound[SOUND_FILE_KEY]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err:
        print('\n' + str(err)+ '\n')
        print(sound[SOUND_FILE_KEY].replace(" ", "\ "))
        return results
    result = out.split(b'\t')
    t1 = float(result[0])
    t2 = float(result[1])
    st1 = float(result[2])
    if st1 >= 0.5:  # Select most probable tempo
        bpm = t1
    else:
        bpm = t2
    results['Bock15'] = {'bpm': bpm}
    return results


def algorithm_rhythm_madmom_acf(sound):
    """
    Accurate Tempo Estimation based on Recurrent Neural Networks and Resonating Comb Filters. Sebastian Bock, Florian
    Krebs and Gerhard Widmer. Proceedings of the 16th International Society for Music Information Retrieval Conference
    (ISMIR), 2015
    :param sound: sound dictionary from dataset
    :return: dictionary with results
    """
    results = dict()
    command = ['TempoDetector', '--mirex', '--method', 'acf', 'single']
    p = subprocess.Popen(command + [sound[SOUND_FILE_KEY]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err:
        print('\n' + str(err) + '\n')
        print(sound[SOUND_FILE_KEY].replace(" ", "\ "))
        return results
    result = out.split(b'\t')
    t1 = float(result[0])
    t2 = float(result[1])
    st1 = float(result[2])
    if st1 >= 0.5:  # Select most probable tempo
        bpm = t1
    else:
        bpm = t2
    results['Bock15ACF'] = {'bpm': bpm}
    return results

def algorithm_rhythm_madmom_dbn(sound):
    """
    Accurate Tempo Estimation based on Recurrent Neural Networks and Resonating Comb Filters. Sebastian Bock, Florian
    Krebs and Gerhard Widmer. Proceedings of the 16th International Society for Music Information Retrieval Conference
    (ISMIR), 2015
    :param sound: sound dictionary from dataset
    :return: dictionary with results
    """
    results = dict()
    command = ['TempoDetector', '--mirex', '--method', 'dbn', 'single']
    p = subprocess.Popen(command + [sound[SOUND_FILE_KEY]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err:
        print('\n' + str(err) + '\n')
        print(sound[SOUND_FILE_KEY].replace(" ", "\ "))
        return results
    result = out.split(b'\t')
    t1 = float(result[0])
    t2 = float(result[1])
    st1 = float(result[2])
    if st1 >= 0.5:  # Select most probable tempo
        bpm = t1
    else:
        bpm = t2
    results['Bock15DBN'] = {'bpm': bpm}
    return results
