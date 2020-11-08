from ADTLib import ADT
import essentia.standard as es
import numpy as np
import soundfiles as sf
import timbral_models

def file_to_hpcp(loop):
    loop = es.array(loop)

    windowing = es.Windowing(type='blackmanharris62')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='magnitude',
                                    magnitudeThreshold=0.001,
                                    maxPeaks=20,
                                    minFrequency=20,
                                    maxFrequency=8000)
    hpcp = es.HPCP(maxFrequency=8000)
    spec_group = []
    hpcp_group = []
    for frame in es.FrameGenerator(loop,frameSize=1024,hopSize=512):
        windowed = windowing(frame)
        fft = spectrum(windowed)
        frequencies, magnitudes = spectral_peaks(fft)
        final_hpcp = hpcp(frequencies, magnitudes)
        spec_group.append(fft)
        hpcp_group.append(final_hpcp)

    mean_hpcp = np.mean(np.array(hpcp_group).T, axis = 1)
    #normalize to 1
    mean_hpcp = mean_hpcp/mean_hpcp.max()


    return mean_hpcp

def analysis_function(loop,sampleRate):
  lp_filter = es.LowPass(cutoffFrequency=90,sampleRate=sampleRate)
  bp_filter = es.BandPass(bandwidth=100 ,cutoffFrequency=280,sampleRate=sampleRate)
  hp_filter = es.HighPass(cutoffFrequency=9000,sampleRate=sampleRate)

  [_, pattern] = ADT([loop], output_act='yes', tab='no', save_dir="analysis/")

  audio_file=es.MonoLoader(filename=loop,sampleRate=sampleRate)

  lpf_audio = lp_filter(audio_file())
  bpf_audio = bp_filter(audio_file())
  hpf_audio = hp_filter(audio_file())

  sf.write("audio_analysis/lpf_" + loop, lpf_audio, sampleRate)
  sf.write("audio_analysis/bpf_" + loop, bpf_audio, sampleRate)
  sf.write("audio_analysis/hpf_" + loop, hpf_audio, sampleRate)

  features_kick = timbral_models.timbral_extractor("audio_analysis/lpf_" + loop, clip_output=True)
  features_snare = timbral_models.timbral_extractor("audio_analysis/bpf_" + loop, clip_output=True)
  features_hh = timbral_models.timbral_extractor("audio_analysis/hpf_" + loop, clip_output=True)
  
  hpcp = file_to_hpcp(loop)

  return pattern,hpcp,features_kick,features_snare,features_hh