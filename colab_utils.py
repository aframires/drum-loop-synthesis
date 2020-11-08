from ADTLib import ADT
import essentia as es
import numpy as np
import soundfile as sf
import timbral_models

import ntpath

def file_to_hpcp(loop):
    loop = es.array(loop)

    windowing = es.standard.Windowing(type='blackmanharris62')
    spectrum = es.standard.Spectrum()
    spectral_peaks = es.standard.SpectralPeaks(orderBy='magnitude',
                                    magnitudeThreshold=0.001,
                                    maxPeaks=20,
                                    minFrequency=20,
                                    maxFrequency=8000)
    hpcp = es.standard.HPCP(maxFrequency=8000)
    spec_group = []
    hpcp_group = []
    for frame in es.standard.FrameGenerator(loop,frameSize=1024,hopSize=512):
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


  loop_basename = ntpath.basename(loop)
  lpf_audio = lp_filter(audio_file())
  bpf_audio = bp_filter(audio_file())
  hpf_audio = hp_filter(audio_file())

  sf.write("analysis/lpf_" + loop_basename, lpf_audio, sampleRate)
  sf.write("analysis/bpf_" + loop_basename, bpf_audio, sampleRate)
  sf.write("analysis/hpf_" + loop_basename, hpf_audio, sampleRate)

  features_kick = timbral_models.timbral_extractor("analysis/lpf_" + loop_basename, clip_output=True)
  features_snare = timbral_models.timbral_extractor("analysis/bpf_" + loop_basename, clip_output=True)
  features_hh = timbral_models.timbral_extractor("analysis/hpf_" + loop_basename, clip_output=True)
  
  hpcp = file_to_hpcp(loop)

  return pattern,hpcp,features_kick,features_snare,features_hh