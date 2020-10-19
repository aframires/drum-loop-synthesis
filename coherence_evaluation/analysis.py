import os
import json
from collections import defaultdict

def analyze_results(folder, output_files_base_name):
    # format the analysis
    files_features = defaultdict(lambda: defaultdict(dict))
    filenames = [filename for filename in os.listdir(folder)
                 if filename.startswith("op_")
                 and len(filename.split('_')) > 4]
    failed_original_filenames = set()
    for filename in filenames:
        try:
            analysis = json.load(open(folder + filename, 'rb'))
            original_filename = '{}_{}'.format(filename.split('_')[1], filename.split('_')[2])
            timbral_feature = filename.split('_')[4]
            timbral_feature_strengh = filename.split('_')[5]
            files_features[original_filename][timbral_feature][timbral_feature_strengh] = analysis[timbral_feature]
        except:
            failed_original_filenames.add(original_filename)
    print(len(failed_original_filenames))
    json.dump(files_features, open('results/{}_analysis_features.json'.format(output_files_base_name), 'w'))
    # count the number of valid comparison
    # type1: high > low
    # type2: high > mid
    # type3: mid > low
    # global
    type1 = []
    type2 = []
    type3 = []
    for file, item in files_features.items():
        for feature, value in item.items():
            if 'high' in value and 'mid' in value and 'low' in value:
                type1.append(value['high']>value['low'])
                type2.append(value['high']>value['mid'])
                type3.append(value['mid']>value['low'])
    # per features
    type1_per_features = defaultdict(list)
    type2_per_features = defaultdict(list)
    type3_per_features = defaultdict(list)
    for file, item in files_features.items():
        for feature, value in item.items():
            if 'high' in value and 'mid' in value and 'low' in value:
                type1_per_features[feature].append(value['high']>value['low'])
                type2_per_features[feature].append(value['high']>value['mid'])
                type3_per_features[feature].append(value['mid']>value['low'])
    type1_per_features_ratio = {}
    type2_per_features_ratio = {}
    type3_per_features_ratio = {}
    for feature, values in type1_per_features.items():
        type1_per_features_ratio[feature] = sum(values) / float(len(values))
    for feature, values in type2_per_features.items():
        type2_per_features_ratio[feature] = sum(values) / float(len(values))
    for feature, values in type3_per_features.items():
        type3_per_features_ratio[feature] = sum(values) / float(len(values))
    stats = {
        'ratio high > low (type1)': type1_per_features_ratio,
        'ratio high > mid (type2)': type2_per_features_ratio,
        'ratio mid > low (type3)': type3_per_features_ratio,
    }
    json.dump(stats, open('results/{}_stats_comparison_feature_values.json'.format(output_files_base_name), 'w'), indent=4, sort_keys=True)


if __name__ == "__main__":
    folders_and_model_names = [
        ('icassp2021/outputs_multi_coherence_analysis/', 'multi'),
        ('icassp2021/outputs_multi_noenv_coherence_analysis/', 'multi_noenv'),
        ('icassp2021/outputs_stft_coherence_analysis/', 'stft'),
        ('icassp2021/outputs_wavstft_coherence_analysis/', 'wavstft'),
    ]
    for folder, model_name in folders_and_model_names:
        analyze_results(folder, model_name)