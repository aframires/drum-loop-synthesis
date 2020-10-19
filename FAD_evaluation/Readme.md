To run:

    source venv/bin/activate

    Add files to file lists:
        ls --color=never /mnt/f/Code/Research/NeuroLoops/all_drum_loops/ts-d-c/*  > test_audio/test_files_background.cvs
        ls --color=never test_audio/test1/*  > test_audio/test_files_test1.cvs
    
    Compute embeddings:
        python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_background.cvs --stats stats/background_stats
        python -m frechet_audio_distance.create_embeddings_main --input_files test_audio/test_files_test1.cvs --stats stats/test1_stats

    Calculate FAD 
        $ python -m frechet_audio_distance.compute_fad --background_stats stats/background_stats --test_stats stats/test1_stats