<h1>LOOPNET: MUSICAL LOOP SYNTHESIS CONDITIONED ON INTUITIVE MUSICALPARAMETERS</h1>

<h2>Pritish Chandna, António Ramires, Xavier Serra, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for loop synthesis
<h3>Installation</h3>
To install, clone the repository and use <pre><code>pip install -r requirements.txt </code></pre> to install the packages required.
The pretrained <a href="https://drive.google.com/file/d/1aak1krpGQWIWesI0gOVm3WZ0XHfYW7qy/view?usp=sharing" rel="nofollow"> model weights</a> can be downloaded along with the <a href="https://drive.google.com/file/d/1Tj29ktt4KDLNmqhEaxhvUuF-qM4LwO37/view?usp=sharing" rel="nofollow"> validation hdf5 file</a>. The path to the unzipped model weights and the validation file need to be set in the command line arguments for the main function.

 The main code is in the *main.py* file.  
 <pre><code>
usage: main.py [-h] [--model MODEL] [--log_dir LOG_DIR] [--val_file VAL_FILE]
               [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Models to use, must be in multi_env, multi, wavespec,
                        wav or spec
  --log_dir LOG_DIR     The directory where the models are saved
  --val_file VAL_FILE   Path to the file containing validation features
  --output_dir OUTPUT_DIR
                        Directory to save the outputs in
  </code></pre> 

 <h2>Acknowledgments</h2>
The TITANX used for this research was donated by the NVIDIA Corporation. This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> (H2020 770376) European project.