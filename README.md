[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aframires/drum-loop-synthesis/blob/main/LOOPNET.ipynb)


<h1>LOOPNET: MUSICAL LOOP SYNTHESIS CONDITIONED ON INTUITIVE MUSICAL PARAMETERS</h1>

<h2>Pritish Chandna, António Ramires, Xavier Serra, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for loop synthesis. Audio examples can be found in the project [website](https://aframires.github.io/drum-loop-synthesis/). An interactive notebook can be found in [here](https://colab.research.google.com/github/aframires/drum-loop-synthesis/blob/main/LOOPNET.ipynb).


<h3>Installation</h3>
To install, clone the repository and use <pre><code>pip install -r requirements.txt </code></pre> to install the packages required.
The pretrained <a href="https://drive.google.com/drive/folders/1iAf4MSLH5yQblWGYkdxBi0NbFpEArJkb?usp=sharing" rel="nofollow"> model weights</a> can be downloaded along with the <a href="https://drive.google.com/file/d/1Tj29ktt4KDLNmqhEaxhvUuF-qM4LwO37/view?usp=sharing" rel="nofollow"> validation hdf5 file</a>. The path to the unzipped model weights and the validation file need to be set in the command line arguments for the main function.

 The code for creating the validation outputs and calculating the FAD is in the *evaluate.py* file.  
 <pre><code>
usage: evalutate.py [-h] [--model MODEL] [--log_dir LOG_DIR] [--val_file VAL_FILE]
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
This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> (H2020 770376) European project. This work is partially supported by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No765068, <a href="https://mip-frontiers.eu/" rel="nofollow">MIP-Frontiers</a>. The TITANX used for this research was donated by the NVIDIA Corporation.
