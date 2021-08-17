# Propgation Simulator Repo #

This repo is a sandbox for an ___Atmospheric Turbulence Simulation___ tool. 

The Python code was adapted from the MATLAB work done in this textbook [Numerical Simulation of Optical Wave Propagation with Examples in MATLAB](https://spie.org/Publications/Book/866274?SSO=1). Most functions can be found with the same name as in the textbook for further reading and explanation.

The [Numerical Simulation of Optical Wave Propagation with Examples in MATLAB](https://spie.org/Publications/Book/866274?SSO=1) textbok is a fantastic resource to read more about wave propagation techniques and atmoshperic turbulence models.


## Setup ##
 - create a python venv:
    ```
    > cd <desired directory for virtual env location>
    > python -m venv <name of venv>
    > <name of venv>\Scripts\activate.bat
    ```
 - Execute:
    ```
    > cd <repository directory>
    > pip install -r 'requirements.txt'
    ```
 - Install CUDA (GPU support for Tensorflow) - [instructions](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#gpu-support-optional)


## Execution ##

As scripts are mostly sandboxed/testing simulator modules, execution is as easy as running any of the `./src/*.py` scripts. Ex: 
```
 > python ./src/gaussian_beam_test.py   
```

## Architechure ##

### Block Diagram: ###

![block diagram](.\\images\\SimplifiedTurbulenceBlockDiagram.png)

### Block Details: ###

![block details](.\\images\\TurbulenceBlockDiagramIndexTables.png)


## Pythonic Details ##

As mentioned, most functions were adapted from MATLAB to work in Python. This was mostly due to my familiarity with Python, as well as it's strong library support (ie: Matplotlib, Numpy, Tensorflow, etc).

Python classes are used to easily encapuslate some of the blocks shown above, like the Phase Screen Generator.

### GPU acceleration with Tensorflow: ###

Tensorflow was used to enable GPU access and execute computationally expensive code very quickly. Speed performance between CPU and GPU execution saw results of 100x or more. The only issue was found to be limited VRAM (5GB) which, if not careful, can be quickly consumed by high resolution phase screens.

There are 2 main execution flows within Tensorflow - Eager or Graph execution. Eager execution, often the default for Tensorflow 2.0 will execute Python code, operation by operation, and return the results. This is often still fast, but limits the speed ceiling. This project mainly uses eager execution of Tensorflow. The Graph execution approach proved to be more difficult to implement, more information can be found [here](https://www.tensorflow.org/guide/intro_to_graphs).


## Example Output (WIP: Description/Labelling needed) ##

<img src=".\\images\\lowFreq.png" width="30%">
<img src=".\\images\\highFreq.png" width="30%">
<img src=".\\images\\phasescreen.png" width="30%">

![low/no turbulence](.\\images\\phaseScreenStats.png)

![low/no turbulence](.\\images\\frozen_flow_rect.gif)

![low/no turbulence](.\\images\\noTurbulence.png)
![low/no turbulence](.\\images\\mildTurbulence.png)

![low/no turbulence](.\\images\\animation_focused.gif)


## TODOs ##
 - Channel Layer Slicer module (used to calculate optimal r0 values along the propagation path). More details can be found in Ch. 9.5 (& Listing 9.5) in the mentioned [MATLAB textbook](https://spie.org/Publications/Book/866274?SSO=1).
 - Graph execution of Tensorflow functions (performance increase). Information [here](https://www.tensorflow.org/guide/intro_to_graphs).
 - Further parallelization (execute multiple end-to-end propagations at once). This may be limited by the GPU VRAM. Optimizations may need to implemented in the storage of phase screens - ie: storing only the random seeds and regenerating the phase screens, or at least parts of them in order to reduce how much is being stored.
    - Perhaps look into running simulator components on GPU clusters on the cloud - where more memory may be available.
