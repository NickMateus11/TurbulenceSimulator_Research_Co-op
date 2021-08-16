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

## Pythonic Details ##

## Example Output ##

## TODOs ##
