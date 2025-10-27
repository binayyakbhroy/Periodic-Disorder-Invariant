# Periodic Disorder Invariant Calculation
Python and Mathematica implementation of calculating a topological invariant for finite disordered 1D SMSC hybrid system to determine their capability of hosting MZMs based on arXiv:2508.13146
## Disorder Profiles
- Files containing disorder profiles for various lengths of the system.
### Code
- cli.py
    - command line interface for python implementation
    - to run the code just simply run the comman python3 cli.py <process-name> --set <parameter-name> = <parameter-value> ...
    - process names that are currently contained in the python codes are: 
        - spectrum  -> for energy spectrum generation
        - wavefuncs -> for generating "Majorana" or "quasi-Majorana" wavefunctions 
        - pdi       -> calculating the pdi value/map for supplied "--mu" and "--zeeman" or mu and zeeman ranges
        - pfaffian  -> calculating the pfaffian value/map for supplied "--mu" and "--zeeman" or mu and zeeman ranges
        - topov     -> calculating the topological visibility (left and right) value/map for supplied "--mu" and "--zeeman" or mu and zeeman ranges
- requirements.txt
    - contains python requirements
#### libraries
- config.py
    - contains initialization/default values for system parameters
- disorder_profiles.py
    - read disorder profiles in from the "../Disorder_profiles/" directory or generate disorder based on system parameters
- pdi.py
    - calculation of PDI based on recursive nearest neighbor Greens' function algorithm
- pfaffian.py
    - mathematical equivalent of PDI (slower in comparison for increasing lengths of wire)
- system_hamiltonian_profiles.py
    - formulation of a 4Nx4N hamiltonian matrix where N is the number of lattice sites in the system
    - methods to investigate properties of the system using energy spectrum and wavefunction characterization
- topological_visibility.py
    - implements the system hamiltonian using kwant
    - calculates topological visibility at left and right ends
##### Results
- Directory for storing of ".npz" formatted files containing results of the simulation
- plot_generation.py
    - reads all the ".npz" file/s generated and plots them accordingly 
