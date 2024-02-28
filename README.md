# V1_L5_Model

Ziao Chen

A ten thousand cell network of the primary visual cortex with 4 unique cell types in layer 5: CP, CS, FSI, and LTS. Utilizes distance dependent connections for more realism.


There are 3 basic experiments to run:
* baseline        (No 50 Hz input)
* short burst     (50 Hz input in 125 ms bursts)
* long burst      (50 Hz input in 1000 ms bursts) 

They are defined in the "simulation_config_xxx.json" files, where xxx is "baseline", "short" or "long".

To switch which experiment is run, use the corresponding config file "config_xxx.json" to reference the experiment.

### Compile Mod Files (Describes ion channel dynamics)
```
cd components/mechanisms
nrnivmodl modfiles
cd ../..
```

OR... when using CoreNeuron simulator:
```
cd components/mechanisms
nrnivmodl -coreneuron modfiles_core
cd ../..
```

### Building Network
```
sbatch build_batch.sh
```

### Building Input
```
python build_input.py
```

### Running
Compress the whole folder to a zip folder. Run on NSG:
* NEURON on EXPANSE
* Run script is run_network.py

OR... From command line:
```
sbatch batchfile_newserver.sh
```
