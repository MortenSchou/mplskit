<!-- I just looked at the github - seems that the README file is broken little bit
(see Elaborated example) and there are a few grammar mistakes in the intro.
For citing purposes, maybe you can add a list of contributors to the
project as well? -->


# MPLS-Kit

MPLS Data Plane Toolkit.


Provides a library with functionalities for handling network topologies and automatically generate forwarding tables for each router (i.e, a data plane) according to the specifications of industry standard protocols LDP, RSVP-TE and MPLS VPN services. The library can also be extended to include experimental protocols. Aditionally, a set of basic simulation capabilities are provided.

A set of command line utilities for using the library is also provide, keep reading for details.

GPLv3 licensed code.

# Authors
 - **Juan Vanerio** (juan.vanerio@univie.ac.at ). Faculty of Computer Science, University of Vienna --Vienna, Austria.
 - **Stefan Schmid**  (schmiste@gmail.com). Faculty of Computer Science, University of Vienna --Vienna, Austria and Internet Network Architectures, Department of Telecommunication Systems, TU Berlin -- Berlin, Germany.
 - **Morten Konggaard Schou** (mksc@cs.aau.dk).  Dept. of Computer Science,  Aalborg University --Aalborg, Denmark.
 - **Jiří Srba** (srba@cs.aau.dk).  Dept. of Computer Science,  Aalborg University --Aalborg, Denmark.

# Usage

## To generate a topology
```
 python3 mpls_kit_topo_gen.py --name <topology_name> --num_routers <num_routers> --output_file <topology.json>
```

### Example:
```
 python3 mpls_kit_topo_gen.py --name example --num_routers 12 --output_file example/topologies/new_example.json
```
Topologies may also be created randomly.

## Create the data plane for an existing topology
```
 python3 mpls_kit_gen.py --topology --conf <conf_file.yml> --output_file <dataplane.json>
```

### Example to create a data plane
```
 python3 mpls_kit_gen.py --topology example/topologies/example.json  --conf example/config.yaml --output_file example/dataplane/example.json
```

## Create a data plane and a simulation
```
 python3 mpls_kit_sim.py --conf <conf_file.yml> --failure_chunk_file <failure_chunks_file.yml> --result_folder <results_folder>
```

## Example to create a data plane and a simulation
```
python3 mpls_kit_sim.py --topology example/topologies/example.json  --conf example/config.yaml --failure_chunk_file example/failure_chunks/0.yml --result_folder example/results/
```

## Create configuration and failure chunks
```
python3 create_confs.py --topology_path example/topologies/example.json --conf_dir /tmp/results --random_seed 1 --K 2  --threshold 50
```

## Elaborated example
Generating forwarding tables and running simulations on them for a set of topologies in directory `<topologies_dir>`:

0. Make sure the configuration destination folder doesn't exist (e.g.: `/tmp/configs`). Both configuration and failure chunks will be stored there.

1.  Create configurations:
```
for TOPO in $(ls <topologies dir>); do python3 create_confs.py --topology_path <topologies dir>/$TOPO --conf_dir /tmp/configs --random_seed 1 ; done
```

2.  Run a specific simulation (e.g., with 'Azrena' network):
```
python3 mpls_kit_sim.py --conf /tmp/configs/Azrena/conf_1.yml --failure_chunk_file /tmp/configs/Azrena/failure_chunks/0.yml --result_folder /tmp/results/Azrena/results/conf_1/
```

3. In the results folder, the file `0.csv` summerizes the simulation results  (0 stands for failure chunk \#0):
```
head /tmp/main_folder/Azrena/results/conf_1/0.csv
484; 484; 0; 484; 2135; 1651
403; 484; 0; 403; 2135; 1651
323; 484; 0; 323; 2135; 1651
365; 484; 0; 365; 2135; 1651
[...]
```
Where the columns stand for, in order:
  1. Number of successful deliveries.
  2. Total number of flows.
  3. Number of forwarding loops.
  4. Number of connected paths from initial router until a valid destination.
  5. Total number of routing entries.
  6. Number of communications among routers.
