<!-- I just looked at the github - seems that the README file is broken little bit
(see Elaborated example) and there are a few grammar mistakes in the intro.
For citing purposes, maybe you can add a list of contributors to the
project as well? -->


# MPLS-Kit

MPLS Data Plane Toolkit.


Provides a library with functionalities for handling network topologies and automatically generate forwarding tables for each router (i.e, a data plane) according to the specifications of industry standard protocols LDP, RSVP-TE and MPLS VPN services. The library can also be extended to include experimental protocols. Aditionally, a set of basic simulation capabilities are provided.

A set of command line utilities for using the library is also provide, keep reading for details.

GNUv3 licensed code.

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

## Creating the data plane for an existing topology
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

## Creation of configuration and failure chunks
```
python3 create_confs.py --topology_path example/topologies/example.json --conf_dir /tmp/results --random_seed 1 --K 2  --threshold 50
```

## Elaborated example
Generating forwarding tables and running simulations on them for a set of topologies in directory `<topologies_dir>`:

0. Make sure the configuration destination folder doesn't exist (e.g.: `/tmp/configs`). Both configuration and failure chunks will be stored there.

1.  Create configurations:
```
for TOPO in $(ls <topologies dir>) ;
    do python3 create_confs.py --topology_path <topologies dir>/$TOPO --conf_dir /tmp/configs --random_seed 1 ;
done
```

2.  Run a specific simulation:
```
python3 mpls_kit_sim.py --conf /tmp/main_folder/Kdl/conf_1.yml --failure_chunk_file /tmp/configs/Kdl/failure_chunks/0.yml --result_folder /tmp/main_folder/Kdl/results/conf_1/
```
