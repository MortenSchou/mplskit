#!/usr/bin/env python3
# coding: utf-8


"""
Script to generate the topologies. - v0.1

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright : Juan Vanerio (juan.vanerio@univie.ac.at)
Created by Juan Vanerio, 2022


Arguments
----------
    # # Arguments

    # random_mode         For random topologies, how to obtain the number of edges.
                          One of: custom, random.log_degree, random.large_degree

    # num_routers         Number of routers in the random topology. Defaults to 6.

    # random_weight_mode  How to generate link weights. One of random, equal or
                          distance.

    # random_gen_method   Method for generating the random topology. 1 or 0.

    # random_seed           Random seed. Leave empty to pick a random one.


Returns
-------
 Prints or save the computed (AalWiNes) JSON topology.

"""

###############

#Load required libraries
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import json, yaml
import math
from pprint import pprint
from itertools import chain, count
import argparse
import sys

# from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra_multisource
# from resource import getrusage, RUSAGE_SELF
from mpls_fwd_gen import *

def main(conf):
    G = generate_topology(conf["random_mode"],
                          conf["num_routers"],
                          weight_mode = conf["random_weight_mode"],
                          gen_method = conf["random_gen_method"],
                          visualize = False,
                          display_tables = False,
                          random_seed = conf["random_seed"]
                         )

    topo_dict = topology_to_aalwines_json(G, name = conf["name"])

    # save topo
    output_file = conf["output_file"]
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(topo_dict, f, indent=2)
    else:
        pprint(net_dict)



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Command line utility to generate MPLS forwarding rules.')

    p.add_argument("--name", type=str, default = "example_network", help="Name for the generated topology.")

    p.add_argument("--random_mode", type=str, default = "random.log_degree", help="custom, random.log_degree, random.large_degree")
    p.add_argument("--num_routers",type=int, default = 6, help="Number of routers in topology.")
    p.add_argument("--random_weight_mode", type=str, default = "random", help="random, equal or distance")
    p.add_argument("--random_gen_method",type=int, default = 1, help="Method for generating the random topology")

    p.add_argument("--random_seed",type=int, default = random.randint(0, 99999999), help="Random seed. Leave empty to pick a random one.")

    p.add_argument("--output_file",type=str, default = "", help="Path of the output file, to store forwarding configuration. Defaults to print on screen.")


    args = p.parse_args()

    print(args)
    conf = vars(args)  # arguments as dictionary


    pprint(conf)
    main(conf)
