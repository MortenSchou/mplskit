#!/usr/bin/env python3
# coding: utf-8


"""
Script to make simulations on the MPLS data plane. - v0.1

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

Copyright: Juan Vanerio (juan.vanerio@univie.ac.at)
Created by Juan Vanerio, 2021

"""

###############

#Load required libraries
import networkx as nx
# import matplotlib.pyplot as plt
import random
import time
import json, yaml
import math
from pprint import pprint
from itertools import chain, count
import argparse
import sys, os

# from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra_multisource
# from resource import getrusage, RUSAGE_SELF
from mpls_fwd_gen import *
from simulation import *

def main(conf):
    if conf["random_topology"]:
        G = generate_topology(conf["random_mode"],
                          conf["num_routers"],
                          weight_mode = conf["random_weight_mode"],
                          gen_method = conf["random_gen_method"],
                          visualize = False,
                          display_tables = False,
                          random_seed = conf["random_seed_gen"]
                         )

    else:
        # Load topology
        G = topology_from_aalwines_json(conf["topology"], visualize = False)
        print("*****************************")
        print(G.graph["name"])

    ## Generate MPLS forwarding rules
    network = generate_fwd_rules(G,
                                 enable_PHP = conf["php"],
                                 numeric_labels = False,
                                 enable_LDP = conf["ldp"],
                                 enable_RSVP = conf["rsvp"],
                                 enable_RMPLS = conf["enable_RMPLS"],
                                 RMPLS_mode = conf["RMPLS_mode"],
                                 num_lsps = conf["rsvp_num_lsps"],
                                 tunnels_per_pair = conf["rsvp_tunnels_per_pair"],
                                 enable_services = conf["vpn"],
                                 num_services = conf["vpn_num_services"],
                                 PE_s_per_service = conf["vpn_pes_per_services"],
                                 CEs_per_PE = conf["vpn_ces_per_pe"],
                                 protection = conf["protection"],
                                 random_seed = conf["random_seed_gen"]
                          )

    # save config
    net_dict = network.to_aalwines_json()

    output_file = conf["output_file"]
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(net_dict, f, indent=2)
    else:
        print(" ~~~ FORWARDING DATAPLANE  ~~~ ")
        pprint(net_dict)

    if conf["print_flows"]:
        flows = network.build_flow_table()
        queries = []
        for router_name, lbl_items in flows.items():
            for in_label, tup in lbl_items.items():
                good_sources, good_targets = tup
                q_targets = ",.#".join(good_targets)
                queries.append(f"<{in_label}> [.#{router_name}] .* [.#{q_targets}] < > 0 OVER\n")

        output_file = conf["output_file"]
        if output_file:
            with open(output_file + ".q", 'w') as f:
                f.writelines(queries)
        else:
            for q in queries:
                print(q)
        return

    result_folder = conf["result_folder"]
    os.makedirs(result_folder, exist_ok = True)
    result_file = os.path.join(result_folder, "default")

    failed_set_chunk = [[]]
    if conf['failure_chunk_file']:
        with open(conf['failure_chunk_file'], 'r') as f:
            failed_set_chunk = yaml.safe_load(f)
            chunk_name = conf['failure_chunk_file'].split('/')[-1].split(".")[0]
            result_file = os.path.join(result_folder, chunk_name+".csv")

    with open(result_file, 'w') as f:
        for failed_set in failed_set_chunk:
            simulation(network, failed_set, f)

def simulation(network, failed_set, f):
    # simulator stuff
    print("STARTING SIMULATION")
    print(failed_set)

    # Mangle the topology:
    def filter_node(n):
        #return False if n in L else True
        return True

    F = [tuple(x) for x in failed_set]  # why need this??
    def filter_edge(n1,n2):
        if (n1,n2) in F or (n2,n1) in F:
            return False
        return True

    # Compute subgraph
    view = nx.subgraph_view(network.topology, filter_node = filter_node, filter_edge = filter_edge)

    # Instantiate simulator object
    s = Simulator(network, trace_mode="links",  restricted_topology = view, random_seed =  conf["random_seed_sim"] )
    # random.seed(random_seed)
    s.run()
    (success,total,codes) = s.success_rate(exit_codes = True)
    loops = codes[1]
    #print("{0:.2f}% ((total: {1})".format(100*success,total))
    #f.write("{0:.4f}; {1}\n".format(success,total))
    # f.write("{0}; {1}; {2}; {3}\n".format(success, total, loops, s.count_connected))
    f.write("{0}; {1}; {2}; {3}; {4}; {5}\n".format(success, total, loops, s.count_connected, network.get_number_of_routing_entries(), network.get_comm_count()  ))
    # results = s.print_traces(store = True)

    # result_file = conf["results_file"]
    # if result_file:

    #         #json.dump(results, f)
    #         f.write(results)
    # else:
    #     print(" ~~~ SIMULATION RESULTS  ~~~ ")
    #     print(results)

    print("SIMULATION FINISHED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Command line utility to generate MPLS forwarding rules.')

    #general options
    p.add_argument("--conf", type=str, default = "", help="Load configuration file.")

    group1 = p.add_mutually_exclusive_group()
    group1.add_argument("--topology", type=str, default = "", help="File with existing topology to be loaded.")
    #random topology options
    group1.add_argument("--random_topology", action="store_true", help="Generate random topology. Defaults False.")
    p.add_argument("--random_mode", type=str, default = "random.log_degree", help="custom, random.log_degree, random.large_degree")
    p.add_argument("--num_routers",type=int, default = 6, help="Number of routers in topology.")
    p.add_argument("--random_weight_mode", type=str, default = "random", help="random, equal or distance")
    p.add_argument("--random_gen_method",type=int, default = 1, help="Method for generating the random topology")


    # Generator options
    p.add_argument("--php", action="store_true", help="Enable Penultimate Hop Popping. Defaults False.")
    p.add_argument("--ldp", action="store_true", help="Enable Label Distribution Protocol (LDP). Defaults False. This doesn't scale well.")
    p.add_argument("--rsvp", action="store_true", help="Enable Resource Reservation Protocol with Traffic Engineering extensions (RSVP-TE). Defaults False.")
    p.add_argument("--rsvp_num_lsps",type=int, default = 2, help="Number of (random) LSPS to compute for RSVP only, if enabled. Defaults to 2")
    p.add_argument("--rsvp_tunnels_per_pair",type=int, default = 5, help="Number of (random) tunnels between same endpoints. RSVP only, if enabled. Defaults to 5")
    p.add_argument("--enable_RMPLS", action="store_true", help="Use experimental RMPLS recursive protection (LFIB post processing). Defaults False")
    p.add_argument("--RMPLS_mode", type=str, default = "link", help="Mode for the experimental RMPLS recursive protection. link (default) or facility")
    p.add_argument("--protection", type=str, default = "facility-node", help="RSVP protection to implement. facility-node (default), facility-link or None ")


    p.add_argument("--vpn", action="store_true", help="Enable MPLS VPN generic services. Defaults False. ")
    p.add_argument("--vpn_num_services",type=int, default = 1, help="Number of (random) MPLS VPN services, if enabled. Defaults to 1")
    p.add_argument("--vpn_pes_per_services",type=int, default = 3, help="Number of PE routers allocated to each MPLS VPN service, if enabled. Defaults to 3")
    p.add_argument("--vpn_ces_per_pe",type=int, default = 1, help="Number of CE to attach to each PE serving a VPN, if enabled. Defaults to 1")

    p.add_argument("--random_seed_gen",type=int, default = random.randint(0, 99999999), help="Random seed for genrating the data plane. Leave empty to pick a random one.")
    p.add_argument("--random_seed_sim",type=int, default = random.randint(0, 99999999), help="Random seed for simulation execution. Leave empty to pick a random one.")

    p.add_argument("--output_file",type=str, default = "", help="Path of the output file, to store forwarding configuration. Defaults to print on screen.")

    #p.add_argument('--verbose', help='Print more data', action='store_true')

    # Simulator options
    p.add_argument("--failure_chunk_file", type=str, default="", help="Failure set, encoded as json readable list. ")
    p.add_argument("--result_folder",type=str, default = "", help="Path to the folder of simulation result files. Defaults to print on screen.")
    p.add_argument("--print_flows", action="store_true", help="Print flows instead of running simulation. Defaults False.")


    args = p.parse_args()

    print(args)
    conf = vars(args)  # arguments as dictionary
    conf_overrride = conf.copy()
    known_parameters = list(conf.keys()).copy()

    # Configuration Load
    if args.conf:
        print(f"Reading configuration from {args.conf}. Remaining options will override the configuration file contents.")
        if args.conf.endswith(".yaml") or args.conf.endswith(".yml"):
            with open( args.conf, 'r') as f:
                conf_new = yaml.safe_load(f)
                conf.update(conf_new)
                print("Configuration read from file.")
        elif args.conf.endswith(".json"):
            #we try json
            with open( args.conf, 'r') as f:
                conf_new = json.load(f)
                conf.update(conf_new)
                print("Configuration read from file.")
        else:
            raise Exception("Unsupported file format. Must be YAML or JSON.")

    # Deal with the topology
    if args.topology and not args.random_topology:
        print(f"Reading topology from {args.topology}. ")
        # with open( args.topology, 'r') as f:
        #     conf["topology"] = json.load(f)
        conf["topology"] = args.topology

    elif args.random_topology:
        conf["random_topology"] = True
        if args.num_routers < 6:
            raise Exception("You need at least 6 routers.")
        conf["random_mode"] = args.random_mode
        conf["num_routers"] = args.num_routers
        conf["random_weight_mode"] = args.random_weight_mode
        conf["random_gen_method"] = args.random_gen_method

    # elif conf["topology"] and not conf["random_topology"] :
    #     print("case 2")
    #     print(f"Reading topology from {args.conf}. ")
    #     with open( args.topology, 'r') as f:
    #         conf["topology"] = json.load(f)

    elif conf["random_topology"]:
        if args.num_routers < 6:
            raise Exception("You need at least 6 routers.")
        conf["random_mode"] = args.random_mode
        conf["num_routers"] = args.num_routers
        conf["random_weight_mode"] = args.random_weight_mode
        conf["random_gen_method"] = args.random_gen_method
    else:
        raise Exception("One (and only one) of topology or random_topology must be selected.")

    # p.add_argument("--fail_set", type=str, default="[]", help="Failure set, encoded as json readable list. "))
    # if args.fail_set:
    #     print(f"Reading fail set from {args.fail_set}. ")
    #     tmp_list = yaml.load(args.fail_set, Loader=yaml.SafeLoader)
    #     print("Failed links: ")
    #     pprint(tmp_list)
    #     conf["fail_set"] = []
    #     for x in tmp_list:
    #         conf["fail_set"].append(tuple(x))

    if conf["protection"] in ["None","none"]:
        conf["protection"] = None

    # raise errors if there is any unknown entry...
    for a in conf.keys():
        if a not in known_parameters:
            raise Exception(f"Unknown argument {a}.")

    # Override config options with cli explicit values
    for a in known_parameters:
        for sysarg in sys.argv:
            # this argument was explicitly overriden!
            if sysarg == "--" + a:   #max prio
                conf[a] = conf_overrride[a]

    pprint(conf)
    main(conf)
