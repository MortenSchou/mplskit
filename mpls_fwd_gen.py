#!/usr/bin/env python3
# coding: utf-8

"""
###################################

     MPLS FORWARDING GENERATOR - MPLS Kit main library.

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
Created by Juan Vanerio, 2020

###################################
"""
import sys

import networkx as nx
import matplotlib.pyplot as plt
import random
import time
# import jsonschema
import json
import math
import copy
from pprint import pprint
from itertools import chain
from functools import reduce

# import networkx.algorithms
import numpy as np

from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra_multisource
from resource import getrusage, RUSAGE_SELF



# Auxiliary functions
def rand_name():
    """
    Generates a random name for general purposes, all lower-case.
    """
    consonants = 'bcdfghjklmnpqrstvwxyz'
    vowels = 'aeiou'
    syllables = list(vowels)


    for c in consonants:
        for v in vowels:
            syllables.append(v+c)
            syllables.append(c+v)
            for cc in consonants:
                syllables.append(c+v+cc)

    syllables = list(set(syllables))
    weights = list()
    for s in syllables:
        if len(s) == 3:
            weights.append(1)
        elif len(s) == 2:
            weights.append(11)
        elif len(s) == 1:
            weights.append(300)

    x = random.choices(syllables, weights = weights, k = random.randint(2,4))
    return("".join(x))

def as_list(x):
    if isinstance(x,list):
        return x
    else:
        return [x]

def dict_filter(_dict, callback):
    '''
    Call boolean function callback(key, value) on all pairs from _dict.
    Keep only True result and return new dictionary.
    '''
    new_dict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in _dict.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            new_dict[key] = value
    return new_dict

def rec_int_val_to_str(d):
    # Goes through a dictionary or list (or combinations thereof)
    # changing numbers into strings.

    if isinstance(d,dict):
        for k,v in d.items():
            d[k] = rec_int_val_to_str(v)
        return d

    elif isinstance(d,list):
        for i in range(len(d)):
            d[i] = rec_int_val_to_str(d[i])
        return d

    elif isinstance(d,str) or isinstance(d,int) or isinstance(d, float):
        return str(d)

    else:
        return d

# generation functions
def gen_connected_random_graph(n, m, seed=0, directed=False, method=0):
    """Returns a random graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges.
    seed : integer (default 0) or random_state.
        Randomness seed.
    directed : bool, optional (default=False)
        If True return a directed graph
    method : integer (default=0).
        Defines the link generation method.
        Method 0 creates random edges between nodes, can't guarantee connectivity.
        Method 1 guarantees connectivity, yet with a non-uniform degree distribution.

    Adapted from a networkx function of the same name.
    """

    if directed:
        G = nx.DiGraph()
        max_edges = n * (n - 1)
    else:
        G = nx.Graph()
        max_edges = n * (n - 1) / 2

    router_names = []
    for i in range(n):
        router_names.append("R{}".format(i))

    G.add_nodes_from(router_names)

    if n == 1:
        return G

    if m >= max_edges:
        G = nx.complete_graph(n, create_using=G)
        G = nx.relabel_nodes(G, lambda i: f"R{i}" , copy=False)
        return G

    random.seed(seed)

    # Generate edges...
    if method == 0:
        # Adds edges randomly between nodes until the graph has m edges
        # Can't ensure graph connectivity.
        edge_count = 0
        while edge_count < m:
            # generate random edge,u,v
            u = router_names[random.randint(0,n-1)]
            v = router_names[random.randint(0,n-1)]
            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1

    elif method == 1:
        # Go through nodes in ascending order, generates new random links with
        # nodes already seen only. Ensures connectivity, yet the first nodes
        # are more likely to have a greater degree than the last ones.
        edge_count = 0
        usable_range = 0
        while edge_count < m:
            # generate random edge,u,v
            u = router_names[random.randint(0,usable_range)]
            if edge_count < n-1:
                v = router_names[edge_count + 1]
            else:
                v = router_names[random.randint(0,n-1)]

            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1
                if edge_count < n-1:
                    usable_range += 1

    return G


def generate_topology(mode, n, weight_mode = "random", gen_method = 1,
                      visualize = False, display_tables = False, random_seed = random.random() ):
    """
    Generates a topology, given a number of nodes and a edge generation mode.

    Parameters:
    mode:           Method for generating the number of edges.
                       custom
                       random.log_degree
                       random.large_degree

    n:              Number of nodes.

    weight_mode:    Edge weight value: random, equal or distance

    gen_method:     Method for generating the random topology

    visualize:      Use only with small networks

    display_tables: Use only with small networks

    Returns generated graph G.
    Empirically the time consumed is approximately proportional to the square of n.

    """


    def _weight_fn(mode = "equal", **kwargs):
        if mode == "random":
            return random.randint(a=1,b=10)

        elif mode == "distance":
            # we expect additional args p0 and p1
            return _geo_distance( p0,  p1)

        else:
            return 1

    def _geo_distance( p0,  p1):
        """
        Calculate distance (in kilometers) between p0 and p1.
        Each point must be represented as 2-tuples (longitude, latitude)
        long \in [-180,180], latitude \in [-90, 90]
        """
        def _hav(theta): # Haversine formula
            return (1 - math.cos(theta))/2

        a = math.pi / 180 # factor for converting to radians
        ER = 6371 # 6371 km is the earth radii

        lo_0, la_0 = p0[0] * a, p0[1] * a
        lo_1, la_1 = p1[0] * a, p1[1] * a
        delta_la, delta_lo = la_1 - la_0, lo_1 - lo_0
        sum_la, sum_lo = la_1 + la_0, lo_1 + lo_0

        h =  _hav(delta_la) + (1 - _hav(delta_la) - _hav(sum_la))*_hav(sum_lo)
        theta = 2*math.sqrt(h)    # central angle in radians

        return ER * theta

    if mode.startswith("random"):
        #number of nodes
        random.seed(random_seed)

        # uneven range for number of links
        if mode == "random.large_degree":
            # avg. degree proportional to n*n
            beta_l = n/4 + 1/2
            beta_u = n/2

        elif mode == "random.log_degree":
            # attempts to get an average degree proportional to log of (n-1)*log n
            beta_l = 0.5*math.log2(n)  # belongs to [1,n/2]
            beta_u = 1.16*math.log2(n)    # belongs to [1,n/2] and is ge than beta_l 1.16

        lower_b = int(beta_l*(n-1))
        upper_b = int(beta_u*(n-1))

        #pprint((lower_b, upper_b))
        e = random.randint(lower_b, upper_b)
        print(f"Number of edges: {e}")
        G = gen_connected_random_graph(n, e, method = gen_method, seed = random_seed)

        pos = nx.spring_layout(G)

        for u,v in G.edges():
            if weight_mode == "distance":
                G[u][v]["weight"] = _weight_fn(weight_mode,
                                            p0 = (u["longitude"], u["latitude"]),
                                            p1 = (v["longitude"], v["latitude"]))
            else:
                G[u][v]["weight"] = _weight_fn(weight_mode)


    elif mode == "custom":
        # custom graph used for testing.
        G = nx.Graph()
        G.add_edge("R0", "R1", weight=_weight_fn(weight_mode))
        G.add_edge("R1", "R2", weight=_weight_fn(weight_mode))
        G.add_edge("R1", "R3", weight=_weight_fn(weight_mode))
        G.add_edge("R2", "R4", weight=_weight_fn(weight_mode))
        G.add_edge("R3", "R4", weight=_weight_fn(weight_mode))
        G.add_edge("R2", "R5", weight=_weight_fn(weight_mode))
        G.add_edge("R2", "R6", weight=_weight_fn(weight_mode))
        G.add_edge("R5", "R7", weight=_weight_fn(weight_mode))
        G.add_edge("R6", "R7", weight=_weight_fn(weight_mode))
        G.add_edge("R7", "R8", weight=_weight_fn(weight_mode))

    if visualize:
        try:
            pos
        except NameError:
            pos = nx.spring_layout(G)

        fig, ax = plt.subplots(figsize=(12, 7))
        nx.draw_networkx_nodes(G, pos, node_size=250, node_color="#210070", alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color="m")

        labels = nx.get_edge_attributes(G,'weight')
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos, font_size=14, bbox=label_options)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    return G


def generate_fwd_rules(G, enable_PHP = True, numeric_labels = False, enable_LDP = False,
                       enable_RSVP = False, num_lsps = 10, tunnels_per_pair = 3, protection = 'facility-node',
                       enable_services = False, num_services = 2, PE_s_per_service = 3, CEs_per_PE = 1,
                       enable_RMPLS = False, RMPLS_mode='link', random_seed = random.random(), enable_mpls_sr = False,
                       mpls_sr_tunnels = None, policies = None):
    """
    Generates MPLS forwarding rules for a given topology.

    Parameters:

    enable_PHP       : Boolean (defaults True). Activate Penultimate Hop Popping functionality.

    numeric_labels   : Boolean (defaults False). Use numeric labels (int) instead of strings.
                       String type labels are required by rthe current service implementation.

    enable_LDP       : Boolean (defaults False). Sets up Label Distribution Protocol (LDP, RFC5036).
                       A label will be allocated for each link and for each node (loopback address)
                       in the network, in a plain fashion (single IGP area/level). Beware that its
                       computation requires O(n^3) cpu time and O(n^4) memory (At least empirically).

    enable_RSVP      : Boolean (defaults True). Sets up Resource Reservation Protocol (RSVP-TE, RFC4090
                       and others). RSVP defines tunnels (LSPs) between a headend and a tail end allowing
                       for traffic engineering. It requires the configuration of the following additional
                       parameters:

                       num_lsps  : (defaults 10). Number of TE tunnels to compute between different
                                   pairs of head and tailend.


                       tunnels_per_pair: (defaults 3) Number of tunnels with between each pair of
                                   head and tailend. So if num_lsps = 10 and tunnels_per_pair = 3,
                                   a total of 30 tunnels would be generated.

    enable_RMPLS     : Enable additional experimental recursive protection by post-processing the LFIB.


    enable_services  : Boolean (defaults True). Sets up MPLS VPN services. These services abstract away
                       the specific type of service (VPWS, VPLS or VPRN), using a single class to represent
                       them all. Requires addtional parameters:

                       num_services: (defaults 2). Number of VPN services to instantiate.

                       PE_s_per_service: (defaults 3). Number of PE routers per VPN service.

                       CEs_per_PE: (defaults 1). Number of CE routers attached to each service PE.

    """
    # Import locally so that the import works ...

    # Instantiate the network object with topology graph G
    network = Network(G)

    for n,r in network.routers.items():
        r.php = enable_PHP
        r.numeric_labels = numeric_labels

    if enable_LDP or enable_mpls_sr:
        print("Computing shortest paths ...")
        random.seed(random_seed)
        network.compute_dijkstra()
        print("Shortest paths computed.")

    if enable_LDP:
        print("Computing LDP...")
        from ldp import ProcLDP
        random.seed(random_seed)
        # TODO: implement LDP over RSVP. Must invert order for calling dijkstra.

        # Start LDP process in each router
        network.start_client(ProcLDP)

        for n,r in network.routers.items():
            r.clients["LDP"].alloc_labels_to_known_resources()
            # Allocate labels for known interfaces

        print("LDP ready.")

    if enable_RSVP:
        print("Computing RSVP...")
        from rsvpte import ProcRSVPTE
        random.seed(random_seed)
        # Start RSVP-TE process in each router
        if protection is not None and protection.startswith("plinko"):
                from plinko import ProcPlinko
                network.start_client(ProcPlinko)
                protocol_name = "ProcPlinko"
        else:
            network.start_client(ProcRSVPTE)
            protocol_name = "RSVP-TE"

        # compute  lsps
        print(f"num_lsps: {num_lsps}")
        # num_lsps can be:
        #   an integer: so we generate randomly.
        #   or a list of tuples (headend, tailend) in order to generate manually.
        if isinstance(num_lsps, int):

            i = 0  # Counter for pairs of headend, tailend
            err_cnt = 0 # Counter of already allocated headend,tailend
            print_thr = 10/num_lsps

            random.seed(random_seed)
            router_names = list(network.routers.keys())

            tunnels = dict()   # track created (headend, tailend) pairs

            while i < num_lsps:
                success = False

                tailend = None
                headend = None
                # get a tunnel across different nodes
                while tailend == headend:
                    headend = router_names[random.randint(0,len(G.nodes)-1)]
                    tailend = router_names[random.randint(0,len(G.nodes)-1)]

                if (headend, tailend) not in tunnels.keys():
                    tunnels[(headend, tailend)] = 0

                for j in range(tunnels_per_pair):
                    tunnels[(headend, tailend)] += 1  #counter to differentiate tunnels on same (h,t) pair

                    try:
                        network.routers[headend].clients[protocol_name].define_lsp(tailend,
                                                                       # tunnel_local_id = j,
                                                                       tunnel_local_id = tunnels[(headend, tailend)],
                                                                       weight='weight',
                                                                       protection=protection)
                        success = True
                    except Exception as e:
                        pprint(e)
                        # We have already generated the tunnels for this pair of headend, tailend.
                        err_cnt += 1
                        if err_cnt > num_lsps*10:
                            # too many errors, break the loop!
                            i = num_lsps
                            break
                if random.random() < print_thr:
                    print("Tunnel_{}: from {} to {}".format(i,headend,tailend))
                if success:
                    i += 1

        elif isinstance(num_lsps, list):
            #verify that this is a list of pairs:
            assert all(filter(lambda x: isinstance(x,(list,tuple)) and len(x)==2 ,num_lsps))
            c = dict()
            for h,t in num_lsps:
                print(f"Manually build LSP from {h} to {t}")
                if (h,t) not in c.keys():
                    c[(h,t)] = 0
                else:
                    c[(h,t)] += 1

                result = network.routers[h].clients[protocol_name].define_lsp(t,
                                                                 tunnel_local_id = c[(h,t)],
                                                                 weight='weight',
                                                                 protection=protection)
                print(f"RESULT IS: {result}")
                pprint(network.routers[h].clients[protocol_name].headended_lsps)


        else:
            raise Exception("num_lsps has wrong type (must be list of pairs or integer)")

        for n,r in network.routers.items():
            r.clients[protocol_name].commit_config()

        for n,r in network.routers.items():
            r.clients[protocol_name].compute_bypasses()

        if protection is not None and protection.startswith("plinko"):
            arguments = protection.split("/")
            max_resiliency = 4 #default value
            if len(arguments) > 1:
                    max_resiliency = int(arguments[1])

            for t in range(1,max_resiliency+1):
                for n,r in network.routers.items():
                    r.clients[protocol_name].add_protections(t)

        for n,r in network.routers.items():
            r.clients[protocol_name].alloc_labels_to_known_resources()

        print(f"RSVP ready (frr variant={protection}).")

    if enable_services:
        print("Computing Services")
        from service import MPLS_Service
        random.seed(random_seed)

        # compute services
        ce_cnum = dict()  #keep track of ce numbering on each node
        for i in range(num_services):
            # choose  routers at random
            if PE_s_per_service > len(list(network.routers.values())):
                PE_s_per_service = len(list(network.routers.values()))
            PEs = random.sample(list(network.routers.values()) ,  PE_s_per_service )

            vpn_name = "Service_{}".format(i)
            for pe in PEs:
                client = pe.get_client(MPLS_Service)
                if not client:
                    client = pe.create_client(MPLS_Service)
                client.define_vpn(vpn_name)
                if pe not in ce_cnum.keys():
                    ce_cnum[pe] = 0
                for _ in range(CEs_per_PE):
                    ce_name = "CE_{}".format(ce_cnum[pe])
                    ce_cnum[pe] += 1
                    client.attach_ce(vpn_name, ce_name)

        for n,r in network.routers.items():
            if "service" in r.clients.keys():
                r.clients["service"].alloc_labels_to_known_resources()

        print("Services ready.")

    # MPLS Segment Routing
    if enable_mpls_sr:
        print("Enabling MPLS Segment routing ...")
        from sr_client import SegmentRoutingClient, SRController, SRDatabase, SRCandidatePath, SRPolicy
        random.seed(random_seed)

        # Instanciate only one controller with one single database
        # Note that there may be multiple controllers, but if so, all have to keep a pointer
        # to the same database.

        # Do not pass controller from the user but lists of arguments
        controller = SRController(topology=network.topology, database=SRDatabase())

        # Parse the input format
        # We return a list of policies and insert them here into the controller,
        # so that we can call this method recursively for composite candidate paths.
        from sr_client import sr_parse_input_format
        for policy in sr_parse_input_format(controller, policies):
            controller.insert_policy(policy)

        network.start_client(SegmentRoutingClient, controller=controller)

        for n, r in network.routers.items():
            if "MPLS_SR" in r.clients.keys():
                print("Let's call alloc labels to build the LIB")
                r.clients["MPLS_SR"].alloc_labels_to_known_resources()
        print("MPLS SR enabled.")
    # END MPLS Segment Routing


    if enable_RMPLS:
        print("Enabling RMPLS -- recursive protection")
        from rmpls import ProcRMPLS
        random.seed(random_seed)

        # Start RMPLS process in each router
        network.start_client(ProcRMPLS, rmpls_mode=RMPLS_mode)

        for n,r in network.routers.items():
            r.clients["RMPLS"].compute_bypasses()

        for n,r in network.routers.items():
            r.clients["RMPLS"].alloc_labels_to_known_resources()

        print("RMPLS enabled.")




    # Forwarding phase: fill the LFIB
    print("building LFIB")

    # Build the Label forwarding information base
    network.LFIB_build_orderly()
    print("LFIB ready.")


    # #idempotency test: try to run it again
    # print("building LFIB again")
    # network.LFIB_build_orderly()
    # print("LFIB ready again?.")


    print("adapting priorities.")   # let the fwd entries have ordinal priorities instead of integer weights.
    for n,r in network.routers.items():
        r.LFIB_weights_to_priorities()

    print("Refining LFIB")
    for n,r in network.routers.items():
        r.LFIB_refine()

    print("Finished.")

    return network


def topology_from_graphml(filename, visualize = True):
    # simple attempt at loading from a graphml file
    GG = nx.read_graphml(filename)
    G = nx.Graph(GG)

    name = filename.split("/")[-1].split(".")[0]
    G.graph["name"] = net_dict_2["network"]["name"]  # load name

    return G


def topology_from_aalwines_json(filename, visualize = True):
    # load topology from aalwines json format
    with open(filename, 'r') as f:
        net_dict_2 = json.load(f)

    G = nx.Graph()

    if "name" in net_dict_2["network"]:
        G.graph["name"] = net_dict_2["network"]["name"]

    #build alias map a (point each alias to actual router name)
    a = dict()

    #build router names List
    router_names = []
    for r_dict in net_dict_2["network"]["routers"]:
        a[r_dict["name"]] = r_dict["name"]
        router_names.append(r_dict["name"])
        for alias_name in r_dict["alias"] if "alias" in r_dict else []:
            a[alias_name] = r_dict["name"]

    #create edges
    links = net_dict_2["network"]["links"]
    for data in links:
        u = data["from_router"]
        v = data["to_router"]
        if "weight" in data.keys():
            w = data["weight"]
        else:
            w = 1

        #random weights
        G.add_edge(a[u], a[v], weight = w)

    # deal with coordinates
    for r_dict in net_dict_2["network"]["routers"]:
        for r in [r_dict["name"]] + (r_dict["alias"] if "alias" in r_dict else []):
            if r in router_names and "location" in r_dict:
                G.nodes[r]["latitude"] =  r_dict["location"]["latitude"]
                G.nodes[r]["longitude"] =  r_dict["location"]["longitude"]
                break

    return G

##############################
def topology_to_aalwines_json(G, name = None):
    # This function generates a dictionary compatible with AalWiNes JSON schema.
    if not name:
        name = rand_name()
    net_dict = {"network":  {"links": [], "name": name, "routers": []} }

    # Topology
    links = net_dict["network"]["links"]
    for from_router,to_router in G.edges():
        links.append({
            "bidirectional": True,
            "from_interface": to_router,
            "from_router": from_router,
            "to_interface": from_router,
            "to_router": to_router,
            "weight": G[from_router][to_router].get("weight", 1)
          })

    # for router_name in G.nodes():
    #     router = self.routers[router_name]
    #     links.append({
    #         "from_interface": router.LOCAL_LOOKUP,
    #         "from_router": router_name,
    #         "to_interface": router.LOOPBACK,
    #         "to_router": router_name
    #       })
    #
    # Forwarding rules
    for router in G.nodes():
        net_dict["network"]["routers"].append({"name": router})

    return net_dict
##############################

# Classes
class FEC_registry(object):
    """
    Experimental
    """
    def __init__(self):
        self.registry = dict()

    def fec_lookup(self, fec):
        # work in progress
        if fec.name not in self.registry:
            self.registry[fec.name] = {'FEC':fec}
        return self.registry[fec.name]['FEC']

    def get_num_fecs(self):
        return len(self.registry)


class Network(object):
    """
    Class keeping track of the topology and all routers in the network.
    It provides helper methods to call router functions network-wide:
    shortest-path computation, create and start client processes start, LFIB table build and
    visualization of LIB and LFIB tables.

    """
    def __init__(self, topology, name=None):

        # load network topology (a networkx graph)
        self.topology = topology
        if not name:
            try:
                self.name = topology.graph["name"]
            except:
                self.name = rand_name()
        else:
            self.name = name

        #self.service_registry = dict()   # hash table with sets of PE routers for each MPLS service.

        # create and keep track of all routers in the network
        routers = dict()
        self.routers = routers
        self.fec_registry = FEC_registry()

        for n in topology.nodes():
            r = Router(self, n)
            routers[n] = r

    def compute_dijkstra(self, weight="weight"):
        # compute the shortest-path directed acyclic graph for each node (how to reach it)
        for n,r in self.routers.items():
            # compute the shortest-path directed acyclic graph for router n
            # this can be computationally expensive!
            r.compute_dijkstra(weight=weight)

    def start_client(self, client_class, **kwargs):
        # Create client_class clients on each router first,
        # then proceed to label allocation if required.
        for n,r in self.routers.items():
            # instantiate a client_class client on each router.
            client = r.create_client(client_class, **kwargs)

        if client_class.start_mode == 'auto':
            for n,r in self.routers.items():
                # Allocate labels for known managed resources
                client.alloc_labels_to_known_resources()


    def _get_build_order(self):
        # helper function to get the order of construction for forwarding rules
        prio = dict()
        for router_name, router in self.routers.items():
            for client_name, client in router.clients.items():
                if client.build_order not in prio.keys():
                        prio[client.build_order] = set()
                prio[client.build_order].add(router)

        order = sorted(list(prio.keys()))

        for o in order:
            yield (o,prio[o])

    def fec_lookup(self, fec):
        # work in progress
        return self.fec_registry.fec_lookup(fec)

    def LFIB_build(self):
        # helper function to generate LFIB entries on each router.
        for router_name, router in self.routers.items():
            router.LFIB_build()

    def LFIB_build_orderly(self):
        # helper function to generate LFIB entries on each router, respecting priorities
        for build_order, router_list in self._get_build_order():
            print("Now building order {}".format(build_order))
            #for router_name, router in router_list:
            for router in router_list:
                router.LFIB_build(build_order = build_order)

    def get_number_of_rules(self):
        N = 0
        # helper function to get the total number of rules in the network
        for router_name, router in self.routers.items():
            N += router.get_number_of_rules()
        return N

    def get_number_of_routing_entries(self):
        N = 0
        # helper function to get the total number of routing entries in the network
        for router_name, router in self.routers.items():
            N += router.get_number_of_routing_entries()
        return N

    def get_comm_count(self):
        N = 0
        # helper function to get the (min) number of communication exchanges among routers
        for router_name, router in self.routers.items():
            N += router.get_comm_count()
        return N

    def to_aalwines_json(self, weigth = "weight"):
        # This function generates a dictionary compatible with AalWiNes JSON schema.
        net_dict = {"network":  {"links": [], "name": self.name, "routers": []} }

        # Topology
        links = net_dict["network"]["links"]
        G = self.topology
        for from_router,to_router in G.edges():
            links.append({
                "bidirectional": True,
                "from_interface": to_router,
                "from_router": from_router,
                "to_interface": from_router,
                "to_router": to_router,
                "weight": G[from_router][to_router][weigth]
              })

        for router_name in G.nodes():
            router = self.routers[router_name]
            links.append({
                "from_interface": router.LOCAL_LOOKUP,
                "from_router": router_name,
                "to_interface": router.LOOPBACK,
                "to_router": router_name
              })

        # Forwarding rules
        for idx, router in self.routers.items():
            r_dict = router.to_aalwines_json()
            net_dict["network"]["routers"].append(r_dict)

        return net_dict

    def build_flow_table(self, verbose = False):
        # Build dict of flows for each routable FEC the routers know, in the sense
        # of only initiating packets that could actually be generated from the router.

        # classify according to fec_type
        print(f"Computing flows for simulation.")

        flows = dict()

        for router_name, r in self.routers.items():
            flows[router_name] = dict()
            if verbose:
                print(f"\n processing router {router_name}")

            for fec in r.LIB:
                ow = r.get_FEC_owner(fec)
                if verbose:
                    print(fec.name)
                good_sources_raw = ow.get_good_sources(fec)
                if not good_sources_raw:
                    continue
                good_sources = list(ow.get_good_sources(fec))
                # good_sources = ow.get_good_sources(fec)
                if not good_sources:
                    continue

                good_targets = ow.get_good_targets(fec)

                if router_name not in good_sources:
                    continue  # this router can be the source of a packet to this FEC

                in_label = r.get_label(fec)
                # Don't even try if this is not in the LFIB
                if in_label not in r.LFIB:
                    continue

                # I have good_sources and good_targets in memory currently...
                flows[router_name][in_label] = (good_sources,good_targets)

        return flows

    def visualize(self, router_list=None):
        # helper function to visualize LIB and LFIB tables of some router.
        # example:   router_list = [0,1,2]   , just provide the identifiers.

        if router_list:
            routers =  { r: self.routers[r] for r in router_list }
        else:
            routers = self.routers

        print(f"Number of forwarding rules in the Network:{self.get_number_of_rules()}")
        print("-"*20)
        for n,r in routers.items():
            # print LIB
            if r.LIB:
                print("{}.LIB".format(n))
                for a,b in r.LIB.items():
                    print("{}: {}".format(str(a), b))
                print()

            # print LFIB
            if r.LFIB:
                print("{}.LFIB".format(n))
                for a,b in r.LFIB.items():
                    print("{}: {}".format(str(a), b))
                print("-"*20)

    def visualize_fec_by_name(self, fec_name):
        for n,r in self.routers.items():

            if r.LIB:
                LIB_entries_by_name = [ (fec,entry) for (fec,entry) in r.LIB.items() if fec_name in fec.name]
                if LIB_entries_by_name:
                    local_labels =  [i[1]['local_label'] for i in LIB_entries_by_name]
                    print("{}.LIB".format(n))
                    for a,b in LIB_entries_by_name:
                        pprint("{}: {}".format(str(a), b))
                    print()

                    if r.LFIB:
                        print("{}.LFIB".format(n))
                        for ll in local_labels:
                            if ll in r.LFIB.keys():
                                pprint("{}: {}".format(str(ll), r.LFIB[ll]))
                    print("-"*20)


# Helper counter generator
def count(start=0, step=1, ext_manager = None):
    n = start
    while True:
        if not ext_manager:
            yield n
            n += step
        else:
            yield n + next(ext_manager)

class Label_Leaser(object):
    """
    EXPERIMENTAL. Leases label blocks.
    """
    def __init__(self, first_label=16, final_label = 1048576, block_size = 10000):

        self.first_label = first_label
        self.final_label = final_label
        self.block_size = block_size

        # check input
        if first_label > final_label:
            raise Exception("first label can't  be smaller than final label.")
        elif first_label < 0:
            raise Exception("First label can't be negative.")
        elif block_size < 1:
            raise Exception("block_size can't be less than 1.")

        self._cur_label = count(start=first_label, step = block_size) # basic label alloc manager

    def next_label_block(self):
        # Returns next available block of MPLS labels
        candidate_first_label = next(self._cur_label)
        if candidate_first_label >= self.final_label:
            raise Exception("MPLS labels depleted in router {}".format(self.name))

        candidate_end_label = candidate_first_label + self.block_size -1
        if candidate_end_label >= self.final_label:
            candidate_end_label = self.final_label

        return (candidate_first_label, candidate_end_label)


class Label_Manager(object):
    """
    A class that handles requests for new MPLS labels.
    """
    def __init__(self, Label_Leaser, numeric_labels=True):
        self.leaser = Label_Leaser
        self.label_registry = dict()
        self.fec_registry = dict()
        self._update()

    def _update(self):
        # Requests a label block to the leaser
        self.first_label, self.last_label = self.leaser.next_label_block()
        self._cur_label = count(start=self.first_label) # basic label alloc manager

    def set_fec_registry(self, fec_registry):
        # fec registry must be a dictionary.
        #  <mpls_fwd_gen.oFEC object at 0x7f60d84cefa0>: {'local_label': '126',
        #                                        'owner': <ldp.ProcLDP object at 0x7f60d8557280>}}
        #  oFEC: {'local_label': 'string label' , 'owner': MPLS_Client}
        self.fec_registry = fec_registry

    def next_label(self, fec):
        # Allocates a new label in the MPLS range
        fid = fec
        if fid in self.label_registry:
            # print(f"already allocated label {self.label_registry[fid]}")
            return self.label_registry[fid]  #already allocated.

        fec = self.fec_registry.fec_lookup(fec)   # gets sure the fec is into the registry

        candidate_label = next(self._cur_label)

        if candidate_label > self.last_label:
            self._update()
            candidate_label = next(self._cur_label)   # it will fail if depleted.

        self.label_registry[fid] = candidate_label
        return candidate_label


class Global_Label_Manager(Label_Manager):
    """
    EXPERIMENTAL Global label manager, tailored towards the
    # global segment id case! (move to sr code?)
    """
    def __init__(self, Label_Leaser,  numeric_labels=True, global_sid_manager = None):
        if not global_sid_manager:
            self.global_sid_manager = count(1)
        else:
            self.global_sid_manager = global_sid_manager
        super().__init__(Label_Leaser)

    def _update(self):
        # Requests a label block to the leaser
        self.first_label, self.last_label = self.leaser.next_label_block()
        self._cur_label = count(start=self.first_label, ext_manager = self.global_sid_manager) # basic label alloc manager

    # def next_label(self, fec):
    #     # print(f"requesting label for {fec.name}: ", end="")
    #     fid = fec
    #     if fid in self.label_registry:
    #         # print(f"already allocated label {self.label_registry[fid]}")
    #         return self.label_registry[fid]  #already allocated, global sync.
    #
    #     fec = self.fec_registry.fec_lookup(fec)   # gets sure the fec is into the registry
    #
    #     # Allocates label in the MPLS range
    #     candidate_label = self.first_label + next(self.global_sid_manager)
    #
    #     if candidate_label > self.last_label:
    #         self.first_label, self.last_label = self.leaser.next_label_block()
    #         candidate_label = self.first_label + next(self.global_sid_manager)
    #
    #     self.label_registry[fid] = candidate_label
    #     print(f"New allocated label {self.label_registry[fid]}")
    #     return candidate_label

class Router(object):
    """
    A class for (MPLS) router. Composed by:

    - A Label Information Base (LIB), that allocates a local MPLS label to resources identified by a FEC.
    - A Label Forwarding Information Base (LFIB), that keeps forwarding instructions for each (local) MPLS label.
    - A table keeping track of registered MPLS clients.

    The router knows its interfaces from the network topology it belongs to.
    It is also responsible for computation of the shortest-path from other routers towards itself, that is,
    the directed acyclic graph for node n (how to reach this node).
    """

    def __init__(self, network, name, alternative_names=[], location = None, php = False, dist=None, pred=None, paths=None, first_label=16,
                 max_first_label = 90000, seed=0, numeric_labels=True):
        self.name = name
        self.network = network
        self.topology = network.topology
        self.location = location  # "location": {"latitude": 25.7743, "longitude": -80.1937 }
        self.alternative_names = alternative_names
        self.PHP = php   # Activate Penultimate Hop Popping
        self.numeric_labels = numeric_labels

        # Define the LOCAL_LOOKUP interface: the interface to which we should send packets that
        # require further processing.
        # case 1: After the are stripped of all their labels and must be forwarded outside of
        #         the MPLS domain.
        # case 2: If they require to be matched again agansts the LFIB after a succesful local pop().
        #         This is the case for recursive MPLS forwarding.
        #
        self.LOCAL_LOOKUP = "local_lookup"

        # LOOPBACK interface. Interface that implements recursive forwarding: will receive the packets send to
        # LOCAL_LOOKUP.
        self.LOOPBACK = "loop_back"

        # Initialize tables
        self.LFIB = dict()
        self.LIB = dict()
        self.clients = dict()
        self.main_label_leaser = Label_Leaser(first_label=16,
                                                final_label = 1048576,
                                                block_size = 10000)

        # Attributes to store shortest path spanning tree
        self.dist = dist                       # Distance from other nodes
        self.paths = paths                     # Paths from other nodes
        self.pred = pred                       # Predecesors (upstream nodes) for each router towards this
        self.succ = self.get_successors(pred)  # next-hops

        random.seed(seed)
        start = random.randint(first_label,max_first_label)


    def get_location(self):
        # get geographical location of router.
        #{"latitude": latitude, "longitude": longitude}
        if self.location:
            return self.location

        G = self.topology

        try:
            location = {"latitude": G.nodes[self.name]["latitude"],
                    "longitude": G.nodes[self.name]["longitude"]}
            self.location = location
            return self.location

        except:
            return None

    def compute_dijkstra(self, weight="weight"):
        # Compute the shortest-path directed acyclic graph in-tree (how to reach this node)
        # Relies on networkx._dijkstra_multisource()

        # Initializations
        paths = {self.name: [self.name]}  # dict of paths from each node
        pred = {self.name: []}    # dict of predecessors from each node

        # compute paths and distances
        G = self.topology
        weight = _weight_function(G, weight)
        dist = _dijkstra_multisource( G, {self.name}, weight, pred=pred, paths=paths )

        # store results for this router.
        self.dist = dist  # dict of distances from each node
        self.pred = pred
        self.succ = self.get_successors(pred) # dict of successors from each node
        self.paths = paths

    def get_successors(self, pred):
        # Given the predecesors it the router in-tree, return the dictionary of succesors (next-hops)
        if not pred:
            return None

        succ = dict()

        for router in pred.keys():
            succ[router] = list()
            for r, predecessor_list in pred.items():
                if router in predecessor_list:
                    succ[router].append(r)

        return succ

    def get_interfaces(self, outside_interfaces = False):
        # Returns generator of router interfaces (identified as router names).
        # If outside_interfaces is False (default) it will return all enabled MPLS
        # interfaces (internal interfaces). If outside_interfacs is True,
        # it will only return non-MPLS interfaces, for example the ones used on
        # VPN services as attachement circuits (AC).

        #return chain(iter([self.name]), self.topology.neighbors(self.name))
        if not outside_interfaces:
            yield self.name

            for neigh in self.topology.neighbors(self.name):
                yield neigh

        else:
            # add all service interfaces:
            if "service" in self.clients:
                for vpn_name, vpn in self.clients["service"].services.items():
                    for ce in vpn["ces"]:
                        yield ce

    def get_client_by_name(self, proto_name):
        # Returns registered client of protocol proto_name if exists, None otherwise
        if proto_name in self.clients.keys():
            return self.clients[proto_name]
        else:
            return None

    def get_client(self, mpls_client):
        # Returns registered client of class mpls_client if it existes, None otherwise
        return self.get_client_by_name(mpls_client.protocol)
        # if mpls_client.protocol in self.clients.keys():
        #     return self.clients[mpls_client.protocol]
        # else:
        #     return None

    def create_client(self, client_class, **kwargs):
        #Creates and registers in this router an MPLS client of class client_class.
        #If a client of same class is already registered return error
        #Only one client per protocol.
        if not issubclass(client_class, MPLS_Client):
            raise Exception("A subclass of MPLS_Client is required.")

        if self.get_client(client_class):
            raise Exception("Client already exists...")

        new_client = client_class(self, **kwargs)  # we refer this client to ourselves.
        self.register_client(new_client)
        return new_client

    def register_client(self, mpls_client):
        # Adds the mpls_client instance to the router's registry.
        # Only one client per protocol is accepted, will override a previous entry.
        self.clients[mpls_client.protocol] = mpls_client
        mpls_client.create_label_manager(self.main_label_leaser)

    def remove_client(self, client_class):
        if not issubclass(client_class, MPLS_Client):
            raise Exception("A subclass of MPLS_Client is required.")

        if self.get_client(client_class):
            del self.clients[mpls_client.protocol]
            # FIXME: what about the LIB and LFIB entries it used to own?
        raise  Exception("No registered client of class to remove...".format(type(client_class)))


    def get_FEC_owner(self, FEC):
        # Return the process owning the FEC.
        if FEC not in self.LIB.keys():
            return None
        else:
            return self.LIB[FEC]["owner"]


    def LIB_alloc(self, process, FEC, new_label, literal = None):
        # Allocates a local MPLS label to the requested FEC and return the allocated label.
        # If already allocated, return the label.
        # If literal is not None and self.numeric is False, then literal must be a NON-NUMERIC
        #   string to be directly allocated to the LIB (Main use: MPLS VPN service interfaces)

        if FEC not in self.LIB.keys():
            # Allocate if it hasn't been yet
            if self.numeric_labels:
                self.LIB[FEC] = {"owner": process, "local_label": new_label }
            elif literal and isinstance(literal, str) and not literal.isnumeric():
                self.LIB[FEC] = {"owner": process, "local_label": literal }
            else:
                self.LIB[FEC] = {"owner": process, "local_label": str(new_label) }
            return self.LIB[FEC]["local_label"]

        #FEC is already in binding database.
        if self.get_FEC_owner(FEC) != process:
            print("The FEC is under control of other mpls client process {}".format(self.LIB[FEC]["owner"]))
            raise Exception("The FEC is under control of other mpls client process {}".format(self.LIB[FEC]["owner"]))
            return "The FEC is under control of other mpls client process {}".format(self.LIB[FEC]["owner"])

        # We just return the allocated label.
        curr_label = self.get_label(FEC)
        if not curr_label:
            raise Exception(f"[{self.name}]: Found FEC{FEC.name} without label on LIB.")
        elif self.numeric_labels and curr_label != new_label:
            # relabeling attempt? Currently forbidden.
            raise Exception(f"[{self.name}]: Attempt to bind label {new_label} to FEC {FEC.name} with current label {curr_label}.")
        elif not self.numeric_labels and curr_label != str(new_label):
            # relabeling attempt? Currently forbidden.
            raise Exception(f"[{self.name}]: Attempt to bind label {new_label} to FEC {FEC.name} with current label {curr_label}.")
        return self.get_label(FEC)

    # Juan : I am not sure if this function is valid or a control-plane violation.
    def get_FEC_from_label(self, label, get_owner = False):
        # Return the FEC corresponding to a label.
        candidates =  [z for z in self.LIB.items() if z[1]['local_label'] == label]  # 0 or 1 candidates only
        if not candidates:
            return None
        if get_owner:
            return candidates[0]      # get LIB data
        else:
            return candidates[0][0]   # just the fec

    def get_fec_by_name_matching(self, name_substring):
        # Return generator of all locally known FEC whose name contains name_substring
        return [ fec for fec in self.LIB.keys() if name_substring in fec.name ]

    def get_label(self, FEC):
        # Get the label allocated to a FEC
        if FEC in self.LIB.keys():
            return self.LIB[FEC]["local_label"]
        return None

    def get_label_from_fec_name(self, name_substring):
        # FIXME: shouldn't this be in another class or something?
        FEC_list = self.get_fec_by_name_matching(name_substring)
        if not FEC_list:
            raise Exception(f"Can't identify a label for flow from {router_name} and fec starting with {name_substring} (matches: {len(FEC_list)})")
        elif len(FEC_list) > 1:
            raise Exception(f"Too many labels match flow from {router_name} and fec starting with {name_substring}. Flow specification insufficient. (matches: {len(FEC_list)})")

        # We know now that there is a single entry.
        FEC = FEC_list[0]
        return self.get_label(FEC)

    def get_routing_entry(self, label):
        # Get the routing entry allocated to a local label
        if label in self.LFIB.keys():
            return self.LFIB[label]
        return None

    def get_routing_entries_from_FEC(self, FEC):
        # Get the label allocated to a (locally registered FEC)
        if FEC in self.LIB.keys():
            return self.get_routing_entry(self.LIB[FEC]["local_label"])

    def compare_routing_entries(self, re1, re2):
        # check if two routing entries are functionally equivalent.
        # quick checks:
        if re1 == re2:
            return True

        if len(re1) != len(re2):
            return False

        #slow check.
        for i in range(len(re1)):
            if ( re1[i] not in re2 ) or (re2[i] not in re1 ):
                return False

        return True

    def LFIB_alloc(self, local_label, routing_entry):
        # NOTE OF FIXME: currently only being used by refinement purposes. Does this makes sense?
        # Append a routing entry for a local label in
        # the LFIB (Label Forwarding Information Database).
        # Currently used only for refinement processes.
        if local_label in self.LFIB.keys():
            if routing_entry not in self.LFIB[local_label]:
                self.LFIB[local_label].append(routing_entry)
        else:
            self.LFIB[local_label] = [routing_entry]

    def LFIB_build_for_fec(self, fec, mpls_client):
        # EXPERIMENTAL
        temp = []
        local_label = None
        for label, routing_entry in mpls_client.LFIB_compute_entry(fec):  # for each computed entry...
            # print("compute on router ", self.name," for  ", fec.name, label, routing_entry )
            if not local_label:
                local_label = label  # We expect a single label per fec
            if fec in self.LIB.keys() and self.LIB[fec]["owner"].protocol == mpls_client.protocol:
                # print("local_label set to ", local_label, " exists on LIB")
                # If fec exists and is managed by this client, allocate the routing entry.
                if routing_entry not in temp:
                        temp.append(routing_entry)
                else:
                    temp = [routing_entry]

        if not local_label:
            return

        if local_label not in self.LFIB:
            self.LFIB[local_label] = temp
        else:
            cur = self.LFIB[local_label]
            if not self.compare_routing_entries(temp, cur):
                #update if different
                print("    updating different routing entries")
                self.LFIB[local_label] = temp
               # self.LFIB_alloc(label, routing_entry)
            else:
                pass
                # print("    no need to update! idempotence in action!")

    def LFIB_build(self, build_order = None):  #FIXME: make idempotent!
        # Proc to build the LFIB from the information handled by each MPLS client
        # build_order = None (default) means attempt to build from all MPLS clients (ignore dependencies)
        print(f"BUILD LFIB for router {self.name}")
        for mpls_client_name, mpls_client in self.clients.items():    #iterate through all registered clients
            if not build_order or build_order == mpls_client.build_order:
                for fec in mpls_client.known_resources():  #iterate through the client resources
                    # for label, routing_entry in mpls_client.LFIB_compute_entry(fec):  # for each computed entry...
                    #     if fec in self.LIB.keys() and self.LIB[fec]["owner"].protocol == mpls_client_name:
                    #         # If fec exists and is managed by this client, allocate the routing entry.
                    #         self.LFIB_alloc(label, routing_entry)
                    self.LFIB_build_for_fec(fec, mpls_client)

    def LFIB_weights_to_priorities(self):
        # Compute the priority of each route entry, remove weight, cast label as string...
        for label in self.LFIB.keys():
            metrics = sorted(set([x['weight'] for x in self.LFIB[label]]))
            for entry in self.LFIB[label]:
                entry['priority'] = metrics.index(entry["weight"])
                entry.pop("weight")

    def LFIB_refine(self):
        # Proc to refine (post-process) the LFIB
        for mpls_client_name, mpls_client in self.clients.items():    #iterate through all registered clients
            for label in self.LFIB.keys():
                new_rules = mpls_client.LFIB_refine(label)
                if new_rules:
                    for new_rule in new_rules:
                        self.LFIB_alloc(label,new_rule)

    def get_number_of_rules(self):
        return len(self.LFIB.keys())

    def get_number_of_routing_entries(self):
        return np.sum([len(v) for v in self.LFIB.values()])

    def get_comm_count(self):
        return np.sum([s.get_comm_count() for s in self.clients.values()])

    def self_sourced(self, fec):
        # finds and calls right mpls client self_sourced() function
        owner = self.get_FEC_owner(fec)
        if not owner:
            return False
        try:
            return owner.self_sourced(fec)
        except:
            return False


    def to_aalwines_json(self):
        # Generates an aalwines json schema compatible view of the router.
        r_dict= {"interfaces": [], "name": str(self.name)}

        if self.alternative_names is list and len(self.alternative_names) > 0:
            r_dict["alias"] = [str(a) for a in self.alternative_names]

        # Now comes the first problem, as the FLIB is ordered by incoming interface...
        # We will just copy it on each interface.

        def _call(t):
            if isinstance(t[0],str) and t[0].startswith("NL_"):
                return False
            return True

        # def _call2(t):
        #     if isinstance(t[0],str) and t[0].startswith("NL_"):
        #         return False
        #     return True

        regular_LFIB = dict_filter(self.LFIB, _call)
        service_LFIB = dict_filter(self.LFIB, lambda t: not _call(t))

        ifaces = set()
        for x in self.get_interfaces():
            if x == self.name:
                iface = "i"+str(x)
            else:
                iface = str(x)
            ifaces.add(iface)
            # cases not yet implemented:
            #   - entries outputing to LOCAL_LOOKUP
            #   - multiple loopback interfaces


        ifaces.add(self.LOOPBACK)
        r_dict["interfaces"].append({"names":list(ifaces), "routing_table": regular_LFIB})

        r_dict["interfaces"].append({"name":self.LOCAL_LOOKUP, "routing_table": {}   })

        # process outside interfaces (non-MPLS)
        from service import MPLS_Service  # is this right?
        for x in self.get_interfaces(outside_interfaces = True):
            iface = str(x)
            # get service for this interface...

            s = self.get_client(MPLS_Service)
            vdict = s.get_service_from_ce(iface)
            vpn_name = list(vdict.keys())[0]
            def _call3(t):
                if isinstance(t[0],str) and vpn_name in t[0]:
                    return True
                return False

            srv_iface_LFIB = dict_filter(service_LFIB, _call3)

            rt = {"null": []}
            for _,re_list in srv_iface_LFIB.items():
                for re in re_list:
                    rt["null"].append(re)
            r_dict["interfaces"].append({"name":iface, "routing_table": rt })

        loc = self.get_location()
        if loc:
            r_dict["location"] = dict()
            r_dict["location"]["latitude"] = round(loc["latitude"], 4)     #fixed precision
            r_dict["location"]["longitude"] = round(loc["longitude"], 4)
        return r_dict

class oFEC(object):
    """
    Models a Forwarding Equivalence Class.
    This involves all packets that should be forwarded in the same fashion in the MPLS network.
    Commonly used to represent a network resource or group of resources.
    Examples: An IP prefix, a TE tunnel/LSP, a VRF, etc.

    FEC objects are managed by a MPLS client process and registered on a router's LIB.
    Required information for initialization:

    - fec_type: string, mandatory.
                Any value representing the fec type.

    - name:     string, mandatory.
                The name to identify this FEC in particular.

    - value:    arbitrary, optional. Defaults to None.
                Any kind of information that makes sense for this given FEC.
                Can be used for additional information, metadata, providing context, etc.

    Both 'name' and 'fec_values' are considered immutable so they must never be changed.
    The class allows to check equality of two oFEC objects directly.
    """

    def __init__(self, fec_type, name, value = None):
        self.fec_type = fec_type
        self.name = name
        self.value = value

    def __hash__(self):
        return hash((self.fec_type, self.name, self.value))

    def __eq__(self, other):
        return isinstance(other, oFEC) and self.value == other.value and self.name == other.name and self.fec_type == other.fec_type

    def __str__(self):
        return "{}({})".format( self.value, self.name, self.fec_type)

class MPLS_Client(object):
    """
    Abstract class with minimal functionality and methods any client must implement.

    An MPLS client is a FEC manager process on a router. It is resposible for:

    - Computing the resulting outcome of a given MPLS control plane protocol
    - Request labels for its FECs on the router's LIB
    - Provide functions to compute appropiate routing entries for LFIB contruction

    For initialization indicate the router.
    """

    #static attribute: start_mode
    # if auto, the network will try to allocate labels to local resources right after initialization.
    start_mode = 'manual'
    EXPLICIT_IPV4_NULL = 0
    EXPLICIT_IPV6_NULL = 2
    IMPLICIT_NULL = 3

    def __init__(self, router, build_order = 100):
        self.router = router
        self.build_order = build_order
        self.LOCAL_LOOKUP = router.LOCAL_LOOKUP
        self.comm_count = 0

    # Common functionality
    def get_comm_count(self):
        return self.comm_count

    def access_remote(self, router_name, router_function, do_count=True):
        if self.router.name == router_name:
            return router_function(self.router)
        else:
            if router_name not in self.router.network.routers.keys():
                return None
            if do_count:
                self.comm_count += 1
            return router_function(self.router.network.routers[router_name])

    def access_client(self, router_name, client_class, client_function, do_count=True):
        def use_client(router):
            client = router.get_client(client_class)
            return client_function(client) if client else None
        return self.access_remote(router_name, use_client, do_count=do_count)

    def alloc_labels_to_known_resources(self):
        # Asks the router for allocation of labels for each known FEC (resource)
        for fec in self.known_resources():
             self.LIB_alloc(fec)

    def get_remote_label(self, router_name, fec, do_count=True):
        # Gets the label allocated by router <router_name> to the FEC <fec>
        router = self.router.network.routers[router_name]  # FIXME: Why not self.comm_count += 1 here?
        if router.php and router.self_sourced(fec):
            return self.IMPLICIT_NULL
        else:
            return self.access_remote(router_name, lambda router: router.get_label(fec), do_count=do_count)

    def get_local_label(self, fec):
        # Gets the local label allocated to the FEC <fec>
        return self.get_remote_label(self.router.name, fec)

    def get_fec_by_name_matching(self, name_substring):
        # Return generator of all locally known FEC whose name contains name_substring
        # return [ fec for fec in self.router.LIB.keys() if name_substring in fec.name ]
        return self.router.get_fec_by_name_matching(name_substring)

    def LIB_alloc(self, fec, literal = None):
        # Wrapper for calling the routerś LIB_alloc function.
        next_label = self.main_label_manager.next_label(fec)
        return self.router.LIB_alloc(self, fec, next_label, literal = literal)

    def create_label_manager(self, label_leaser = None):
        if not label_leaser:
            label_leaser = self.router.main_label_leaser
        self.main_label_manager = Label_Manager(label_leaser,numeric_labels=True)
        self.main_label_manager.set_fec_registry(self.router.network.fec_registry)

    # Abstract functions to be implemented by each client subclass.
    def LFIB_compute_entry(self, fec, single = False):
        # Each client must provide an generator to compute routing entries given the fec.
        # optional parameter "single" forces the function to return just one routing entry.
        # returns tuple (label, routing_entry)
        # routing entries have format:
        #  routing_entry = { "out": next_hop_iface, "ops": [{"pop":"" | "push": remote_label | "swap": remote_label}], "weight": cost  }
        # A few rules regarding the ops:
        #
        # Rule 1:          NOp := [{"push":x}, {"pop":""}]  # should never appear in an entry.
        # Rule 2: {"swap": x } := [{"pop":""}, {"push":x}]  # can be analized with just 2 operations.
        # Corollary 1: All ops can be written with just pop and push operations
        # Corollary 2: All ops must have a form like:  [{"pop":""}^n, prod_{i=1}^{m} {"push": x_i}]
        #              which in turn can have at most one swap operation at the deepest inspected stack
        #              level, so (if m > 1 and n>1):
        #                 [{"pop":""}^{n-1}, {"swap": x_1} ,prod_{i=2}^{m} {"push": x_i}]
        pass

    def LFIB_refine(self, label):
        # Some process might require a refinement of the LFIB.
        pass

    def known_resources(self):
        # Returns a generator to iterate over all resources managed by the client.
        # Each client must provide an implementation.
        pass

    def self_sourced(self, FEC):
        # Returns True if the FEC is sourced or generated by this process.
        pass

    def get_good_sources(self, fec):
        # simulater helper method
        # must return a list of valid source routers for the fec`
        pass

    def get_good_targets(self, fec):
        # simulater helper method
        # must return a list of valid target routers for the fec`
        pass


class ProcStatic(MPLS_Client):
    """
    Class implementing static MPLS forwarding entries, or to override labels previously
    allocated by other MPLS protocols.

    This is the more flexible yet administratively expensive option to build MPLS
    forwarding; there is no automatic interaction between routers for this mechanism,
    so the admin is responsible for building hop by hop tunnels.

    It can also 'hijack' local labels owned by other protocols to implement hacks.

    Manages the tunnels as resources/FECs.

    This class has the following structures:

    static_mpls_rules:
        List of TE tunnels that are requested to start on this node.
        This is esentially configuration for the setup of new tunnels.
        Entry format:
            (static) lsp name => static FEC, 'label_override': <local label to be instered or overriden>

    The static FEC is of type "STATIC_LSP" and includes:
        - lsp_name -> the FEC name
        - value -> manual routing information.0

    For initialization indicate the router.
    """

    start_mode = 'manual'  # All static labels must be allocated manually.
    protocol = "STATIC"

    def __init__(self, router):
        super().__init__(router)
#         self.protocol = "STATIC"
        self.static_mpls_rules = dict()
        self._cur_count = count()  #Counter for autogenerated names of static entries.

    def define_lsp(self, lsp_name, ops, outgoing_interface, incoming_interface = None,
                   priority=0, label_override = None):
        #
        # Define a static MPLS entry.
        # Inputs:
        # - lsp_name: the FEC name. Must be unique per router. To autogenerate, set lsp_name="".
        # - ops: The full ordererd set of MPLS stack manipulation for a matching packet.
        # - outgoing_interface: Next hop interface.
        # - incoming_interface: For per interface label space. Defaults to None (per-platform label space)
        # - priority: routing entry priority.
        # - label_override: the local label to be inserted. If it is already allocated, then it
        #                   will be overriden. Leave empty is a fresh new label must be allocated.

        # TODO: Add verifications here.

        # Create an automatic incremental local lsp name.
        if lsp_name == "":
            lsp_name = "static_lsp_{}_{}".format(self.router.name, next(self._cur_count))

        #compose the routing entry.
        data = {'ops':ops, 'out': outgoing_interface, 'priority': priority}
        if not incoming_interface:
            data['in'] = incoming_interface

        # Insert the request on the table
        self.static_mpls_rules[lsp_name] = { 'FEC': oFEC("STATIC_LSP", lsp_name, value = data),
                                             'label_override': label_override}
        return lsp_name

    def known_resources(self):
        # Returns a generator to iterate over all requested static MPLS entries.
        if not self.static_mpls_rules:
            # If there are no requests, we are done.
            return None

        for lsp_name, lsp_data in self.static_mpls_rules.items():
            fec = lsp_data['FEC']
            yield fec


    def alloc_labels_to_known_resources(self):
        # Requests the router for allocation of labels for each known FEC (resource).
        # The inherited functionality is not enough for the case of label override.

        for fec in self.known_resources():
            if self.static_mpls_rules[fec.name]['label_override']:
                # Get requested label
                local_label = self.static_mpls_rules[fec.name]['label_override']

                # If exists, delete previously allocated FEC for that label.
                prev_FEC = self.router.get_FEC_from_label(local_label)
                if prev_FEC:
                    del self.router.LIB[prev_FEC]

                # manual forced entry in LIB (a hack)
                self.router.LIB[fec] = {'owner': self.process, 'local_label': local_label }

            else:
                # Regular case
                self.LIB_alloc(fec)


    def LFIB_compute_entry(self, fec, single = False):
        # Each client must provide an implementation.
        # Return a generator for routing entries for the
        # requested static FEC: returns a generator of
        # (label, routing_entry).

        router = self.router
        network = router.network

        if not fec.value:
            return

        # Compose the routing entry from the FEC information.
        ops = fec.value['ops']
        out = fec.value['out']
        priority = fec.value['priority']
        routing_entry = { "out": next_hop.name, "ops": ops, "priority": priority  }

        # If required append incoming interface condition.
        if 'in' in fec.value.keys():
            incoming_interface = fec.value['in']
            routing_entry['incoming'] = incoming_interface

        # Get local label from LIB
        local_label = self.get_local_label(fec)

        yield (local_label, routing_entry)
