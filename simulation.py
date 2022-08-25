################################################################################
### SIMULATION FUNCTIONALITIES AND CLASSES

"""
###################################

     MPLS FORWARDING SIMULATOR

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
# import sys

# import networkx as nx
# import matplotlib.pyplot as plt
# import random
# import time

# import json
# import math
# import copy
# from pprint import pprint
# from itertools import chain

# import networkx.algorithms
# import numpy as np

# from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra_multisource
# from resource import getrusage, RUSAGE_SELF

from mpls_fwd_gen import *
import networkx as nx

class MPLS_packet(object):
    """
    Class intended for packet tracing, for debugging or testing purposes.

    Each packet has a stack,
    and records its paths while it is forwarded through the network in variables
    "traceroute" and "link_traceroute".

    Params:
    - network             : The network object on which the packet will live.
    - init_router         : Initial router (on th eMPLS domain) for the packet.
    - init_stack          : (default empty list). The stack the packet has initially.
    - mode                : 'packet'      -- simulates a single packet forwarding. default.
                            'pathfinder'  -- identifies all possible paths in a given topology. Costly.
    - restricted_topology : Topology with missing links/nodes for the simulation.
    - verbose             : (default False). Outpt lots of information.

    This is just a proof of concept, must be further developed.
    """

    def __init__(self, network, init_router, init_stack = [], restricted_topology = None, mode="packet", max_ttl = 255, verbose = False):
        self.network = network
        self.ttl = max_ttl
        if restricted_topology is not None:
            self.topology = restricted_topology
        else:
            self.topology = network.topology

        self.stack = init_stack
        self.init_stack = init_stack.copy()

        self.mode = mode   # options: packet, pathfinder.

        if isinstance(init_router, str):
            self.init_router = network.routers[init_router]
        elif isinstance(init_router, Router):
            self.init_router = init_router
        else:
            raise Exception("Unknown init_router")

        self.traceroute = [self.init_router]                       # list of router
        self.link_traceroute = [(self.init_router,None)]           #  list of links (as tuples)
        self.trace = [(self.init_router,None,self.init_stack)]     # list of links and stack
        self.state = "uninitialized"
        self.verbose = verbose
        self.alternatives = []
        self.cause = ""
        self.exit_code = None     # 0 means success, None is not finished yet, other are errors.
        self.success = None

    def info(self):
        print(".....INFO.....")
        print("Packet from {} with initial stack: [{}]" .format(self.init_router.name, "|".join(self.init_stack)))
        print("   current state: {}".format(self.state))
        print("   current cause: {}".format(self.cause))
        print("   current path: {}".format(self.traceroute))
        # print("   detailed path: {}".format(self.trace))
        print("   current stack: {}".format("|".join(self.stack)))
        print("...INFO END...")

    def get_next_hop(self, outgoing_iface, verbose = True):
        # this implementation depends explicitly on interfaces named after the connected router
        curr_r = self.traceroute[-1]
        if verbose:
            print("Into get_next_hop")
            print(f" current_router {curr_r.name}")
            self.info()
            print(f"neighbors: {list(self.topology.neighbors(curr_r.name))}")
            print(f"Is {outgoing_iface} a neighbor?")
            print(outgoing_iface in self.topology.neighbors(curr_r.name))

            print("edges:")
            pprint(self.topology.edges())

            print(f"Let's see if we have a edge towards {outgoing_iface} ")
            print(  self.topology.has_edge(curr_r.name,outgoing_iface))
        if outgoing_iface in self.topology.neighbors(curr_r.name) and self.topology.has_edge(curr_r.name,outgoing_iface):
            return self.network.routers[outgoing_iface]
        else:
            return None

    def step(self):
        # Simulate one step: send the packet to the next hop.
        # packet mode: returns boolean success
        # pathfinder mode: returns tuple: (success,[packet list])
        if self.state == "uninitialized":
            raise Exception("Can't move an UnInitialized packet.")

        if not self.stack:
            self.state = "finished"
            self.cause = " FORWARDING Complete: empty stack."
            self.exit_code = 0
            if self.verbose:
                print(self.cause)
                print("Current router: {}".format(self.traceroute[-1].name))
                print("Current outmost label: {}".format(None))
                print("stack: {}".format([]))
                print()
            if self.mode == "pathfinder":
                return (True, [])
            return True

        curr_r = self.traceroute[-1]    # current router
        outer_lbl = self.stack[-1]      # outmost label

        if self.verbose:
            print()
            print("Current router: {}".format(curr_r.name))
            print("Current outmost label: {}".format(outer_lbl))

        # check and update time-to-live
        if self.ttl <= 0:
            self.state = "finished"
            self.cause = " FORWARDING Aborted: Time to live expired!"
            self.exit_code = 1
            if self.mode == "pathfinder":
                return (False, [])
            return False  # Time to live expired; possible loop.
        else:
           self.ttl -= 1

        # gather the forwarding rules.
        if outer_lbl in curr_r.LFIB:
            rules = curr_r.LFIB[outer_lbl]
        else:
        # if there are no rules
            self.state = "finished"
            self.cause = " FORWARDING Complete: No available forwarding rules at router {} for label {}, yet MPLS stack is not empty".format(curr_r.name, outer_lbl)
            self.exit_code = 2
            if self.verbose:
                print(self.cause)
            if self.mode == "pathfinder":
                return (False, [])
            return False  # Return false because there are no rules here yet the MPLS stack is not empty

        # get all available priorities in the rules
        priorities = sorted(list(set([x['priority'] for x in rules ])))

        # code split here (between packet and pathfinder modes)...
        rule_list = []   # store here the acceptable forwarding rules.

        # now consider only the routes from higher to lower priority
        for prio in priorities:
            f_rules = list(filter(lambda x: x["priority"] == prio, rules))
            if len(f_rules) > 1:
                random.shuffle(f_rules)   #inplace random reorder, allows balancing, non determinism.
                # FIXME: id this really the place where we should do this?
                # FIXME: can this be made compatible with weights?

            for candidate in f_rules:
                # determine validity of outgoing interface and next_hop
                outgoing_iface = candidate["out"]
                next_hop = self.get_next_hop(outgoing_iface, verbose = False)

                # if it points to LOCAL_LOOKUP then that's it, that SHOULD NOT be balanced.
                # if not, but next_hop is reacheable in our topology, we also found a valid rule.
                # FIXME: this doesn't make sense, what if we have randomized earlier?
                if outgoing_iface == curr_r.LOCAL_LOOKUP:
                    #we found an usable route!
                    rule_list = [candidate]
                    break
                elif next_hop:
                    rule_list.append(candidate)
                elif not next_hop and outer_lbl:
                    #MPLS stack still not empty, maybe a VPN service CE?

                    try:
                        srvs = curr_r.clients["service"].services
                        y = [x['ces'] for x in srvs.values()]
                        if outgoing_iface in chain(*y):  #check local CEs
                            candidate["x-leaving"] = True
                            rule_list.append(candidate)
                    except:
                        # try next candidates
                        # print("broken candidate?")
                        continue
                else:
                    # Exiting the MPLS domain.
                    self.state = "finished"
                    self.cause = " FORWARDING Complete: Exiting the MPLS domain at router {} for label {}".format(curr_r.name, outer_lbl)
                    self.exit_code = 3
                    if self.verbose:
                        print(self.cause)
                    if self.mode == "pathfinder":
                        return (True, [])
                    return True  # I ended up considering this condition as a SUCCESS.

            if rule_list:
                # if this priority level has any usable rule we are done and move on.
                break

        # If we couldn't find a rule that is the end of the story.
        if not rule_list:
            self.state = "finished"
            self.cause = " FORWARDING Complete: Can't find a forwarding rule at router {} for label {}".format(curr_r.name, outer_lbl)
            self.exit_code = 4
            if self.verbose:
                print(self.cause)
            if self.mode == "pathfinder":
                return (False, [])
            return False

        p_list = []
        for i in range(len(rule_list)):

            if i < len(rule_list) - 1:
                if self.mode == "packet":
                    continue               # when forwarding a single packet we process just once.
                p = copy.deepcopy(self)    # when pathfinding, advance copies before myself
            else:
                p = self

            rule = rule_list[i]
            outgoing_iface = rule["out"]
            next_hop = p.get_next_hop(outgoing_iface, verbose = False)

            if "x-leaving" in candidate.keys():
                # Service packet exiting the MPLS domain towards a customer's CE
                candidate.pop("x-leaving")
                outgoing_iface = curr_r.LOCAL_LOOKUP

            if outgoing_iface == curr_r.LOCAL_LOOKUP:
                p.cause = " FORWARDING recurrent: attempt to process next level on {}".format(curr_r.name)
                self.exit_code = 5
                if p.verbose:
                    print(p.cause)
                # note: not finished yet!
                # Recycle. We won't record and just run again on the same router, after processing the stack

            else:
                p.traceroute.append(next_hop)
                p.link_traceroute[-1] = (p.link_traceroute[-1][0], outgoing_iface)
                p.link_traceroute.append((next_hop, None))
                p.trace[-1] = (p.trace[-1][0],outgoing_iface, p.trace[-1][2])
                p.trace.append((next_hop,None,self.stack))
                # p.trace.append("|".join(self.stack))


            # Processing the stack
            ops_list = rule["ops"]
            for op in ops_list:
                if "pop" in op.keys():
                    p.stack.pop()
                elif "push" in op.keys():
                    p.stack.append(op["push"])
                elif "swap" in op.keys():
                    p.stack.pop()
                    p.stack.append(op["swap"])
                else:
                    raise Exception("Unknown MPLS operation")

            if p.verbose:
                print(ops_list)
                print("matching rules: {}".format(len(rules)))
                print("fwd:")
                print(rules)
                if next_hop:
                    print("NH: {} ".format(next_hop.name))
                print("stack: {}".format(p.stack))
                print()

            p_list.append(p)

        if self.mode == "pathfinder":
            return (True, p_list)
        return True

    def initialize(self):
        if self.state == "uninitialized":
            self.state = "live"
            return True
        elif self.state == "live":
            # already initilizated.
            return True
        return False

    def fwd(self, random_seed = None):
        # Control high level forwarding logic.
        # we make sure that we can forward the packet.
        if not self.initialize():
            return False

        if random_seed:
            # We received a specific seed, so we reset.
            # Other wise just proceed.
            random.seed(self.random_seed)

        while not self.state == "finished":
            res = self.step()

        self.success = res   # store the result in the packet.
        return res  # we return the result of the last step before finish



class Simulator(object):
    """
    Class intended for running simulations.

    This is just a proof of concept, must be further developed.
    """
    def __init__(self, network, trace_mode = "router", restricted_topology = None, random_seed = random.random()):

        self.network = network
        self.traces = dict()
        self.trace_mode= trace_mode
        self.random_seed = random_seed
        self.count_connected = 0

        if restricted_topology is not None:
            self.topology = restricted_topology
        else:
            self.topology = network.topology

        self.original_topology = self.topology

    def run_blind(self):
        # Forward a packet for each label on LFIB entry
        random.seed(self.random_seed)
        for router_name, r in self.network.routers.items():
            self.traces[router_name] = dict()
            for in_label in r.LFIB:
                p = MPLS_packet(self.network, init_router = router_name, init_stack = [in_label], verbose = True)
                res = p.fwd()
                self.traces[router_name][in_label] = [{"trace": p, "result": res}]


    def set_failed_topo(self, failed_routers = set(), failed_links=set()):

        # Auxiliary filter functions for this case
        def filter_node(n):
            # Remove potentially failed node from the graph.
            return False if n in failed_routers else True

        def filter_edge(n1,n2):
            if ((n1,n2) in failed_links) or ((n2,n1) in failed_links):
                return False
            return True

        # Compute subgraph
        self.topology = nx.subgraph_view(self.original_topology, filter_node = filter_node, filter_edge = filter_edge)
        print(self.original_topology.edges)
        print(self.topology.edges)

    def flow_flatten(self, flows, flow_type):
        for router_name, lv1 in flows.items():
            if router_name == "__flow_type":
                continue   # special key, ignore.

            r = self.network.routers[router_name]

            if flow_type == 0:
                for local_id, tup in lv1.items():
                    in_label = local_id
                    good_sources, good_targets = tup
                    yield router_name, in_label, good_sources, good_targets

            elif flow_type == 1:
                good_sources = [router_name]
                for fec_name_subsr, lv2 in lv1.items():
                    good_targets = [lv2]
                    in_label = r.get_label_from_fec_name(fec_name_subsr)
                    yield router_name, in_label, good_sources, good_targets


            elif flow_type == 2:
                good_sources = [router_name]
                for local_id, lv2 in lv1.items():

                    dst_router  = local_id
                    good_targets = [dst_router]

                    # iterate through lv2 entries.
                    for proto, lv3 in lv2.items():
                        if proto == "LDP":
                            bw = lv3
                            fec_name_subsr = "lo_{}".format(dst_router)
                            print(fec_name_subsr)
                            in_label = r.get_label_from_fec_name(fec_name_subsr)
                            yield router_name, in_label, good_sources, good_targets

                        elif proto == "RSVPTE":
                            for tunnel_id, bw in lv3.items():
                                fec_name_subsr = "lsp_from_{}_to_{}-{}".format(router_name, dst_router, tunnel_id)
                                print(fec_name_subsr)
                                in_label = r.get_label_from_fec_name(fec_name_subsr)
                                yield router_name, in_label, good_sources, good_targets

                        else:
                            raise Exception(f"Unknown format or protocol ({proto}?).")


    def run(self, verbose = False, flows = None):
        # Forward a packet for each flow in the 'flows' list, and returnresults and stats.

        # Let's now admit different formats for the flows.
        # All flows must be structured as nested dictionaries (ideally loaded from json files).
        # the type is identified by the special key "__flow_type".
        #
        # __flow_type=0 (Default)   [src_router][incoming_label] => (good_sources, good_targets)
        # __flow_type=1             [src_router][fec_name] => good_targets
        # __flow_type=2 -           [src_router][dst_router]["LDP"] => bw
        # __flow_type=3 -           [src_router][dst_router]["RSVPTE"][tunnel_id] => bw
        #

        # TODO: Fix -- this import should not be necessary, if generate_fwd_rules is called before.
        # But for jupyter it is ...
        # from sr_client import SegmentRoutingClient

        # classify according to fec_type
        print(f"running simulation with seed {self.random_seed}" )
        random.seed(self.random_seed)

        # Load flows.
        flow_type = 0  # default
        if not flows:
            flows = self.network.build_flow_table()
            if verbose:
                pprint(flows)
        else:
            assert isinstance(flows, dict)
            if "__flow_type" in flows.keys():
                flow_type  = flows["__flow_type"]

        #initiate processing.
        for router_name, in_label, good_sources, good_targets in self.flow_flatten(flows, flow_type):
            if verbose:
                print(f"\n processing router {router_name}")

            if router_name not in self.traces:
                self.traces[router_name] = dict()

            p = MPLS_packet(self.network, init_router = router_name, init_stack = [in_label],
                            verbose = False, restricted_topology = self.topology)
            res = p.fwd()

            last_router_name = p.traceroute[-1].name

            if res and last_router_name not in good_targets:
                res = False

            if verbose:
                print(f"label: {in_label}, Initial result: {res}")
            # print(f"router: {router_name}, label: {in_label}, Initial result: {res}")

            self.traces[router_name][in_label] = [{"trace": p, "result": res}]

            # for good_target in list(good_targets):
            for good_target in list(good_targets):
                if nx.has_path(self.topology, router_name, good_target):
                    self.count_connected += 1
                    break

            if not res and verbose:
                print(" ##### DEBUG INFO ###")
                pprint(f"Router: {router_name} Lbl: {in_label}")
                print("Good Sources:")
                pprint(good_sources)
                print("Good Targets:")
                pprint(good_targets)
                pprint(last_router_name)
                pprint(f"{fec.name}/{fec.fec_type}/{fec.value}")
                pprint(self.decode_trace(p.traceroute))
                pprint(f"Result: {res}")
                print(" ####################")



    def decode_trace(self, trace):
        l = []
        for i in trace:
            if isinstance(i, Router):
                l.append(i.name)
            elif isinstance(i, tuple) and isinstance(i[0], Router):
                l.append((i[0].name, i[1]))
        return l

    def print_traces(self, store = False):
        # Print traces from the simulations run.
        output = ""
        for router_name, r in self.network.routers.items():
            if router_name not in self.traces.keys():
                continue
            t = self.traces[router_name]
            for in_label in r.LFIB:
                if in_label not in t:
                    continue
                for entry in t[in_label]:
                    p = entry["trace"]
                    res = entry["result"]
                    if self.trace_mode == "router":
                        #link_traceroute
                        s = f"{res};{p.exit_code};{router_name};{in_label};{self.decode_trace(p.traceroute)};"
                    elif self.trace_mode == "links":
                        s = f"{res};{p.exit_code};{router_name};{in_label};{self.decode_trace(p.link_traceroute)};"

                    if store:
                        output += "\n"+ s
                    else:
                        pprint(s)
        if store:
            return output


    def success_rate(self, exit_codes = False):
        # Find ratio of succesful traces.
        success = failures = 0
        codes = [0] * 6  # exit code counter list
        for t_by_router, rtraces  in self.traces.items():
            for in_label, xlist  in rtraces.items():
                for p in xlist:
                    if p["result"]:
                        success += 1
                    else:
                        failures += 1

                    if exit_codes:
                        c = p["trace"].exit_code
                        codes[c] += 1

        total = success + failures
        success_ratio = success/total
        if exit_codes:
            return (success,total, codes)

        return (success_ratio,total)
