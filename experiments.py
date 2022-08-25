#!/usr/bin/env python3
# coding: utf-8


#Load required libraries
import networkx as nx
import random
import time
import json, yaml
import math
from pprint import pprint
from itertools import chain, count
import argparse
import sys, os

from mpls_fwd_gen import *

########################
## WARNING! EXPERIMENTAL 2

class ProcRSVPTE_custom(ProcRSVPTE):
    """
    Class implementing a Resource Reservation Protocol with Traffic Engineering
    extensions (RSVP-TE) client.
    """

    def __init__(self, router, max_hops = 3):
        super().__init__(router, max_hops)
        self.n_bypass_keys = 0
        self.multi_bypass = 4
        print("Initiating magic darkness here...")
        # self.final_bypasses = dict()

    def commit_config(self):
        # Iterates over the headended_lsps requests to request the
        # corresponding entries for lsps and bypases on the routers
        # along the main LSP path.
        #
        # This function should be executed on all nodes in the
        # network before actually asking for known resources.

        network = self.router.network

        for lsp_name, lsp_data in self.headended_lsps.items():

            G = self.router.topology   #change G to add restrictions...
            spath = lsp_data['path']   #shortest path result, as list of hops..
            fec = lsp_data['FEC']      #an oFEC object with type="TE_LSP"
            protection = fec.value[2]  #protection mode
            length = len(spath)

            # create entry in local requested_lsps. value should be spath (a list object)
            self.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': (spath[0],spath[1])}

            #create entry in tailend
            tailend_pRSVP = network.routers[spath[-1]].clients["RSVP-TE"]
            tailend_pRSVP.requested_lsps[lsp_name] = {'FEC':fec}

            # create entry in requested_lsps of downstream nodes.
            # Value should be a 3-tuple or a 2-tuple only.
            # iterate on upstream direction.
            for i in range(length-2, 0, -1):
                # compute protections only for intermediate nodes.
                PLR = spath[i]       #PLR: Point of Local Repair
                pRSVP = network.routers[PLR].clients["RSVP-TE"]

                if i == length-2 or protection == "facility-link":
                    #penultimate node in LSP, circumvent the last edge
                    MP = spath[i+1]  # MP: Merge Point
                    protected_tuple = (PLR, MP) # the last edge

                elif i < length-2 and protection == "facility-node":
                    #circumvent next node
                    facility = spath[i+1] #original next node
                    MP = spath[i+2]
                    protected_tuple = (PLR,facility,MP)


                # create a protection for all LSPs that traverse the protected_tuple
                if protection in ["facility-link", "facility-node"]:
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': protected_tuple}

                    # create entry in requested_bypasses (if it doent's exist) of
                    # downstream nodes.
                    # Key should be the corresponding protected 3-tuple or 2-tuple.
                    # bypass_name = "bypass_{}_{}_{}".format(*triplet)
                    if protected_tuple not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[protected_tuple] = dict()

                # elif protection == "one-to-one":
                #     # use tailend as MP
                #     pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': (PLR,spath[-1])}
                #
                #     # create entry in requested_bypasses (if it doent's exist) of
                #     # downstream nodes.
                #     # Key should be the corresponding protected 3-tuple or 2-tuple.
                #     # bypass_name = "bypass_{}_{}_{}".format(*triplet)
                #     if protected_tuple_2 not in pRSVP.requested_bypasses.keys():
                #         pRSVP.requested_bypasses[protected_tuple_2] = dict()
                #     if protected_tuple_3 and protected_tuple_3 not in pRSVP.requested_bypasses.keys():
                #         pRSVP.requested_bypasses[protected_tuple_3] = dict()
        #no change

    def compute_bypasses(self):
        # Compute all requested bypasses LSPs from the formation of LSPs during commit_config().
        #
        # This function must only be executed only execute after all RSVP-TE nodes
        # have executed and completed commit_config().
        # This function must be executed on all nodes in the network before actually requesting for known
        # resources.
        print(f"\n >>> Start compute_bypass on {self.router.name}")
        if not self.requested_bypasses:
            # No ProcRSVPTE requested bypasses for this router.
            return None

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)

                if bypass_path and len(bypass_path) >= max_hops + 2:
                    #bypass too large
                    bypass_path = None

            except nx.NetworkXNoPath as err_nopath:
                #No path!
                bypass_path = None

            finally:
                return bypass_path

        network = self.router.network
        # Iterate through the tuples to protect.
        bypass_path = None
        bypass_paths = dict()

        self.n_bypass_keys = len(self.requested_bypasses.keys()) # so we can check if something changed
        print(f"Number of bypass keys on {self.router.name}: {self.n_bypass_keys}")
        print("requested_bypasses at start:")
        pprint(self.requested_bypasses)
        # for curr_tuple, bypass_pre_data in self.requested_bypasses.items():
        list_of_bypasses = list(self.requested_bypasses.keys()).copy()

        for curr_tuple in list_of_bypasses:
            bypass_pre_data = self.requested_bypasses[curr_tuple]
            prio = 100
            if bypass_pre_data:
                #Already computed, try to find protections to this one.
            #     continue
                tuple_list = []
                for prio, bypass_data in bypass_pre_data.items(): #we need to iterate here!
                    path = bypass_data['bypass_path']
                    i = path.index(self.router.name)
                    if i < len(path)-2:
                        pprint(f"debug: bpath:{path}, segm: {path[i:i+3]}")
                        candidate = tuple(path[i:i+3])
                    elif i < len(path)-1:
                        pprint(f"debug: bpath:{path}, segm: {path[i:i+2]}")
                        candidate = tuple(path[i:i+2])
                    else:
                        continue

                    # if candidate in self.requested_bypasses:
                    #     print("Warning! Candidate already in self.requested bypasses, so we have already been here.")
                    #     print(f"Candidate: {candidate}")
                    #     print(f"path: {path}")
                    #     for tdict in self.requested_bypasses[candidate].values():
                    #         pprint(tdict)
                    #         if path == tdict['bypass_path']:
                    #             print("Already discovered and computed")
                    #             continue
                    if candidate in self.requested_bypasses:
                        print("Already discovered and computed")  # What somebody else
                        continue

                    tuple_list.append(candidate)
            else:
                tuple_list = [curr_tuple]
            tuple_list = list(set(tuple_list))
            print("TUPLE LIST")
            pprint(tuple_list)
            for protected_tuple in tuple_list:
                pprint(f"We want to protect tuple {protected_tuple}")

                bypass_priority = prio + 1

                G = self.router.topology   #change G to add constraints
                PLR = protected_tuple[0]   # Point of Local Repair
                MP = protected_tuple[-1]   # Merge Point

                if len(protected_tuple) == 3:
                #Node failure / NextNextHop  protection
                    facility = protected_tuple[1] # potentially failed node

                    L = [facility]  # List of already tried first nodes.

                    # while True:
                    for _ in range(self.multi_bypass):
                        # Auxiliary filter functions for this case
                        def filter_node(n):
                            # Remove potentially failed node from the graph.
                            return False if n in L else True
                            #return True
                        def filter_edge(n1,n2):
                            return True

                        # Compute subgraph
                        view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
                        # Get bypass path
                        bypass_path = _get_path(view, PLR, MP, weight='weight', max_hops = self.frr_max_hops)
                        pprint(f"candidate bypass_path computed at {self.router.name} to protect {protected_tuple}:{bypass_path}")
                        pprint(f"facility: {L}")
                        if bypass_path:
                            print(f"good bypass! ({bypass_path})")
                            bypass_paths[bypass_priority] = bypass_path
                            L.append(bypass_path[1])
                            bypass_priority += 1
                        else:
                            print(f"No valid bypass, evaluated False! ({bypass_path})")
                            break

                    if not bypass_paths:
                        # if we can't protect the next node, attempt to protect the link.
                        MP = facility  # We will merge on the next node downstream instead of skipping it.


                if len(protected_tuple) == 2 or not bypass_paths:
                    #Link failure / NextHop  protection
                    print("Let's look for link protections...")

                    L = [(PLR, MP)]  # List of already tried first nodes.

                    if (PLR, MP) not in G.edges() and (MP, PLR) not in G.edges():
                        # TO protect a link, it has to exist!
                        pprint(f"There is no edge ({PLR},{MP}), can't protect.")
                        continue

                    # while True:
                    for _ in range(self.multi_bypass):
                        # Auxiliary filter functions for this case
                        def filter_node(n):
                            return True
                        def filter_edge(n1,n2):
                            # Remove potentially failed link from the graph.
                            # return False if (n1,n2) == (PLR, MP) or (n2,n1) == (PLR, MP) else True
                            return False if (n1,n2) in L or (n2,n1) in L else True

                        # Compute subgraph
                        view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
                        # Get bypass path
                        bypass_path = _get_path(view, PLR, MP, weight='weight', max_hops = self.frr_max_hops)
                        print("%%"*20)
                        print(f"candidate bypass_path for link protection computed at {self.router.name} to protect {protected_tuple}:{bypass_path}")
                        print(f"facility: {L}")
                        # print(f"PLR: {PLR}, MP:{MP}")
                        print("%%"*20)
                        if bypass_path:
                            print(f"good bypass! ({bypass_path})")
                            #bypass_paths[bypass_priority] = bypass_path  //this is again at the end...
                            # L.append(bypass_path[1])
                            if len(bypass_path) > 2:
                                print("Filtering out this part of the path for next step: {}.".format(tuple(bypass_path[0:2])))
                                # pprint(bypass_path)
                                #pprint(tuple(bypass_path[0:2]))
                                L.append(tuple(bypass_path[0:2]))   # this will break when we move to multigraph
                            else:
                                print("found very very short path around?:")
                                pprint(bypass_path)
                                break
                            bypass_priority += 1
                        else:
                            #continue
                            print(f"No valid bypass, evaluated False! ({bypass_path})")
                            break #?
                        bypass_paths[bypass_priority] = bypass_path


                if not bypass_paths:
                    # We could not find a valid protection, move on to next request.
                    # Nevertheless an empty entry would remain in requested_bypasses.
                    continue

                print("so far this is what we found these prio => bypass:")
                pprint(bypass_paths)

                # Pass through bypasses..
                # WE FOUND AT LEAST ONE USABLE BYPASS PATH! Create a FEC object for it.
                for bypass_priority, bypass_path in bypass_paths.items():
                    bypass_name = "bypass_{}".format("_".join([str(item) for item in protected_tuple ]))
                    bypass = oFEC("bypass", bypass_name, protected_tuple)

                    # Iterate through the bypass path
                    length = len(bypass_path)
                    for i in range(0, length):
                        # Note: we will compute protections only for intermediate nodes.
                        #Get current router in path, and its RSVP process.
                        curr_router = bypass_path[i]
                        pRSVP = network.routers[curr_router].clients["RSVP-TE"]
                        # pprint("//"*20)
                        # pprint(bypass_path)
                        # print(i)
                        # pprint("//"*20)
                        # Get its nexthop, it is needed to set up the path.
                        if i+1 < length:
                            next_hop = bypass_path[i+1]
                        else:
                            #for the MP, we will need a local label.
                            next_hop = None

                        # Insert a bypass request on the router.
                        # not pRSVP.requested_bypasses[protected_tuple]
                        if protected_tuple not in pRSVP.requested_bypasses.keys():
                            pRSVP.requested_bypasses[protected_tuple] = dict()  #initialize if necesary

                        if bypass_priority not in pRSVP.requested_bypasses[protected_tuple].keys():
                            print(f"Inserting requested bypass on {curr_router}:")
                            pprint([protected_tuple,bypass_priority,{'FEC': bypass,'next_hop': next_hop, 'bypass_path': bypass_path }])
                            pRSVP.requested_bypasses[protected_tuple][bypass_priority] = {'FEC': bypass,
                                                                                          'next_hop': next_hop,
                                                                                          'bypass_path': bypass_path }
                        else:
                            print(f"UNEXPECTED: bypass priority already computed {bypass_priority} ")


        print(f" >>> Finished compute_bypass on {self.router.name}. Result:")
        pprint(self.requested_bypasses)

    def known_resources(self):
        # Returns a generator to iterate over all tunnels and bypasses (RSVP-TE resources).

        def my_sort():
            # we sort entries, starting with those bypasses starting on this router and protecting a node failure,
            # second those starting in this router and protecting link failure, then starting elsewhere protecting nodes
            # and finally starting elsewhere protecting links.
            d = { 1:[],  2:[], 3:[], 4:[] }
            for bypass_tuple in self.requested_bypasses.keys():
                if self.router.name == bypass_tuple[0]:
                    if len(bypass_tuple) == 3:
                        d[1].append(bypass_tuple)
                    else:
                        d[2].append(bypass_tuple)
                else:
                    if len(bypass_tuple) == 3:
                        d[3].append(bypass_tuple)
                    else:
                        d[4].append(bypass_tuple)

            # return d[1] + d[2] + d[3] + d[4]
            return d[1] + d[2] + d[1] + d[2] + d[3] + d[4]


        if not self.requested_lsps and not self.requested_bypasses:
            # ProcRSVPTE in not initialized.
            return None

        #First return the bypases
        #for bypass_tuple, bypass_pre_data in self.requested_bypasses.items():
        for bypass_tuple in my_sort():
            for prio, bypass_data in  self.requested_bypasses[bypass_tuple].items():
                if 'FEC' not in bypass_data.keys():
                    # The requested bypass was impossible/unsolvable.
                    continue
                fec = bypass_data['FEC']
                yield fec

        # return the tunnel LSPs
        for lsp_name, lsp_data in self.requested_lsps.items():
            fec = lsp_data['FEC']
            yield fec

    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network
        print(f"Start LFIB_compute_entry (router {router.name})")
        # Starting with bypasses
        try:
            if fec not in used_ifaces.keys():
                used_ifaces[fec] = []
        except:
            used_ifaces = dict()  #initialize
            used_ifaces[fec] = []


        if  fec.fec_type == "bypass":
            if not fec.value:
                return   # this case should never happen.

            # recover the intended protected tuple.
            tuple_data = fec.value

            print(f"tuple_data: {tuple_data}")
            if router.name not in tuple_data:
                #intermediate router, process the bypass like a regular LSP

                # should we also build protections here? Yes! we definitely should.

                for bypass_prio, data in self.requested_bypasses[tuple_data].items():
                    pprint("###############")
                    pprint(f"bypass {bypass_prio}: {data}")
                    pprint("###############")
                    bypass_path = data['bypass_path']

                    alternatives = []
                    i = bypass_path.index(self.router.name)
                    if i < len(bypass_path)-2:
                        pprint(f"debug finding alternatives: bpath:{bypass_path}, segm: {bypass_path[i:i+3]}")
                        candidate = tuple(bypass_path[i:i+3])
                        if candidate != tuple_data:
                            alternatives.append(candidate)

                    if i < len(bypass_path)-1:
                        pprint(f"debug finding alternatives: bpath:{bypass_path}, segm: {bypass_path[i:i+2]}")
                        candidate = tuple(bypass_path[i:i+2])
                        if candidate != tuple_data:
                            alternatives.append(candidate)
                    print("alternatives")
                    pprint(alternatives)

                    next_hop_name = data['next_hop']
                    if not next_hop_name:
                        continue
                    next_hop =  network.routers[next_hop_name]

                    local_label = self.get_local_label(fec)
                    remote_label = self.get_remote_label(next_hop_name, fec)
                    cost = self.bypass_cost + bypass_prio

                    # A swap op will do in the normal case.
                    if remote_label == self.IMPLICIT_NULL:
                        re_ops = [{"pop": ""}]
                        # routing_entry = { "out": re_out, "ops": [{"pop": ""}], "weight": cost  }
                    else:
                        re_ops = [{"swap": remote_label}]
                        # routing_entry = { "out": re_out, "ops": [{"swap": remote_label}], "weight": cost  }

                    routing_entry = { "out": next_hop_name, "ops": re_ops, "weight": cost  }
                    print("this: ")
                    pprint((local_label, routing_entry))
                    used_ifaces[fec].append(next_hop_name)
                    yield (local_label, routing_entry)

                    # now explore protections:
                    for a in alternatives:

                        if a not in self.requested_bypasses.keys():
                            continue

                        for  bypass_prio_alt, data_alt in self.requested_bypasses[a].items():
                            print(f"Exploring alternative: {a} (got data:)")
                            print("data_alt:")
                            pprint(data_alt)
                            pprint(data_alt['FEC'].name)
                            pprint(data_alt['FEC'].value)

                            cost_alt = cost + bypass_prio_alt
                            try:
                                local_label_alt = self.get_local_label(data_alt['FEC'])
                                print(f"alternative local label: {local_label_alt}")
                                for d_alt in self.router.LFIB[local_label_alt]:
                                    pprint(d_alt)
                                    nh_alt_name, ops_alt, w_alt  = d_alt['out'], d_alt['ops'], d_alt['weight']

                                    # if "swap" in ops_alt[0].keys():
                                    #     ops_alt[0] = {"push": ops_alt[0]["swap"]}
                                    # elif "pop" in ops_alt[0].keys():
                                    #     ops_alt = ops_alt[1:]

                                    pprint(used_ifaces[fec])
                                    if nh_alt_name in used_ifaces[fec]:
                                        continue
                                    else:
                                        used_ifaces[fec].append(nh_alt_name)
                                    # re_out_alt = nh_alt.name

                                    #if nh_alt_name in tuple_data[:-1]
                                    if nh_alt_name in bypass_path[:i]:
                                        w_alt = 1000*w_alt

                                    print("="*10)
                                    pprint(f"component_1 bpath:{bypass_path}, ops: {re_ops}")
                                    pprint(f"component_2 bpath:{bypass_path}, ops: {ops_alt}")
                                    print("="*10)
                                    # routing_entry_alt = { "out": nh_alt_name, "ops": re_ops+mod_ops_alt, "weight": 100*cost_alt + w_alt  }
                                    routing_entry_alt = { "out": nh_alt_name, "ops": ops_alt, "weight": 100*cost_alt + w_alt  }
                                    print("and then this: ")
                                    pprint((local_label, routing_entry_alt))
                                    print()
                                    yield (local_label, routing_entry_alt)
                            except KeyError as e:
                                print("These are not the tuples you are looking for either...")
                                pprint(e)
                                continue


            elif router.name == tuple_data[-1] or (len(tuple_data) == 3 and router.name == tuple_data[-2]):
                #this is the MP for the requested bypass. No next_hop, just pop.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

            # elif len(tuple_data) == 3 and router.name == tuple_data[-2]:
            #     #this is the MP for the requested bypass. No next_hop, just pop.
            #     #in this case we wanted 3-tuple protection but only got link protection.
            #     local_label = self.get_local_label(fec)
            #     routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
            #     yield (local_label, routing_entry)

#### WORK IN PROGRESS
            elif router.name == tuple_data[0]:
                # This is the PLR!
                #### PAY ATTENTION HERE!!
                print("#### WORK IN PROGRESS!")
                print("This is an opportunity to explore protection of protections by adding protections to a bypass")

                #intermediate router, process the bypass like a regular LSP
                for bypass_prio, data in self.requested_bypasses[tuple_data].items():
                    pprint("###############")
                    pprint(f"{bypass_prio}: {data}")
                    pprint("###############")
                    bypass_path = data['bypass_path']
                    next_hop_name = data['next_hop']
                    bypass_fec = data['FEC']

                    if len(bypass_path) > 2:
                        MP_name = bypass_path[2]
                    else:
                        MP_name = bypass_path[1]

                    local_label = self.get_local_label(fec)
                    cost = self.bypass_cost + bypass_prio

                    merge_label = self.get_remote_label(MP_name, fec)
                    # And the bypass label for the FRR next hop
                    bypass_label = self.get_remote_label(next_hop_name, bypass_fec)

                    if merge_label == self.IMPLICIT_NULL and bypass_label == self.IMPLICIT_NULL:
                        r_info = [{"pop": ""}]
                    elif merge_label == self.IMPLICIT_NULL:
                        r_info = [{"swap": bypass_label} ] #we don't push an IMPNULL
                    else:
                        r_info = [{"swap": merge_label}, {"push": bypass_label} ]

                    backup_entry = { "out": next_hop_name, "ops": r_info, "weight": cost  }
                    pprint((local_label, backup_entry))
                    print("#### END WORK IN PROGRESS!")
                    yield (local_label, backup_entry)

#### END WORK IN PROGRESS


        if  fec.fec_type == "TE_LSP":
            if 'tuple' not in self.requested_lsps[fec.name].keys():  #There should be a tidier way
                #Then I am the tailend for this LSP.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

            else:
                # Regular case
                # Recover the protected tuple (subpath).
                tuple_data = self.requested_lsps[fec.name]['tuple']
                if len(tuple_data) == 3:     # Node protection
                    _, next_hop_name, MP_name = tuple_data
                elif len(tuple_data) == 2:   # Link protection
                    _, next_hop_name = tuple_data
                    MP_name = next_hop_name

                # Gather next_hop forwarding information
                next_hop =  network.routers[next_hop_name]
                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec)
                #cost = self.bypass_cost
                headend_name = fec.value[0]
                headend_proc = network.routers[headend_name].get_client(type(self))
                cost = headend_proc.headended_lsps[fec.name]["cost"]

                # A swap is enough to build the main tunnel
                if remote_label == self.IMPLICIT_NULL:
                    main_entry = { "out": next_hop.name, "ops": [{"pop": ""}], "weight": cost  }
                else:
                    main_entry = { "out": next_hop.name, "ops": [{"swap": remote_label}], "weight": cost  }

                yield (local_label, main_entry)

                # If there is no bypass requested for the tuple/subpath, we are done.
                if tuple_data not in self.requested_bypasses.keys():
                    return

                # FRR -- Fast Re Route requirements.
                # If there is a protection request
                # Check feasability.
                bypass_pre_data = self.requested_bypasses[tuple_data]


                for bypass_prio, bypass_data in bypass_pre_data.items():

                    if 'FEC' not in bypass_data.keys():
                        # The requested bypass is impossible/unsolvable, we are done.
                        return

                    # Feasability confirmed, gather forwarding information
                    bypass_fec = bypass_data['FEC']
                    bypass_next_hop_name = bypass_data['next_hop']
                    bypass_next_hop = network.routers[bypass_next_hop_name]

                    # We need to know the label that the MP expects
                    merge_label = self.get_remote_label(MP_name, fec)
                    # And the bypass label for the FRR next hop
                    bypass_label = self.get_remote_label(bypass_next_hop_name, bypass_fec)
                    # Fix the entry priority
                    cost = self.bypass_cost

                    # This is a backup entry. It must push down a label on the stack to
                    # encapsulate forwarding through the bypass path.

                    # backup_entry = { "out": bypass_next_hop.name, "ops": [{"swap": merge_label}, {"push": bypass_label} ], "weight": cost  }

                    backup_entry = { "out": bypass_next_hop.name, "weight": cost + bypass_prio  }
                    if merge_label == self.IMPLICIT_NULL and bypass_label == self.IMPLICIT_NULL:
                        r_info = [{"pop": ""}]
                    elif merge_label == self.IMPLICIT_NULL:
                        r_info = [{"swap": bypass_label} ] #we don't push an IMPNULL
                    else:
                        r_info = [{"swap": merge_label}, {"push": bypass_label} ]

                    backup_entry["ops"] = r_info

                    yield (local_label, backup_entry)

## WARNING! EXPERIMENTAL 2
########################


########################
## WARNING! EXPERIMENTAL 3

class ProcRSVPTE_aalwi(ProcRSVPTE):

    def __init__(self, router, max_hops = 3, leader = None):
        super().__init__(router, max_hops)
        self.leader = leader
        # self.final_bypasses = dict()

    def compute_bypasses(self):
        # def orchestrate_protections(self):
        #execute only once!
        if self.router.name != self.leader:
            # Only the leader must execute this.
            return None

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)

                if bypass_path and len(bypass_path) >= max_hops + 2:
                    #bypass too large
                    bypass_path = None

            except nx.NetworkXNoPath as err_nopath:
                #No path!
                bypass_path = None

            finally:
                return bypass_path

        network = self.router.network
        G = network.topology

        for e in G.edges():
            # Auxiliary filter functions for this case
            def filter_node(n):
                # Remove potentially failed node from the graph.
                # return False if n in L else True
                return True

            def filter_edge(n1,n2):
                # Remove potentially failed link from the graph.
                # return False if (n1,n2) == (PLR, MP) or (n2,n1) == (PLR, MP) else True
                return False if ((n1,n2) == e or (n2,n1) == e) else True

            # Compute subgraph
            view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
            # Get bypass path
            bypass_path = _get_path(view, e[0], e[1], weight='weight', max_hops = self.frr_max_hops)

            if not bypass_path:
                print(f"No valid bypass for edge {e}, evaluated False!")
                continue

            pprint(f"candidate bypass_path computed at {self.router.name} to protect {e}")
            bypass_name = "bypass_{}".format("_".join([str(item) for item in e ]))
            bypass = oFEC("bypass", bypass_name, e)

            # Iterate through the bypass path
            length = len(bypass_path)
            for i in range(0, length):
                curr_router = bypass_path[i]
                pRSVP = network.routers[curr_router].clients["RSVP-TE"]
                if i+1 < length:
                    next_hop = bypass_path[i+1]
                else:
                    #for the MP, we will need a local label.
                    next_hop = None

                # Insert a bypass request on the router.
                # not pRSVP.requested_bypasses[protected_tuple]
                if e not in pRSVP.requested_bypasses.keys():
                    pRSVP.requested_bypasses[e] = dict()  #initialize if necesary

                pRSVP.requested_bypasses[e] = {'FEC': bypass,
                                               'next_hop': next_hop,
                                               'bypass_path': bypass_path }

    def known_resources(self):
        # Returns a generator to iterate over all tunnels and bypasses (RSVP-TE resources).

        if not self.requested_lsps and not self.requested_bypasses:
            # ProcRSVPTE in not initialized.
            return None

        pprint(self.requested_bypasses)
        for bypass_tuple, bypass_data in self.requested_bypasses.items():
            # for bypass_data in  self.requested_bypasses[bypass_tuple].items():
            if 'FEC' not in bypass_data.keys():
                # The requested bypass was impossible/unsolvable.
                continue
            fec = bypass_data['FEC']
            yield fec

        for lsp_name, lsp_data in self.requested_lsps.items():
            fec = lsp_data['FEC']
            yield fec

    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network
        print(f"Start LFIB_compute_entry (router {router.name})")
        # Starting with bypasses
        try:
            if fec not in used_ifaces.keys():
                used_ifaces[fec] = []   #for each FEC an interface can appear in at most a single entry.
        except:
            used_ifaces = dict()  #initialize
            used_ifaces[fec] = []


        if  fec.fec_type == "bypass":
            if not fec.value:
                return   # this case should never happen.

            # recover the intended protected tuple.
            tuple_data = fec.value

            print(f"tuple_data: {tuple_data}")
            if (router.name not in tuple_data) or router.name == tuple_data[0]:
                #intermediate router or PLR,

                data = self.requested_bypasses[tuple_data]
                bypass_path = data['bypass_path']
                next_hop_name = data['next_hop']
                if next_hop_name in used_ifaces[fec]:
                    # inteafec already used for this FEC, we ca't allocate it again.
                    return

                next_hop =  network.routers[next_hop_name]

                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec)
                cost = self.bypass_cost

                # A swap op will do in the normal case.
                if router.name == tuple_data[0]:
                    re_ops = [{"push": remote_label} ]
                elif remote_label == self.IMPLICIT_NULL:
                    re_ops = [{"pop": ""}]
                else:
                    re_ops = [{"swap": remote_label}]

                routing_entry = { "out": next_hop_name, "ops": re_ops, "weight": cost  }
                print("this: ")
                pprint((local_label, routing_entry))
                used_ifaces[fec].append(next_hop_name)
                yield (local_label, routing_entry)

            elif router.name == tuple_data[-1]:
                #this is the MP for the requested bypass. No next_hop, just pop.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)


        if  fec.fec_type == "TE_LSP":
            if 'tuple' not in self.requested_lsps[fec.name].keys():
                #Then I am the tailend for this LSP.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

            else:
                # Regular case
                # Recover the protected tuple (subpath).
                tuple_data = self.requested_lsps[fec.name]['tuple']
                if len(tuple_data) == 3:     # Node protection
                    _, next_hop_name, MP_name = tuple_data
                elif len(tuple_data) == 2:   # Link protection
                    _, next_hop_name = tuple_data
                    MP_name = next_hop_name

                # Gather next_hop forwarding information
                next_hop =  network.routers[next_hop_name]
                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec)
                #cost = self.bypass_cost
                headend_name = fec.value[0]
                headend_proc = network.routers[headend_name].get_client(type(self))
                cost = headend_proc.headended_lsps[fec.name]["cost"]

                # A swap is enough to build the main tunnel
                if remote_label == self.IMPLICIT_NULL:
                    main_entry = { "out": next_hop.name, "ops": [{"pop": ""}], "weight": cost  }
                else:
                    main_entry = { "out": next_hop.name, "ops": [{"swap": remote_label}], "weight": cost  }

                yield (local_label, main_entry)

                # If there is no bypass requested for the tuple/subpath, we are done.
                if tuple_data not in self.requested_bypasses.keys():
                    return


    def LFIB_refine(self, label):
        # Some process might require a refinement of the LFIB.
        r = self.router
        entries = r.get_routing_entry(label)
        Q = sorted(entries, key = lambda x:x['priority'])

        additional_entries = []
        for q in Q:

            e_1 = (r.name, q["out"])
            e_2 = (q["out"], r.name)
            if e_1 in self.requested_bypasses:
                if "FEC" not in self.requested_bypasses[e_1]:
                    #Couldn't find a protection or alternative path.
                    continue
                fec = self.requested_bypasses[e_1]["FEC"]
            elif e_2 in self.requested_bypasses:
                if "FEC" not in self.requested_bypasses[e_2]:
                    #Couldn't find a protection or alternative path.
                    continue
                fec = self.requested_bypasses[e_2]["FEC"]
            else:
                continue

            protection_label = r.get_label(fec)
            new_ops = q["ops"] + [{"push": protection_label}]
            routing_entry = { "out": r.LOCAL_LOOKUP, "ops":new_ops , "priority": q["priority"] + 1  }
            additional_entries.append(routing_entry)

        return additional_entries


## WARNING! EXPERIMENTAL 3
########################



########################
## WARNING! EXPERIMENTAL 4
# This is the orignal ProcRMPLS class used until Jun2021, with partial loop avoidance.
class ProcRMPLS_jun2021(ProcRSVPTE):

    protocol = "RMPLS"

    def __init__(self, router, max_hops = 999999999, leader = None):
        super().__init__(router, max_hops)
        self.leader = leader
        self.build_order = 10000
        print(f"Creating RMPLS process in router {router} (leader: {leader})")

    def compute_bypasses(self):
        # def orchestrate_protections(self):
        #execute only once!
        if self.router.name != self.leader:
            # Only the leader must execute this.
            return None

        print(f"I am the leader {self.leader} and will compute RMPLS bypasses!")

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)

                if bypass_path and len(bypass_path) >= max_hops + 2:
                    #bypass too large
                    bypass_path = None

            except nx.NetworkXNoPath as err_nopath:
                #No path!
                bypass_path = None

            finally:
                return bypass_path

        network = self.router.network
        G = network.topology

        for e in G.edges():
            # Auxiliary filter functions for this case
            e_rev = (e[1],e[0])
            print(f"Let's protect {e} and {e_rev}")
            def filter_node(n):
                # Remove potentially failed node from the graph.
                # return False if n in L else True
                return True

            def filter_edge(n1,n2):
                # Remove potentially failed link from the graph.
                # return False if (n1,n2) == (PLR, MP) or (n2,n1) == (PLR, MP) else True
                return False if ((n1,n2) == e or (n2,n1) == e) else True


            def plug(e, bypass_path):
                if not bypass_path:
                    print(f"No valid rmpls bypass for edge {e}, evaluated False!")
                    return

                pprint(f"candidate rmpls bypass_path computed at {self.router.name} to protect {e}")
                bypass_name = "bypass_rmpls_{}".format("_".join([str(item) for item in e ]))
                bypass = oFEC("bypass_rmpls", bypass_name, e)

                # Iterate through the bypass path
                length = len(bypass_path)
                for i in range(0, length):
                    curr_router = bypass_path[i]
                    pRSVP = network.routers[curr_router].clients["RMPLS"]
                    if i+1 < length:
                        next_hop = bypass_path[i+1]
                    else:
                        #for the MP, we will need a local label.
                        next_hop = None

                    # Insert a bypass request on the router.
                    # not pRSVP.requested_bypasses[protected_tuple]
                    if e not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[e] = dict()  #initialize if necesary

                    pRSVP.requested_bypasses[e] = {'FEC': bypass,
                                                   'next_hop': next_hop,
                                                   'bypass_path': bypass_path }


            # Compute subgraph
            view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
            # Get bypass path
            bypass_path = _get_path(view, e[0], e[1], weight='weight', max_hops = self.frr_max_hops)
            bypass_path_rev = _get_path(view, e[1], e[0], weight='weight', max_hops = self.frr_max_hops)

            plug(e, bypass_path)
            plug(e_rev, bypass_path_rev)


    def known_resources(self):
        # Returns a generator to iterate over all tunnels and bypasses (RSVP-TE resources).

        if not self.requested_bypasses:
            # RMPLS in not initialized.
            return None

        pprint(self.requested_bypasses)
        for bypass_tuple, bypass_data in self.requested_bypasses.items():
            # for bypass_data in  self.requested_bypasses[bypass_tuple].items():
            if 'FEC' not in bypass_data.keys():
                # The requested bypass was impossible/unsolvable.
                continue
            fec = bypass_data['FEC']
            yield fec

    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network
        print(f"Start RMPLS first step LFIB_compute_entry (router {router.name})")
        # Starting with bypasses
        try:
            if fec not in used_ifaces.keys():
                used_ifaces[fec] = []   #for each FEC an interface can appear in at most a single entry.
        except:
            used_ifaces = dict()  #initialize
            used_ifaces[fec] = []


        if  fec.fec_type == "bypass_rmpls":
            if not fec.value:
                return   # this case should never happen.

            # recover the intended protected tuple.
            tuple_data = fec.value

            print(f"tuple_data: {tuple_data}")
            if (router.name not in tuple_data) or router.name == tuple_data[0]:
                #intermediate router or PLR,

                data = self.requested_bypasses[tuple_data]
                bypass_path = data['bypass_path']
                next_hop_name = data['next_hop']
                if next_hop_name in used_ifaces[fec]:
                    # interfacefec already used for this FEC, we can't allocate it again.
                    return

                next_hop =  network.routers[next_hop_name]

                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec)
                cost = self.bypass_cost

                # A swap op will do in the normal case.
                if router.name == tuple_data[0]:
                    # re_ops = [{"push": remote_label} ]
                    re_ops = [{"swap": remote_label} ]
                elif remote_label == self.IMPLICIT_NULL:
                    re_ops = [{"pop": ""}]
                else:
                    re_ops = [{"swap": remote_label}]

                routing_entry = { "out": next_hop_name, "ops": re_ops, "weight": cost  }
                print("this: ")
                pprint((local_label, routing_entry))
                used_ifaces[fec].append(next_hop_name)
                yield (local_label, routing_entry)

            elif router.name == tuple_data[-1]:
                #this is the MP for the requested bypass. No next_hop, just pop.
                local_label = self.get_local_label(fec  )
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)


    def LFIB_refine(self, label):
        # Some process might require a refinement of the LFIB.
        r = self.router
        entries = r.get_routing_entry(label)

        fec_0 = r.get_FEC_from_label(label)
        protected_edge = fec_0.value


        Q = sorted(entries, key = lambda x:x['priority'])
        # print(f"printing sorted entries on {r.name} ")
        # pprint(Q)
        max_prio = max([v['priority'] for v in Q]) + 1
        # print(f"MAX PRIO: {max_prio}")
        additional_entries = []
        for q in Q:

            e_1 = (r.name, q["out"])
            e_2 = (q["out"], r.name)
            if e_1 in self.requested_bypasses:
                if "FEC" not in self.requested_bypasses[e_1]:
                    #Couldn't find a protection or alternative path.
                    continue
                fec = self.requested_bypasses[e_1]["FEC"]
                bypass_path = self.requested_bypasses[e_1]["bypass_path"]
            elif e_2 in self.requested_bypasses:
                if "FEC" not in self.requested_bypasses[e_2]:
                    #Couldn't find a protection or alternative path.
                    continue
                fec = self.requested_bypasses[e_2]["FEC"]
                bypass_path = self.requested_bypasses[e_2]["bypass_path"]
            else:
                continue

            # protected_edge = fec.value

            protection_label = r.get_label(fec)
            new_ops = q["ops"] + [{"push": protection_label}]
            # routing_entry = { "out": r.LOCAL_LOOKUP, "ops":new_ops , "priority": q["priority"] + 1  }
            edges_on_bypass_path = list(zip( bypass_path[:-1] , bypass_path[1:] ))
            if protected_edge not in edges_on_bypass_path:   # New Loop protection.
                routing_entry = { "out": r.LOCAL_LOOKUP, "ops":new_ops , "priority": q["priority"] + max_prio  }
                print(f"MPLS: add entry {routing_entry}")
                additional_entries.append(routing_entry)

        return additional_entries


## WARNING! EXPERIMENTAL 4
########################




# The following code is a function in the Simulator class:

    def find_paths(self):
        # FIXME: This is blind, we must have a variation of this method for valid routers only...
        for router_name, r in self.network.routers.items():
            self.traces[router_name] = dict()
            for in_label in r.LFIB:
                # all path finding start with a single packet

                first_p = MPLS_packet(self.network, init_router = router_name,
                                    init_stack = [in_label], mode = "pathfinder", verbose = True)
                first_p.initialize()
                p_list=[first_p]
                p_index = 0

                safe_cnt = 0
                while any(map(lambda x:x.state == "live" , p_list)):
                    safe_cnt +=1
                    if safe_cnt > 100:
                        raise Exception("Safe counter stopped the loop that was running wild!")

                    p=p_list[p_index]
                    while p.state == "live":
                        res, p_list_new = p.step()
                        p_list += p_list_new

                    p_index += 1

                self.traces[router_name][in_label] = [{"paths_discovered": p_list, "result": res}]
