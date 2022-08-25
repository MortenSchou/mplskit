# coding: utf-8

"""
ReSerVation Protocol w/ Traffic Engineering extensions (RSVP TE)
RFC 3209 implementation for MPLS Kit - v0.1

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

"""

from mpls_fwd_gen import MPLS_Client, oFEC, Label_Manager, Network, Router
import networkx as nx
from pprint import pprint


class ProcRSVPTE(MPLS_Client):
    """
    Class implementing a Resource Reservation Protocol with Traffic Engineering
    extensions (RSVP-TE) client.

    This protocol is used for negotiation and setup of tunnels in the MPLS network.
    The tunnels could then be used as virtual interfaces for traffic forwarding, or as
    building blocks for other applications (e.g. L3VPNs)

    Tunnels provide traffic steering capabilities, as the netadmin can setup
    a tunnel with restrictions on which links can be used, what routers should the
    path go through or avoid, and even account for previously allocates bandwidth.
    Using the IGP's  Traffic Engineering database it can compute a constrained
    shortest path (CSPF) from the headend router (responsible for the tunnel itself)
    towards the tailend router.

    NOTE: currently only shortest path computation is supported!

    Manages the tunnels as resources/FECs.

    Provides FRR functionality by default according to RFC4090 with many-to-one (facility)
    option. This means that every router on the network will try to protect all TE tunnels
    against failure of the next downstream node, and if that is unfeasible, against failure
    of the downstream link. The operatoin implies pushing an new label to the stack.

    The fec_types it handles are "TE_LSP" and "bypass" (for FRR bypass lsps).

    Requires access to the router's routing table, nexthop information, the
    linkstate database to generate routing entries. Labels, both local and remotes, are
    allocated only on demand.

    This class has the following structures:

    headended_lsps:
        List of TE tunnels that are requested to start on this node.
        This is esentially configuration for the setup of new tunnels.
        Entry format:
            tunnel name => tunnel FEC , path as list of nodes.

    requested_lsps:
        Has an entry for every TE tunnel that traverses, starts or ends in this tunnel.
        Entries in this table can be generated from other routers.
        Entry format:
            tunnel name => {'FEC':fec, 'tuple': (<link or 3-tuple of nodes to be FRR protected>)}

    requested_bypasses:
        Has an entry for every link of 3-tuple of nodes for which a FRR protection bypass
        lsp must be computed. Entries in this table can be generated from other routers.
        Entry format:
            <link or 3-tuple of nodes to be FRR protected> => {'FEC': fec, 'next_hop': next_hop }


    For initialization indicate the router.
    """

    start_mode = 'manual'   # we must wait until all RSPV clients are initializated
                            # before starting to negotiate tunnels.
    protocol = "RSVP-TE"

    def __init__(self, router, max_hops = 3):
        super().__init__(router)
#         self.protocol = "RSVP-TE"
        self.build_order = 100

        self.bypass_cost = 16385          # Cost/weight allocated to bypass routes.
        self.frr_max_hops = max_hops      # Max number of additionals hops a FRR bypass may have.
        self.headended_lsps = dict()      # Table of tunnels configured to start in this router.
        self.requested_lsps = dict()      # Table of tunnels passing through this router.
        self.requested_bypasses = dict()  # Table of FRR backup paths passing through this router.


    def define_lsp(self, tailend, tunnel_local_id = 0, weight='weight',protection = None, **kwargs):
        # Store a request for a new tunnel; a headended LSP.
        # Compute the main path for it.
        # kwargs are reserved for future use, to pass restrictions for LSP creation.

        # Allowed protections:
        #  None: no protection at all.
        #  "facility-link": attempt only many-to-one link protection.
        #  "facility-node": First attempt many-to-one node protection, if imposible then link.
        #  "one-to-one": (WARNING: TBI) 1 to 1 protection, towards the tailend, trying to avoid overlapping with original LSP.

        # define tunnel name
        lsp_name = "lsp_from_{}_to_{}-{}".format(self.router.name, tailend, tunnel_local_id)
        # Check existence:
        if lsp_name in self.headended_lsps.keys():
            raise Exception("Requested creation of preexisting tunnel {}!. \
                            You might want to use a different tunnel_local_id.".format(lsp_name))

        G = self.router.topology

        constraints = None  #for future use
        if constraints:
            # Constraints can be a tuple of function allowing to build a view (subgraph)
            filter_node, filter_edge = constraints
            # Compute subgraph
            G = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
            # There could also be the 'subgraph_view' object itself
            # To consider link's delay, loss, price, or color, the topology itself should be already
            # labeled and the filter functions leverage that information.

        try:
            headend = self.router.name
            # compute spath: find the shortest path from headend to tailend.
            spath = nx.shortest_path(G, headend, tailend, weight=weight) #find shortest path, stored as list.
            length = len(spath) #numer of routers in path, including extremes.

            # Placeholder: attempt to find secondary path (To be implemented)
            sec_path = None
            # The idea is to (pre)compute a secondary path for the tunnel, sharing as few
            # links as possible with the primary path in order to be used as alternative
            # to FRR when the failure is adjacent to the headend, or as best possible path
            # after a failure FRR kicked in.
            #
            # A pseudocode idea follows, untested:
            # F = list(zip(spath[:-1],spath[1:]))  # transform the patyh from list of routers to list of links.
            # num_hops = len(F)
            # for i in range(num_hops-1):
            #     F = F[:-1]  # I avoid the furthest link if I have to choose.
            #     def filter_node_sec(n):
            #         #return False if n in L else True
            #         return True
            #     def filter_edge_sec(n1,n2):
            #         if (n1,n2) in F or (n2,n1) in F:
            #             return False
            #         return True
            #
            #     GG = nx.subgraph_view(G, filter_node = filter_node_sec, filter_edge = filter_edge_sec)
            #     try:
            #         sec_path = nx.shortest_path(GG, headend, tailend, weight=weight) #find shortest path, stored as list.
            #     except NetworkXNoPath:
            #         continue   #No path found, try with fewer constraints.

            cost = 0
            for i in range(length-1):
                cost += G.edges[spath[i],spath[i+1]][weight]

            #create a FEC object for this tunnel and store it in the pending configuration .
            self.headended_lsps[lsp_name] = { 'FEC': oFEC("TE_LSP",
                                                          lsp_name,
                                                          (self.router.name, tailend, protection)
                                                         ),
                                              'path': spath,
                                              'sec_path': sec_path,
                                              'cost': cost }
            return lsp_name

        except nx.NetworkXNoPath as err_nopath:
            return None

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
            sec_path = lsp_data['sec_path']   #alternative shortest path result, as list of hops..
            paths = [spath,sec_path]
            fec = lsp_data['FEC']      #an oFEC object with type="TE_LSP"
            protection = fec.value[2]  #protection mode

            length = len(spath)

            # create entry in local requested_lsps. value should be spath (a list object)
            self.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': (spath[0],spath[1]), 'path': spath}
            # a bad patch (not 100% RFC compliant) but if we don't make it, poor success
            if ((spath[0],spath[1]) not in self.requested_bypasses.keys()) and protection:
                self.requested_bypasses[(spath[0],spath[1])] = dict()

            #create entry in tailend
            tailend_pRSVP = network.routers[spath[-1]].clients["RSVP-TE"]
            self.comm_count += 1
            tailend_pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'path': spath}

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

                else:
                    #circumvent next node
                    facility = spath[i+1] #original next node
                    MP = spath[i+2]
                    protected_tuple = (PLR,facility,MP)


                # create a protection for all LSPs that traverse the protected_tuple
                if protection in ["facility-link", "facility-node"]:
                    self.comm_count += 1
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': protected_tuple,  'path': spath}

                    # create entry in requested_bypasses (if it doent's exist) of
                    # downstream nodes.
                    # Key should be the corresponding protected 3-tuple or 2-tuple.
                    # bypass_name = "bypass_{}_{}_{}".format(*triplet)
                    if protected_tuple not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[protected_tuple] = dict()

                elif protection == "one-to-one":
                    # use tailend as MP
                    self.comm_count += 1
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': (PLR,spath[-1]),  'path': spath}

                    # create entry in requested_bypasses (if it doent's exist) of
                    # downstream nodes.
                    # Key should be the corresponding protected 3-tuple or 2-tuple.
                    # bypass_name = "bypass_{}_{}_{}".format(*triplet)
                    if protected_tuple not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[protected_tuple] = dict()

                elif not protection:
                    # use tailend as MP
                    self.comm_count += 1
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': protected_tuple,  'path': spath}



    def compute_bypasses(self):
        # Compute all requested bypasses LSPs from the formation of LSPs during commit_config().
        #
        # This function must only be executed only execute after all RSVP-TE nodes
        # have executed and completed commit_config().
        # This function must be executed on all nodes in the network before actually requesting for known
        # resources.

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
        for protected_tuple in self.requested_bypasses.keys():

            G = self.router.topology   #change G to add constraints
            PLR = protected_tuple[0]   # Point of Local Repair
            MP = protected_tuple[-1]   # Merge Point

            if len(protected_tuple) == 3:
            #Node failure / NextNextHop  protection
                facility = protected_tuple[1] # potentially failed node

                # Auxiliary filter functions for this case
                def filter_node(n):
                    # Remove potentially failed node from the graph.
                    return True if n != facility else False
                def filter_edge(n1,n2):
                    return True

                # Compute subgraph
                view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
                # Get bypass path
                bypass_path = _get_path(view, PLR, MP, weight='weight', max_hops = self.frr_max_hops)

                if not bypass_path:
                    # if we can't protect the next node, attempt to protect the link.
                    MP = facility  # We will merge on the next node downstream instead of skipping it.


            if len(protected_tuple) == 2 or not bypass_path:
            #Link failure / NextHop  protection

                # Auxiliary filter functions for this case
                def filter_node(n):
                    return True
                def filter_edge(n1,n2):
                    # Remove potentially failed link from the graph.
                    return False if (n1,n2) == (PLR, MP) or (n2,n1) == (PLR, MP) else True

                # Compute subgraph
                view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
                # Get bypass path
                bypass_path = _get_path(view, PLR, MP, weight='weight', max_hops = self.frr_max_hops)

                if not bypass_path:
                    # We could not find a valid protection, move on to next request.
                    # Nevertheless an empty entry would remain in requested_bypasses.
                    continue


            # Found an usable bypass path! Create a FEC object for it.
            bypass_name = "bypass_{}".format("_".join([str(item) for item in protected_tuple ]))
            bypass = oFEC("bypass", bypass_name, protected_tuple)

            # Iterate through the bypass path
            length = len(bypass_path)
            for i in range(0, length):
                # Note: we compute protections only for intermediate nodes.
                #Get current router in path, and its RSVP process.
                curr_router = bypass_path[i]
                pRSVP = network.routers[curr_router].clients["RSVP-TE"]

                # Get its nexthop, it is needed to set up the path.
                if i+1 < length:
                    next_hop = bypass_path[i+1]
                else:
                    #for the MP, we will need a local label.
                    next_hop = None

                # Insert a bypass request on the router.
                if protected_tuple not in pRSVP.requested_bypasses.keys():
                    pRSVP.requested_bypasses[protected_tuple] = dict()  #initialize if necesary

                pRSVP.requested_bypasses[protected_tuple] = {'FEC': bypass,
                                                             'next_hop': next_hop,
                                                             'bypass_path': bypass_path }
                if curr_router != self.router.name:
                    self.comm_count += 1


    def known_resources(self):
        # Returns a generator to iterate over all tunnels and bypasses (RSVP-TE resources).

        if not self.requested_lsps and not self.requested_bypasses:
            # ProcRSVPTE in not initialized.
            return None

        network = self.router.network

        #First return the bypases
        for bypass_tuple, bypass_data in self.requested_bypasses.items():
            if 'FEC' not in bypass_data.keys():
                # The requested bypass was impossible/unsolvable.
                continue
            fec_candidate = bypass_data['FEC']
            fec = network.fec_lookup(fec_candidate)
            yield fec

        # return the tunnel LSPs
        for lsp_name, lsp_data in self.requested_lsps.items():
            fec_candidate = lsp_data['FEC']
            fec = network.fec_lookup(fec_candidate)
            yield fec


    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network

        # Starting with bypasses
        if  fec.fec_type == "bypass":
            if not fec.value:
                return   # this case should never happen.

            # recover the intended protected tuple.
            tuple_data = fec.value
            if router.name not in tuple_data:
                #intermediate router, process the bypass like a regular LSP
                next_hop_name = self.requested_bypasses[tuple_data]['next_hop']

                next_hop =  network.routers[next_hop_name]
                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec, do_count = False) #We cached it
                cost = self.bypass_cost

                # A swap op will do.
                if remote_label == self.IMPLICIT_NULL:
                    routing_entry = { "out": next_hop.name, "ops": [{"pop": ""}], "weight": cost  }
                else:
                    routing_entry = { "out": next_hop.name, "ops": [{"swap": remote_label}], "weight": cost  }
                yield (local_label, routing_entry)


            elif router.name == tuple_data[-1]:
                #this is the MP for the requested bypass. No next_hop, just pop.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

            #there is another possibility: that the router is the intermediate node. If this is the case it is because only a
            # link protection could be found  #FIX attempt number 1. #DEADLINE
            elif len(tuple_data) == 3 and router.name == tuple_data[1]:
                    #this is the MP for the requested bypass. No next_hop, just pop.
                    local_label = self.get_local_label(fec)
                    routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                    yield (local_label, routing_entry)



        if  fec.fec_type == "TE_LSP":
            headend, tailend, protection = fec.value

            # if 'tuple' not in self.requested_lsps[fec.name].keys():  #There should be a tidier way
            if router.name == tailend:
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
                remote_label = self.get_remote_label(next_hop_name, fec, do_count = False) #we cached it
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
                if not protection or tuple_data not in self.requested_bypasses.keys():
                # if tuple_data not in self.requested_bypasses.keys():
                    return


                # FRR -- Fast Re Route requirements.

                # Check feasability.
                bypass_data = self.requested_bypasses[tuple_data]
                if 'FEC' not in bypass_data.keys():
                    # The requested bypass is impossible/unsolvable, we are done.
                    return

                # Feasability confirmed, gather forwarding information
                bypass_fec = bypass_data['FEC']
                bypass_next_hop_name = bypass_data['next_hop']
                bypass_next_hop = network.routers[bypass_next_hop_name]
                bypass_path = bypass_data["bypass_path"]

                if MP_name != bypass_path:
                    MP_name = bypass_path[-1]  #force it
                # We need to know the label that the MP expects
                merge_label = self.get_remote_label(MP_name, fec)
                # And the bypass label for the FRR next hop
                bypass_label = self.get_remote_label(bypass_next_hop_name, bypass_fec, do_count = False) #we cahced it
                # Fix the entry priority
                cost = self.bypass_cost

                # This is a backup entry. It must push down a label on the stack to
                # encapsulate forwarding through the bypass path.

                backup_entry = { "out": bypass_next_hop.name, "weight": cost  }
                if merge_label == self.IMPLICIT_NULL and bypass_label == self.IMPLICIT_NULL:
                    r_info = [{"pop": ""}]
                elif merge_label == self.IMPLICIT_NULL:
                    r_info = [{"swap": bypass_label} ] #we don't push an IMPNULL
                else:
                    r_info = [{"swap": merge_label}, {"push": bypass_label} ]

                backup_entry["ops"] = r_info

                yield (local_label, backup_entry)

    def self_sourced(self, fec):
        # Returns True if this router is the tailend/mergepoint for this FEC.
        router = self.router
        network = router.network

        self_sourced = False

        if  fec.fec_type == "bypass":
            tuple_data = fec.value
            if router.name == tuple_data[-1]:
                # I am the MP.
                self_sourced = True

        elif fec.fec_type == "TE_LSP" and 'tuple' not in self.requested_lsps[fec.name].keys():
            # Then i am the tailend router. (There should be a tidier way to check this)
            self_sourced = True

        return self_sourced

    def get_good_sources(self, fec):
        # simulater helper method
        # must return a list of valid source routers for the fec`
        if fec.fec_type.startswith("bypass"):
            return []

        elif fec.fec_type == "TE_LSP":
            return [fec.value[0]]   #the headend router


    def get_good_targets(self, fec):
        # simulater helper method
        # must return a list of valid target routers for the fec`
        if fec.fec_type.startswith("bypass"):
            return []
        elif fec.fec_type == "TE_LSP":
            return [fec.value[1]]   #the tailend router
