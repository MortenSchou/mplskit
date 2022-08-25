# coding: utf-8

"""
Label Distribution Protocol (LDP) RFC 4090 implementation for MPLS Kit - v0.1

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

# import networkx as nx
from mpls_fwd_gen import MPLS_Client, oFEC, Network, Router
import networkx as nx
from pprint import pprint


class ProcLDP(MPLS_Client):
    """
    Class implementing a Label Distribution Protocol (LDP) client.

    This protocol is used for allocation of labels to (some) IP prefixes
    installed on the routing table. Its main purpose is to allow immediate
    forwarding after the IGP has solved shortest path routes in the network.
    It builds its LSPs on a hop-by-hop basis.
    Supports ECMP, in which case it is out of its scope to decide which packets
    should use which path.

    Manages IP prefixes as resources/FECs.
    The fec_types it handles are "link" and "loopback"

    Requires access to the router's routing table, nexthop information, the
    linkstate database and the remote labels to generate routing entries.

    LDP FRR is not implemented yet.

    For initialization indicate the router.
    """

    start_mode = "auto" # the network will try to allocate labels immediately.
    protocol = "LDP"

    def __init__(self, router):
        super().__init__(router)
        self.build_order = 200

    def known_resources(self):
        # LDP, known resources are all IP prefixes in the topology
        # We assume that each node in the topology has at least one
        # loopback interface with an IP address, and each link between
        # nodes also have an IP prefix.
        G = self.router.topology
        network = self.router.network
        # Return FECs for each loopback interface in the network.
        for node in G.nodes():
            fec_candidate = oFEC("loopback","lo_{}".format(node), node)
            fec = network.fec_lookup(fec_candidate)
            yield fec

        # Return FECs for each link in the network.
        for e in G.edges():
            edge = e
            # Use a canonical ordering in the undirected edges vertices.
            if e[0] > e[1]:
                edge = (e[1],e[0])  # always ordered.
            fec_candidate = oFEC("link", "link_{}_{}".format(edge[0],edge[1]),edge)
            fec = network.fec_lookup(fec_candidate)
            yield fec


    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is an IP prefix that we want to reach, so the computation
        # will require access to remote lables and to routing information.

        router = self.router
        network = router.network

        # Get root router object, the router we want to reach
        if  fec.fec_type == "loopback":
            root = network.routers[fec.value]  # a loopback fec value is its router name.

        elif fec.fec_type == "link":
            e = fec.value   # a link fec is a (canonically) ordered 2-tuple of nodes.
            e0, e1 = network.routers[e[0]], network.routers[e[1]]
            # the root has to be the closest node in the link.
            if e0.dist[router.name] <= e1.dist[router.name]:  #dist( router-> e0) vs. dist( router-> e1)
                root = e0
            else:
                root = e1

        #access routing information on how to reach the root router.
        if router.name == root.name:
            # access labels
            local_label = self.get_local_label(fec)
            # build routing entry
            routing_entry = { "out": router.LOCAL_LOOKUP, "ops": [{"pop": ""}], "weight": 0 }
            yield (local_label, routing_entry)

        for predecessor_name in root.pred[router.name]:  #for each next_hop from router towards root...
            # access labels
            local_label = self.get_local_label(fec)
            remote_label = self.get_remote_label(predecessor_name, fec)
            cost = root.dist[router.name]   #IGP distance from router towards root
            # build routing entry
            if remote_label == self.IMPLICIT_NULL:
                routing_entry = { "out": predecessor_name, "ops": [{"pop": ""}], "weight": cost }
            else:
                routing_entry = { "out": predecessor_name, "ops": [{"swap": remote_label}], "weight": cost }
            yield (local_label, routing_entry)
            if single:
                break  #compute for one predecessor, no load balancing.

    def self_sourced(self, fec):
        # Returns True if the FEC is sourced or generated by this process.
        router = self.router
        network = router.network
        self_sourced = False
        # Get root router object, the router we want to reach
        if  fec.fec_type == "loopback" and router.name == fec.value:
            # a loopback fec value is its router name.
            self_sourced = True

        elif fec.fec_type == "link" and router.name in fec.value:
            # a link fec is a (canonically) ordered 2-tuple of node names.
            self_sourced = True

        return self_sourced

    def get_good_sources(self, fec):
        # simulater helper method
        # must return a list of valid source routers for the fec`
        router_name = self.router.name
        good_sources = set(self.router.network.routers)

        if fec.fec_type == "loopback":
            if fec.name.endswith(router_name):
                good_sources.difference_update([router_name])

        elif fec.fec_type == "link":
            if "_"+router_name+"_" in fec.name or fec.name.endswith("_"+router_name):
                good_sources.difference_update([router_name])

        return good_sources


    def get_good_targets(self, fec):
        # simulater helper method
        # must return a list of valid target routers for the fec`
        if fec.fec_type == "loopback":
            good_targets = [fec.value]

        elif fec.fec_type == "link":
            good_targets = list(fec.value)

        return good_targets
