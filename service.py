# coding: utf-8

"""
Implementation of Abstract MPLS VPN Services for MPLS Kit. - v0.1
Including abstractions for: VPRN, VWPS, pseudowire.

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

from mpls_fwd_gen import MPLS_Client, oFEC, Network, Router, as_list, dict_filter
import networkx as nx
from pprint import pprint

class MPLS_Service(MPLS_Client):
    """
    Class implementing a MPLS VPN service.

    This class is a skeleton for services allocating labels to different
    virtual private network schemas.

    Manages tuples of VPNs and attached CEs as resources/FECs.
    The fec_types might reflect different offerings.

    Requires access to the router's routing table, nexthop information,
    the linkstate database, local LDP/RSVP-TE and the remote labels to
    generate routing entries.

    Examples of services that might be emulated include point-to-point
    VPNs (pseudowires) as well as VPLS (Martini draft), yet this is not
    implemented yet.

    For initialization indicate the router.
    """

    start_mode = 'manual' # the network will try to allocate labels immediately.
    protocol = "service"


    def __init__(self, router, max_hops = 3):
        super().__init__(router)
        self.build_order = 1000

        # 'services' keys are services names, their values are dicts containing
        # service type, list of locally attached CEs and an optional service
        # description. vnp_name => {ces:[], vpn_type: , vpn_descr: "" }
        self.services = dict()
        self.fec_type = "vpn_endpoint"

    def define_vpn(self, vpn_name, vpn_type="default", vpn_descr=""):

        if vpn_name in self.services:
            raise Exception("VPN {} already defined".format(vpn_name))
        self.services[vpn_name] = {"ces":[], "vpn_type":vpn_type, "vpn_descr": vpn_descr, "ack": False}


    def remove_vpn(self, vpn_name, vpn_type="default", vpn_descr=""):
        if vpn_name not in self.services:
            raise Exception("Can't remove a VPN (){}) that is not yet defined".format(vpn_name))
        del self.services[vpn_name]

    def attach_ce(self, vpn_name, ce):
        if vpn_name not in self.services:
            raise Exception("VPN {} is not defined.".format(vpn_name))
        self.services[vpn_name]["ces"].append(ce)

    def dettach_ce(self, vpn_name, ce):
        if vpn_name not in self.services:
            raise Exception("VPN {} is not defined.".format(vpn_name))
        self.services[vpn_name]["ces"].remove(ce)

    def get_service_from_ce(self,ce):
        def _call(t):
            return True if ce in t[1]["ces"] else False

        return dict_filter(self.services, _call)

    def locate_service_instances(self, vpn_name):
        network = self.router.network

        # Filter routers that have services 'vpn_name' instantiated
        s_objs = [s.clients[self.protocol] for s in network.routers.values() if self.protocol in s.clients.keys()]
        s_instances = [s for s in s_objs if vpn_name in s.services.keys() ]

        return s_instances


    def known_resources(self):
        # Return FECs for each attached circuit (pe,ce) of each vpn instantiated on this router.

        network = self.router.network

        for vpn_name, vpn_data in self.services.items():
            for service in self.locate_service_instances(vpn_name):  #this includes me!!
                vpn_data = service.services[vpn_name]
                if not vpn_data["ack"]:
                    service.services[vpn_name]["ack"] = True
                    self.comm_count += 1

                pe = service.router.name
                for ce in vpn_data["ces"]:
                    fec_candidate = oFEC(self.fec_type,"vpn_ep_{}_{}_{}".format(vpn_name, pe, ce), (vpn_name, pe, ce))
                    fec = network.fec_lookup(fec_candidate)
                    yield fec

    def alloc_labels_to_known_resources(self):
        # Asks the router for allocation of labels for each known FEC (resource)
        for fec in self.known_resources():
            if self.self_sourced(fec):
                self.LIB_alloc(fec)
            else:
                # Don't allocate an actual label, directly the name.
                # This introduces a non-determinism (indirection)
                # that has to be solved by the FIB (e.g: on VPLS,
                # by ading MAC learning)
                # self.LIB_alloc(fec, literal = "NL_"+fec.name)   # NL_ stands for NO LABEL, null.

                # what if I just assume that some process will map packets to
                # the correconding FEC (service, router, ce) and here I just
                # care about reacheability
                self.LIB_alloc(fec)

    def LFIB_compute_entry(self, fec):
        # Each client must provide an generator to compute routing entries given the fec.
        # returns tuple (label, routing_entry)
        # routing entries have format:
        #  routing_entry = { "out": next_hop_iface, "ops": [{"push"|swap": remote_label}], "weight": cost  }

        # only PEs are concerned!
        # 1. Gather all PEs where the service is instantiated: SPEs (Service PEs)
        if fec.fec_type != self.fec_type:
            # invalid fec type, nothing to do!
            return

        vpn_name, pe, ce = fec.value
        network = self.router.network

        if vpn_name not in self.services.keys():
            # vpn not instantiated here
            return

        pe_router = network.routers[pe]
        pe_service_proc = pe_router.clients[self.protocol]
        if vpn_name not in pe_service_proc.services.keys():
            # vpn not instantiated on pe router (should not happen...)
            return

        # 2. Find local LSP to each PE
        local_label = self.get_local_label(fec)
        if pe == self.router.name  and self.self_sourced(fec):
            # compute entries for locally attached service interfaces (local ACs)
            routing_entry = { "out": ce, "ops": [{"pop": "" }], "weight": 0  }
            yield (local_label, routing_entry)

        else:
            # compute entries for remotely attached service interfaces (remote ACs)

            service_label = self.get_remote_label(pe, fec)
            tunnel_label = None

            # try RSVP-TE:
            lsp_name = "lsp_from_{}_to_{}".format(self.router.name, pe)
            candidate_tunnel_fec = self.get_fec_by_name_matching(lsp_name)

            if candidate_tunnel_fec:
                # get list of fecs with lower weight (might be many)
                refined = []
                curr_weight = 9999999999999   # arbitrary initial high value
                for cfec in candidate_tunnel_fec:
                    best = None
                    clabel = self.get_local_label(cfec)
                    for routing_entry in as_list(self.router.get_routing_entry(clabel)):
                        if not routing_entry:
                            continue
                        if routing_entry["weight"] < curr_weight:
                            best = cfec
                            curr_weight = routing_entry["weight"]
                    refined.append((best,curr_weight))

                curr_weight = min(refined,key=lambda x:x[1])[1]
                refined = [ tupl[0] for tupl in refined if  tupl[1] == curr_weight]

                # Choose first tunnel. (This introduces a non-determinism, I could have done other things...)
                tunnel_label = self.get_local_label(refined[0])

            if not tunnel_label:
                # try LDP:
                candidate_tunnel_fec = oFEC("loopback","lo_{}".format(pe), pe)
                tunnel_label = self.get_local_label(candidate_tunnel_fec)
                # print("ON: ", self.router.name ,"TRY LDP, fec: ", candidate_tunnel_fec.name, " tunnel_label: ", tunnel_label)

            if not tunnel_label:
                # ?I don't know how to reach pe_router inside MPLS
                return


            # 3. For each local fwd rule, add a new one swapping!! the service label.
            for routing_entry in as_list(self.router.get_routing_entry(tunnel_label)):
                if routing_entry is None:
                    print("Something is wrong!!!")
                    print("tunnel_label: ", tunnel_label)
                new_ops = routing_entry["ops"].copy()
                if "swap" in new_ops[0].keys():
                    lsp_label = new_ops[0]["swap"]
                    new_ops[0] = {"push": lsp_label} # recall: swap = pop + push, drop the pop.
                    new_ops.insert(0, {"swap": service_label})
                elif "pop" in new_ops[0].keys() and self.router.php:
                    # packet coming from AC (attached CE) don't have MPLS labels, drop the pop op.
                    # We know this can happen with PHP enabled.
                    new_ops[0] = {"swap": service_label} # replace the pop.
                else:
                    new_ops.insert(0, {"swap": service_label})

                new_routing_entry = { "out":routing_entry["out"],
                                      "ops": new_ops,
                                      "weight": routing_entry["weight"]  }

                yield (local_label, new_routing_entry )

    def self_sourced(self, fec):
        # Returns True if the FEC is sourced or generated by this process.
        self_sourced = False
        if  fec.fec_type == self.fec_type and type(fec.value) is tuple and len(fec.value) == 3:
            vpn_name, pe, ce = fec.value
            if vpn_name in self.services.keys():
                if pe == self.router.name and ce in self.services[vpn_name]["ces"]:
                    self_sourced = True
        return self_sourced

    def get_remote_label(self, router_name, fec):
        # Gets the label allocated by router <router_name> to the FEC <fec>
        router = self.router.network.routers[router_name]
        owner = router.get_FEC_owner(fec)
        if router.php and owner and owner.self_sourced(fec) and fec.fec_type != self.fec_type:
            # for services we ignore PHP!
            return self.IMPLICIT_NULL
        else:
            return self.router.network.routers[router_name].get_label(fec)

    def get_good_sources(self, fec):
        # simulater helper method
        # must return a list of valid source routers for the fec`
        if fec.fec_type == "vpn_endpoint":
            vpn_name = fec.value[0]
            tgt_pe = fec.value[1]
            tgt_ce = fec.value[2]
            good_sources = []
            for srv_inst in self.router.get_FEC_owner(fec).locate_service_instances(vpn_name):
                good_sources.append(srv_inst.router.name)

    def get_good_targets(self, fec):
        # simulater helper method
        # must return a list of valid target routers for the fec`
        if fec.fec_type == "vpn_endpoint":
            vpn_name = fec.value[0]
            tgt_pe = fec.value[1]
            tgt_ce = fec.value[2]
            good_targets = [tgt_pe]   #actually we don't have implemented a way of checking delivery to a CE
