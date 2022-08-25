# coding: utf-8

"""
Recursive MPLS - RMPLS - v0.1

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
Created by Juan Vanerio, 2022

"""

from cmath import inf
from mpls_fwd_gen import MPLS_Client, oFEC, Network, Router
from rsvpte import ProcRSVPTE
import networkx as nx
from pprint import pprint


class ProcRMPLS(ProcRSVPTE):

    protocol = "RMPLS"

    def __init__(self, router, max_hops = 999999999, rmpls_mode = "link"):
        super().__init__(router, max_hops)
        self.build_order = 10000
        self.bad_protection_pairs = []
        self.bad_protection_computations_covered = [] #We cache which protections we have done computation for
        self.path_cache = dict()
        self.rmpls_mode = rmpls_mode
        print(f"Creating RMPLS process in router {router} )")

    def access_client(self, router_name, client_function):
        return super().access_client(router_name, ProcRMPLS, client_function)

    def compute_bypasses(self):
        # function in charge of computing the protections for local edges, meaning:
        # [...,tuple( local_edge, bypass_path),...] with as many bypass_paths per edge as required.

        edge_cache = dict()

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)
                if not bypass_path or len(bypass_path) >= max_hops + 2: #bypass too large
                    return []
                else:
                    return [bypass_path]
            except nx.NetworkXNoPath as err_nopath:
                return []

        G = self.router.network.topology

        for neig in G.neighbors(self.router.name):
            # Auxiliary filter functions for this case
            e = (self.router.name, neig)

            def node_protection(edge):
                def filter_node(n): # Remove potentially failed node from the graph.
                    return n != edge[1]
                # Compute subgraph
                view = nx.subgraph_view(G, filter_node = filter_node)
                # Get bypass path to each next-next hop.
                bypass_paths = []
                for next_next in G.neighbors(edge[1]):
                    if next_next != edge[0]:
                        bypass_paths += _get_path(view, edge[0], next_next, weight='weight', max_hops = self.frr_max_hops)
                return bypass_paths

            def link_protection(edge):
                def filter_edge(n1,n2): # Remove potentially failed link from the graph.
                    return False if ((n1,n2) == edge or (n2,n1) == edge) else True
                # Compute subgraph
                view = nx.subgraph_view(G, filter_edge = filter_edge)
                # Get bypass path
                bypass_paths = _get_path(view, edge[0], edge[1], weight='weight', max_hops = self.frr_max_hops)
                return bypass_paths

            bypass_paths = []
            if self.rmpls_mode == "link":
                bypass_paths = [(p,1) for p in link_protection(e)]
            elif self.rmpls_mode == "facility":
                bypass_paths = [(p,1) for p in node_protection(e)] + [(p,2) for p in link_protection(e)]  # Use node protection if possible, else (secondarily) use link protection
            else:
                print("WARNING: unknown mode for R-MPLS: ", self.rmpls_mode)

            # Collect the bypass requests for each router
            bypass_requests_for_router = {}
            for bypass_path, priority in bypass_paths:
                bypass_path = tuple(bypass_path)

                bypass_path_edges = self.path_routers_to_edges(bypass_path)

                # Ensure unique names
                if e not in edge_cache:
                    edge_cache[e] = 0
                else:
                    edge_cache[e] += 1
                bypass_name = "bypass_rmpls_{}_{}".format("_".join([str(item) for item in e ]), edge_cache[e])

                bypass_fec = oFEC("bypass_rmpls", bypass_name, (e, bypass_path_edges))  #unique FEC per (edge, path)

                # Iterate through the bypass path
                length = len(bypass_path)
                for i in range(0, length):
                    curr_router = bypass_path[i]
                    if i+1 < length:
                        next_hop = bypass_path[i+1]
                    else:
                        #for the MP, we will need a local label.
                        next_hop = None
                    if curr_router not in bypass_requests_for_router:
                        bypass_requests_for_router[curr_router] = []
                    bypass_requests_for_router[curr_router].append({'FEC': bypass_fec, 'next_hop': next_hop, 'priority': priority})

            # Send the bypass requests in one message per router
            for router, bypass_requests in bypass_requests_for_router.items():
                self.access_client(router, lambda pRMPLS: pRMPLS.handle_bypass_requests(bypass_requests))

    def handle_bypass_requests(self, bypass_requests):
        for bypass_request in bypass_requests:
            if 'FEC' in bypass_request:
                self.requested_bypasses[bypass_request['FEC'].value] = bypass_request

    def known_resources(self):
        # Returns a generator to iterate over all local requested_bypasses.

        if not self.requested_bypasses:
            # RMPLS in not initialized.
            return None

        network = self.router.network

        for bypass_tuple, bypass_data in self.requested_bypasses.items():
            if 'FEC' not in bypass_data.keys():
                # The requested bypass was impossible/unsolvable.
                continue
            fec_candidate = bypass_data['FEC']
            fec = network.fec_lookup(fec_candidate)
            yield fec


    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a bypass identifier
        # The next_hop is already in the path information,
        # we require access to the next-hop allocated labels.

        router = self.router
        network = router.network
        # print(f"Start RMPLS first step LFIB_compute_entry (router {router.name})")
        # Starting with bypasses
        try:
            if fec not in used_ifaces.keys():
                used_ifaces[fec] = []   #for each FEC an interface can appear in at most a single entry.
        except:
            used_ifaces = dict()  #initialize
            used_ifaces[fec] = []

        if  fec.fec_type != "bypass_rmpls":
            raise Exception("Unknown FEC type.")

        if not fec.value:
            return   # this case should never happen.

        # recover the intended protected edge
        protection = fec.value
        bypass_path = protection[1]
        mp_name = bypass_path[-1][1]
        # print(self.router.name, protected_edge)
        if router.name != mp_name:
            data = self.requested_bypasses[protection]

            print(router.name, " data ", data)

            next_hop_name = data['next_hop']
            if next_hop_name in used_ifaces[fec]:
                # interface fec already used for this FEC, we can't allocate it again.
                return

            local_label = self.get_local_label(fec)
            remote_label = self.get_remote_label(next_hop_name, fec, do_count = False)
            self.comm_count += 1
            cost = self.bypass_cost

            # A swap op will do in the normal case.
            if remote_label == self.IMPLICIT_NULL:
                re_ops = [{"pop": ""}]
            else:
                re_ops = [{"swap": remote_label}]

            routing_entry = { "out": next_hop_name, "ops": re_ops, "weight": cost  }  #FIXME: all costs are equal!
            used_ifaces[fec].append(next_hop_name)
            yield (local_label, routing_entry)

        elif not router.php: # (router.name == mp_name), only install on MP, if we don't use PHP.
            #this is the MP for the requested bypass. No next_hop, just pop.
            local_label = self.get_local_label(fec)
            routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
            yield (local_label, routing_entry)

    def self_sourced(self, fec):
        # Returns True if this router is the tailend/mergepoint for this FEC.
        if  fec.fec_type == "bypass_rmpls":
            protected_edge, bypass_path = fec.value
            if self.router.name == bypass_path[-1][1]: # I am the MP.
                return True
        return False

    def path_routers_to_edges(self, router_path):
        if not router_path:
            return None
        return tuple(zip(router_path[:-1], router_path[1:]))

    def path_edges_to_routers(self, edge_path):
        if not edge_path:
            return None
        return tuple([edge_path[0][0]]+[e[1] for e in edge_path])

    def get_protection_paths_edges(self, edge):
        # This method allows another client to request bypass_path information stored in a third router.
        if edge[0] not in self.path_cache:
            self.path_cache[edge[0]] = self.access_client(edge[0], lambda pRMPLS: [x for x in pRMPLS.requested_bypasses if x[0][0] == edge[0]]) # Get all protections originating at edge[0]
        return [x[1] for x in self.path_cache[edge[0]] if x[0] == edge] # Filter: only protections for edge.


    def bad_protection_dfs(self, current_prot, present_set, visited):
            # finds and returns a cycle...
        # visited is a set of protections, while present_set is a set of links.

        current_edge, current_path = current_prot

        if current_edge in present_set or len(present_set.intersection({p[0] for p in visited})) > 0:
            # 'Good' loop i.e. it will not occur, since a link was assumed to be both active and failed
            return None
        # Check if loop was detected
        if current_prot in visited:
            return [current_prot]
        # Continue search on children

        new_present = set()
        i = 0
        for e in current_path:
            paths = self.get_protection_paths_edges(e)
            for path in paths:
                if path[-1][1] in [e[1] for e in current_path[i:]]: # End of the new paths is downstream on current path
                    other_prot = (e, tuple(path))  #make it hashable
                    if (current_prot, other_prot) not in self.bad_protection_pairs:
                        back_track = self.bad_protection_dfs(other_prot, present_set.union(new_present), visited.union({current_prot}))
                        if back_track is not None:
                            if len(back_track) < 2 or back_track[0] != back_track[-1]: # back_track not yet contains full cycle
                                back_track = [current_prot] + back_track # add current to back_track
                            return back_track
            new_present.add(e)
            i+=1
        return None

    def bad_protection_dfs_go(self, prot):
        cycle = self.bad_protection_dfs(prot, set(), set())
        if cycle is not None:
            assert(len(cycle) > 0 and cycle[0] == cycle[-1])
            potential_bad_protection_pairs = list(zip(cycle[:-1], cycle[1:])) # We need to remove at least one of these prot pairs to break the cycle.
            if len(potential_bad_protection_pairs) == 2: # If cycle has length 2, we can safely remove both prot-pairs.
                assert(len(set(self.bad_protection_pairs).intersection(set(potential_bad_protection_pairs))) == 0)
                self.bad_protection_pairs += potential_bad_protection_pairs # Store and use in future computations
            else: # For longer cycles, we only remove one prot-pair to break cycle, as these pairs can be good in some other scenarios.
                chosen_bad_protection_pair = sorted(potential_bad_protection_pairs)[0] # Take smallest pair, given some global prot order.
                assert(chosen_bad_protection_pair not in self.bad_protection_pairs)
                self.bad_protection_pairs.append(chosen_bad_protection_pair) # Store and use in future computations
            return True
        return False

    def compute_bad_protection_pairs_from(self, protection):
        # print("running compute_bad_protection_pairs_from")
        # This function computes the pairs of protections (A,B) such that when on the bypass_path for A, if B also fails, we will not enter B's bypass_path like we normally would with RMPLS.
        # We do this to avoid forwarding loops. The algorithm is designed to guarantee that no forwarding loops can occur when using the computed bad_protection_pairs.
        if protection not in self.bad_protection_computations_covered: # Only compute once for each starting protection
            self.bad_protection_computations_covered.append(protection)
            while True: # Keep adding bad edge pairs as long as cycles are found.
                change = self.bad_protection_dfs_go(protection)
                if not change:
                    break

    def protects(self, label):
        fec_0, ow = self.router.get_FEC_from_label(label, get_owner=True)
        if ow['owner'] != self:
            return None

        protection = fec_0.value
        return protection

    def protectable_by(self, label, routing_entry, protection):
        edge, path = protection
        if edge[1] == path[-1][1]: # Link protection is always possible
            return routing_entry["ops"]
        # WARNING: THIS FUNCTION REQUIRES REVERSE ENGINEERING OF THE DATA PLANE
        # For node protection, we query the next router for the entries.
        ops = routing_entry["ops"]
        if len(ops) == 1 and "swap" in ops[0].keys():
            next_label = ops[0]["swap"]
            next_entries = self.access_remote(edge[1], lambda router: router.get_routing_entry(next_label)) # WARNING: This is the HACKY line!!!
            min_prio = min([v['priority'] for v in next_entries])
            filtered = [entry for entry in next_entries if entry['priority'] == min_prio and entry['out'] == path[-1][1]]
            if len(filtered) > 0:
                return filtered[0]["ops"]
        return None

    def virtual_forwarding(self, label, steps = 100):
        # WARNING: THIS FUNCTION ACTUALLY IMPLEMENTS REVERSE ENGINEERING OF THE DATA PLANE
        pass

    def get_protection_label(self, protection):
        fec = self.requested_bypasses[protection]["FEC"]
        return self.router.get_label(fec) # TODO: Why not self.get_local_label(fec) ?

    def LFIB_refine(self, label):
        # Some processes may require a refinement of the LFIB.

        # Skip implicit null labels.
        if label == self.IMPLICIT_NULL:
            return []

        # NOTE: this function is only valid for link protection. For node protection a different approach must be taken.
        #  Possible exception: deal with node protection as multiple link failure.
        r = self.router
        entries = r.get_routing_entry(label)  # LFIB query

        Q = sorted(entries, key = lambda x:x['priority'])
        max_prio = max([v['priority'] for v in Q]) + 1
        min_prio = min([v['priority'] for v in Q])
        Q = [v for v in Q if v['priority'] == min_prio] # Only protect highest priority entries, as other protections will be redundant
        additional_entries = []

        for q in Q:

            now_failing_edge = (r.name, q["out"])   #e

            protection_prime = self.protects(label)

            protections = [(protection, data['priority']) for protection,data in self.requested_bypasses.items() if protection[0] == now_failing_edge]
            protections = sorted(protections, key=lambda elem: elem[1])
            installed_protection_with_priority = inf
            for protection,priority in protections:
                if priority > installed_protection_with_priority: # stop if higher priority protection was already installed.
                    break
                p = protection[1]

                protection_label = self.get_protection_label(protection)

                if not protection_prime: # label is not part of a RMPLS protection path
                    q_prime = self.protectable_by(label, q, protection)
                    if q_prime is None:
                        continue

                    if q_prime == [{"pop":""}]:
                        new_ops = [{"swap": protection_label}]  # Replace pop,push by swap.
                    else:
                        new_ops = q_prime + [{"push": protection_label}]    # Line15
                    routing_entry = { "out": r.LOCAL_LOOKUP, "ops":new_ops , "priority": max_prio  }

                    additional_entries.append(routing_entry)
                    installed_protection_with_priority = priority

                else: # label is part of a RMPLS protection path
                    self.compute_bad_protection_pairs_from(protection)
                    if (protection_prime, protection) in self.bad_protection_pairs:   # New Loop protection.
                        continue

                    mp_name = p[-1][1]   #tgt_p = mp = merge point
                    p_prime = self.path_edges_to_routers(protection_prime[1])

                    my_pos = p_prime.index(r.name)
                    if mp_name not in p_prime[my_pos:]:
                        continue

                    if mp_name == p_prime[-1]:
                        new_ops = [{"swap": protection_label}]   #line 19a
                    elif p_prime.index(mp_name) == my_pos + 1: # For link protection we know the expected_label, so we can avoid communication.
                        new_ops = q["ops"] + [{"push": protection_label}]
                    else: # For protection that merge further down, we need to query the merge-point for the expected_label.
                        expected_label = self.access_client(mp_name, lambda pRMPLS: pRMPLS.get_protection_label(protection_prime))
                        new_ops = [{"swap": expected_label}, {"push": protection_label}] #line 19b

                    routing_entry = { "out": r.LOCAL_LOOKUP, "ops":new_ops , "priority": max_prio  }
                    # print(f"MPLS: add entry {routing_entry}")
                    additional_entries.append(routing_entry)
                    installed_protection_with_priority = priority


        # print("FOUND BAD PROTECTION PAIRS", self.bad_protection_pairs)
        return additional_entries
