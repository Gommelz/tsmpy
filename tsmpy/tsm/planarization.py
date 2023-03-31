from .utils import convert_pos_to_embedding
from tsmpy.dcel import Dcel
import networkx as nx
from networkx.algorithms import PlanarEmbedding


critical_edges = {}
help_nodes = []
help_to_critical_nodes = {}
bend_to_normal_edges = {}
critical_node_with_help_nodes = {}

# contains all edges of new embedding (one way edge) except edges between help nodes
normal_to_replaced_edges = {}
replaced_to_bend_edges = {}

class Planarization:
    """Determine the topology of the drawing which is described by a planar embedding.
    """

    def __init__(self, G, pos=None):
        if pos is None:
            is_planar, embedding = nx.check_planarity(G)
            # only tested for pos = None
            embedding = self.create_new_embedding(embedding)
            pos = nx.combinatorial_embedding_to_pos(embedding)
            G = self.embedding.to_undirected(reciprocal=True)
        else:
            embedding = convert_pos_to_embedding(G, pos)
            pass

        self.G = G.copy()
        self.dcel = Dcel(G, embedding)
        self.dcel.ext_face = self.get_external_face()
        self.dcel.ext_face.is_external = True

    # gets the face with the most surrounded nodes. The face do not only consist of 
    def get_external_face(self):
        sorted_faces = list(self.dcel.faces.values())
        sorted_faces.sort(key=lambda face: len(list(face.surround_vertices())), reverse=True)
        
        for face in sorted_faces:
            is_surrounded = face.surrounded_by_help_nodes
            if is_surrounded:
                sorted_faces.remove(face)
        
        return sorted_faces[0]
    
    def create_new_embedding(self, curr_embedding: PlanarEmbedding):
        """
        Takes a current embedding with critical nodes (nodes with degree heigher than 4) 
        and creates a new embedding with maximum node degree of 4.
        """

        opt_embedding = PlanarEmbedding()
    
        ringNodes = self.add_nodes(curr_embedding, opt_embedding)

        self.add_edges(curr_embedding, opt_embedding)
                
        # adds half edges between the newly created sub-nodes of a node with high degree to build a ring
        self.connect_ringnodes(opt_embedding, ringNodes)
        
        return opt_embedding
    
    
    
    def add_nodes(self, curr_embedding: PlanarEmbedding, opt_embedding: PlanarEmbedding):
        """
        Checks for all nodes of old_embedding if node is normal (deg(node) < 5) or critical (deg(node) >= 5).
        If node is critical for each neighbor a new help node will be added in the opt_embedding.
        Current notation: if node u is critical and has neighbor 1, the new help node will be named u.1.
        """

        # stores old critical node with new ring nodes
        ringNodes = {}

        # add new nodes
        # if deg(v) < 5 then add new node
        # else create "ring" of critical node
        for node in curr_embedding.nodes:
            if curr_embedding.degree(node) / 2 < 5:
                opt_embedding.add_node(node, status="old")
            else:
                newNodes = []
                neigh_gen = curr_embedding.neighbors_cw_order(node)
        
                for neighbor in neigh_gen:
                    opt_embedding.add_node(f"{node}.{neighbor}", status="new")
                    newNodes.append(f"{node}.{neighbor}")
                    help_nodes.append(f"{node}.{neighbor}")
                    help_to_critical_nodes[f"{node}.{neighbor}"] = node
                ringNodes[node] = newNodes
                critical_node_with_help_nodes[node] = newNodes
        return ringNodes

    def get_pred_succ_of_ringnode(self, node, newNodes: list):
        """
        Returns the predecessor and the successor of the element node in the list newNodes.
        """

        i = newNodes.index(node)
        pred = newNodes[i-1]
        if i + 1 >= len(newNodes):
            succ = newNodes[0]
        else:
            succ = newNodes[i+1]
        return pred, succ

    def add_edges(self, curr_embedding: PlanarEmbedding, opt_embedding: PlanarEmbedding):
        """
        Adds edges of old_embedding except from/to critical nodes to new opt_embedding.
        Adds edges between new help nodes and the neighbors of the original critical nodes.
        """

        for node in curr_embedding.nodes:
            neigh_gen = curr_embedding.neighbors_cw_order(node)
            for neighbor in neigh_gen:
                if opt_embedding.has_node(node):
                    order = list(curr_embedding.neighbors_cw_order(node))
                    i = order.index(neighbor)
                    if i - 1 < 0:
                        reference_node = None
                    else:
                        reference_node = order[i - 1]
                        if not opt_embedding.has_node(reference_node):
                            reference_node = f"{reference_node}.{node}"
                    start_node = node           
                else:
                    reference_node = None
                    start_node = f"{node}.{neighbor}"

                if opt_embedding.has_node(neighbor):
                    # deg(neighbor) < 5
                    end_node = neighbor
                else:
                    # deg(neighbor) >= 5
                    end_node = f"{neighbor}.{node}"

                # original
                if not (neighbor, node) in normal_to_replaced_edges:
                    normal_to_replaced_edges[(node, neighbor)] = (start_node, end_node)
                opt_embedding.add_half_edge_cw(start_node, end_node, reference_node)

    def connect_ringnodes(self, opt_embedding: PlanarEmbedding, ringNodes: dict):
        """
        For each old critical node all new help nodes will be connected in a ring.
        """

        for node, newNodes in ringNodes.items():
            for newNode in newNodes:
                pred, succ = self.get_pred_succ_of_ringnode(newNode, newNodes)
                
                assert len(list(opt_embedding.neighbors_cw_order(newNode))) == 1
                
                opt_embedding.add_half_edge_cw(newNode, succ, list(opt_embedding.neighbors_cw_order(newNode))[0])

                opt_embedding.add_half_edge_cw(newNode, pred, succ)