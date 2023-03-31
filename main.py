# in root dir
import networkx as nx
from tsmpy import TSM
from matplotlib import pyplot as plt

# G = nx.Graph(nx.read_gml("test/inputs/case2.gml"))
G = nx.Graph(nx.read_graphml("test/inputs/test.graphml"))


# initial layout, it will be converted to an embedding
pos = {node: eval(node) for node in G}

# pos is an optional, if pos is None, embedding will be given by nx.check_planarity

# use linear programming to solve minimum cost flow program
tsm = TSM(G, pos, uselp=False)

# or use nx.min_cost_flow to solve minimum cost flow program
# it is faster but produces worse result
# tsm = TSM(G, pos, uselp=False)

tsm.display()
plt.savefig("test/outputs/case2.nolp.svg")
# plt.close()