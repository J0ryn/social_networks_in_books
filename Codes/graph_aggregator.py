#The file containing the class responsible for storing and creating the networks
import networkx as nx
from typing import TypeVar, Callable, Any, Type

from literature_objects import Segment

T = TypeVar("T", bound=nx.Graph)

def create_graph(relationships: list[dict], graph_type: T = nx.Graph):
    G = graph_type

    for character in relationships:
        G.add_node(character['name'])

    for relationship in relationships:
        for connection in relationship['connections']:
            G.add_edge(relationship['name'], connection['name'], sentiment=connection['description'])

    return G

def merge_graphs(G_base: T, G_new: T, aggregrate_function: Callable[[Any, Any], Any] = lambda x, y: x+y , weight_name: str = "sentiment" ) -> T:

    if type(G_base) != type(G_new):
        raise Exception(f"Both graphs should be of the same type. \n G1 type: {type(G_new)} \n G2 type: {type(G_base)}")

    G_total = G_base.copy()
    
    
    G_total.add_nodes_from(G_new.nodes(data=True))
    

    for u, v, data in G_new.edges(data=True):
        new_sentiment = data.get(weight_name, 0)
        
        if G_total.has_edge(u, v):
            G_total[u][v][weight_name] = aggregrate_function(G_total[u][v][weight_name], new_sentiment)
        else:
            G_total.add_edge(u, v, **data)
            
    return G_total

class SceneAggregator:
    """Handles the storage, merging, and logic of network scenes."""
    
    def __init__(self, graph_type: Type[nx.Graph] = nx.Graph):
        self.scenes = [] #stores the graphs of the scenes, the indexes should correspond to the self.segments indexes 
        self.segments = [] #stores Segment objects with all data regarding that scene, the indexes should correspond to the self.scenes indexes
        self.graph_factory = graph_type #sets the graph type to be returned when the scenes are aggregated. Should be the same as the graphs in self.scenes
        self.agg_func = lambda x, y: x + y 
        self.min_connections = 0

    def __repr__(self):
        return f"Network built from {len(self.scenes)} graph(s)\n" + "\n".join([str(G) for G in self.scenes])

    def add_scene(self, G_scene: nx.Graph, segment: Segment = Segment(start_phrase = "No startphrase given", end_phrase= "No endphrase given")):
        """Adds a scene to the list of scenes. If not provided with a segment, an empty Segment object is added."""
        self.scenes.append(G_scene)
        self.segments.append(segment)
    
    def add_segment(self, segment: Segment):
        """Adds a segment to the list of segments and creates the corresponding scene graph for it."""
        self.segments.append(segment)
        self.scenes.append(create_graph(segment.network["characters"], self.graph_factory()))
    
    def set_agg_func(self, func: Callable[[Any, Any], Any]):
        """Sets the function used to combine the edges in the scene graphs. The default function is lambda x, y: x + y"""
        self.agg_func = func

    def set_min_connections(self, min_connections: int) -> None:
        """Sets the min_connections attribute of the class. The get_network_at_scene function returns a graph with nodes having at least min_connections degree."""
        self.min_connections = min_connections

    def get_network_at_scene(self, scene_index: int, use_min_connections=True) -> nx.Graph:
        """Creates the aggregated network up until a specific index."""
        
        scene_index = min(scene_index, len(self.scenes) - 1)

        aggregate = self.graph_factory()
        for i in range(scene_index + 1):
            aggregate = merge_graphs(aggregate, self.scenes[i], self.agg_func) # Assuming merge_graphs is defined elsewhere

        if self.min_connections > 0 and use_min_connections:
            nodes_to_delete = [node for node in aggregate.nodes() if aggregate.degree(node) < self.min_connections]
            aggregate.remove_nodes_from(nodes_to_delete)

        return aggregate

    def get_absolute_network_at_scene(self, scene_index: int, aggregate=True) -> nx.Graph:
        """Returns the network with absolute sentiment values."""

        G = self.get_network_at_scene(scene_index).copy() if aggregate else self.scenes[scene_index].copy()
        
        for u, v, d in G.edges(data=True):
            d['sentiment'] = abs(d.get('sentiment', 0)) 
        return G
