# The file containing the class responsible for visualizing instances of SceneAggregator objects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import networkx as nx
from typing import TypeVar, Callable, Any, Optional, Dict

from graph_aggregator import SceneAggregator

class SceneVisualizer:
    """Handles layout calculation, rendering, and animating of scenes."""
    
    def __init__(self, manager: SceneAggregator):
        self.manager = manager
        self.positions = {}
        self.cbar = None

    def calculate_incremental_layout(self, scene_index: int, iterations=50) -> dict:
        """Calculates positions based on the previous scene's coordinates."""
        G_current = self.manager.get_network_at_scene(scene_index)

        if scene_index == 0:
            pos = nx.spring_layout(G_current, seed=42)
        else:
            if (scene_index - 1) not in self.positions:
                self.calculate_incremental_layout(scene_index - 1)

            prev_pos = self.positions[scene_index - 1]
            
            pos = nx.arf_layout(
                G_current,
                pos=prev_pos,
                max_iter=iterations,
                seed=42
            )

        self.positions[scene_index] = pos
        return pos

    def _draw_frame(self, ax, i: int, min_connections=0, aggregate=True):
        """The single source of truth for drawing a scene."""
        if not aggregate:
            G_current = self.manager.scenes[i]
        else:
            G_current = self.manager.get_network_at_scene(i)

        if min_connections > 0:
            valid_nodes = [node for node, degree in G_current.degree() if degree >= min_connections]
            G_draw = G_current.subgraph(valid_nodes)
        else:
            G_draw = G_current

        pos = self.positions[i]
        edges_data = G_draw.edges(data=True)
        sentiment_list = [data.get('sentiment', 0) for u, v, data in edges_data] if edges_data else []
        
        ax.clear()
        
        if not edges_data:
            # Handle empty graph cleanly
            nx.draw_networkx_nodes(G_draw, pos, node_color="lightblue", node_size=150, ax=ax)
            ax.set_title(f"Network State: Scene {i}", fontsize=15, pad=25)
            ax.axis('off')
            return

        vmin, vmax = min(sentiment_list), max(sentiment_list)
        norm = TwoSlopeNorm(vmin=min(vmin, -0.1), vcenter=0, vmax=max(vmax, 0.1))
        cmap = LinearSegmentedColormap.from_list('sentiments', ['red', 'grey', 'green'], N=256)
        mapper = ScalarMappable(norm=norm, cmap=cmap)
        
        abs_sentiments = [abs(s) for s in sentiment_list]
        as_min, as_max = min(abs_sentiments) if abs_sentiments else 0, max(abs_sentiments) if abs_sentiments else 1
        
        alpha_min, alpha_max = 0.2, 1.0
        edge_colors = [mapper.to_rgba(s) for s in sentiment_list]

        # 1. Draw Nodes
        nx.draw_networkx_nodes(G_draw, pos, node_color="lightblue", node_size=150, ax=ax)
        
        # 2. Draw Edges
        # Safe handling for alpha division by zero if all sentiments are equal
        alpha_range = (as_max - as_min) if (as_max - as_min) > 0 else 1
        alphas = [(alpha_min + (alpha_max - alpha_min) * (abs_s - as_min) / alpha_range) - 0.01 for abs_s in abs_sentiments]

        nx.draw_networkx_edges(
            G_draw, pos,
            edgelist=list(G_draw.edges()),
            edge_color=edge_colors,
            alpha=alphas,
            connectionstyle="arc3, rad=0.1",
            width=1.5,
            arrows=isinstance(G_draw, nx.DiGraph),
            arrowsize=12,
            ax=ax,
        )

        # Handle Colorbar
        if self.cbar is not None:
            self.cbar.remove()
        
        self.cbar = ax.get_figure().colorbar(mapper, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        self.cbar.set_label('Sentiment Score', fontsize=10)
        
        nx.draw_networkx_labels(G_draw, pos, font_size=9, ax=ax)
        ax.set_title(f"Network State: Scene {i}", fontsize=15, pad=25)
        ax.text(0.5, 1.01, self.manager.descriptions[i], fontsize=11, ha='center', va='bottom', transform=ax.transAxes)
        ax.axis('off')

    def replay_evolution(self, min_connections = 0, aggregate=True):
        """
        Draws the evolution of the network scene by scene.

        params:
        min_connections (int): minimum degree a node can have in a graph draw. Default to number set in the SceneAggregator object, 0 if not overwritten.
        aggregate (bool): If set to true, the whole network is displayed. If set to false, only the scenes. Default to True.

        return None, draws matplotlib plots, as many as there are scenes
        """

        print(f"Generating {len(self.manager.scenes)} plots...")
        if min_connections == 0:
            min_connections = self.manager.min_connections

        for i in range(len(self.manager.scenes)):
            self.calculate_incremental_layout(i)
            fig, ax = plt.subplots(figsize=(12, 8))
            self._draw_frame(ax, i, min_connections, aggregate)
            plt.show() 

    def create_animation(self, frame_length=1000, min_connections=0, aggregate=True) -> FuncAnimation:
        """
        Draws the evolution of the network scene by scene in an animation.

        params:
        frame_length: The time each frame is displayed in miliseconds.
        min_connections (int): minimum degree a node can have in a graph draw. Default to number set in the SceneAggregator object, 0 if not overwritten.
        aggregate (bool): If set to true, the whole network is displayed. If set to false, only the scenes. Default to True.

        return FuncAnimation object
        """

        if min_connections == 0:
            min_connections = self.manager.min_connections

        for i in range(len(self.manager.scenes)):
            self.calculate_incremental_layout(i)
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update(frame):
            self._draw_frame(ax, frame, min_connections, aggregate)

        # FIX: Removed dead code that existed after the return statement
        return FuncAnimation(fig, update, frames=len(self.manager.scenes), interval=frame_length, repeat=False)
    
    def save_scene_as_png(self, scene_index: int, filename="scene.png", min_connections=0, aggregate=True):
        """
        Saves a network at a given index by a defined file name.

        params:
        scene_index (int): The index of the scene in the SceneAggregator object's scene list.
        filename (str): The name of the saved image. Default is "scene.png"
        min_connections (int): minimum degree a node can have in a graph draw. Default to number set in the SceneAggregator object, 0 if not overwritten.
        aggregate (bool): If set to true, the whole network is displayed. If set to false, only the scenes. Default to True.

        return None, draws matplotlib plots, as many as there are scenes
        """
        self.calculate_incremental_layout(scene_index)
        fig, ax = plt.subplots(figsize=(12, 8))
        self._draw_frame(ax, scene_index, min_connections, aggregate)
        plt.savefig(filename, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Successfully saved network scene {scene_index} to {filename}")



def draw_graph(G: nx.Graph|nx.DiGraph, pos=None, title: str ="Harry Potter Character Relationships", 
               labels: bool =True, weight: str ='sentiment', alpha_min: float =0.2, alpha_max: float =1.0, figsize: tuple[int]=(8,6),
               node_args: Optional[Dict]=None, edge_args: Optional[Dict]=None, label_args: Optional[Dict]=None) -> None:
    """
    Visualizes a character relationship network with sentiment-weighted edges.

    Args:
        G (nx.Graph): The NetworkX graph object to draw.
        pos (dict, optional): Dictionary of node positions. Defaults to spring_layout.
        title (str): The title displayed at the top of the plot.
        labels (bool): Whether to display node names. Defaults to True.
        weight (str): The edge attribute key used for sentiment coloring.
        alpha_min (float): Minimum transparency for weak relationships.
        alpha_max (float): Maximum transparency for strong relationships.
        node_args (dict, optional): Overrides for nx.draw_networkx_nodes.
        edge_args (dict, optional): Overrides for nx.draw_networkx_edges.
        label_args (dict, optional): Overrides for nx.draw_networkx_labels.

    Returns:
        None: Displays a Matplotlib plot.
    """

    # Initialize dictionaries
    node_args = node_args or {}
    edge_args = edge_args or {}
    label_args = label_args or {}

    pos = nx.spring_layout(G, seed=42, k=0.5) if pos is None else pos
    
    #Sentiment Calculation
    edges_data = G.edges(data=True)
    sentiment_list = [data.get(weight, 0) for u, v, data in edges_data] if edges_data else [0]
    
    cmap = LinearSegmentedColormap.from_list('sentiments', ['red', 'grey', 'green'], N=256)
    vmin, vmax = min(sentiment_list), max(sentiment_list)
    norm = TwoSlopeNorm(vmin=min(vmin, -0.1), vcenter=0, vmax=max(vmax, 0.1))
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    
    abs_sentiments = [abs(s) for s in sentiment_list]
    as_min, as_max = min(abs_sentiments), max(abs_sentiments)
    denom = (as_max - as_min) if as_max != as_min else 1
    
    #Define Default Style Dictionaries
    default_node = {
        'node_color': 'lightblue',
        'node_size': 1000
    }

    default_edge = {
        'edge_color': [mapper.to_rgba(s) for s in sentiment_list],
        'alpha': [(alpha_min + (alpha_max - alpha_min) * (abs_s - as_min) / denom) - 0.01 for abs_s in abs_sentiments],
        'connectionstyle': "arc3, rad=0.1",
        'width': 1.5,
        'arrows': True,
        'arrowsize': 15
    }

    default_label = {
        'font_size': 9,
        'font_weight': 'bold'
    }

    default_node.update(node_args)
    default_edge.update(edge_args)
    default_label.update(label_args)

    # --- 4. Drawing ---
    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, **default_node)
    nx.draw_networkx_edges(G, pos, **default_edge)

    if labels:
        nx.draw_networkx_labels(G, pos, **default_label)

    # Colorbar logic
    cbar = plt.colorbar(mapper, ax=plt.gca(), orientation='vertical', shrink=0.8)
    cbar.set_label('Sentiment Intensity', rotation=270, labelpad=15)
    
    plt.title(title, fontsize=16)
    plt.show()
