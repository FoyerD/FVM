from itertools import product
from typing import Callable, Dict, List, Set, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt

from TransitionSystem import TransitionSystem


Location = Union[str, Tuple]  # A state can be a string or a tuple (location, environment)
Action = str  # Actions are represented as strings
Condition = str # Conditions are represented as strings
Transition = Tuple[Location, Condition, Action, Location]  # (source_state, action, target_state)
Environment = Dict[str, Union[str, bool, int, float]]      # Variable assignments

class ProgramGraph:
    def __init__(
        self,
        locations: Set[Location],
        initial_locations: Set[Location],
        actions: Set[Action],
        transitions: Set[Transition],
        eval_fn: Callable[[Condition, Environment], bool],
        effect_fn: Callable[[Action, Environment], Environment],
        g0: Condition
    ):
        """
        A representation of a Program Graph.

        :param locations: Set of all program locations (Loc).
        :param initial_locations: Set of initial locations (Loc0).
        :param actions: Set of possible actions (Act).
        :param transitions: Set of transitions in the form (loc_from, condition, action, loc_to).
        :param eval_fn: Function to evaluate a condition string in an environment.
        :param effect_fn: Function to compute the new environment after applying an action.
        :param g0: Initial condition string for filtering valid starting environments.
        """
        self.Loc = set(locations)
        self.Loc0 = set(initial_locations)
        self.Act = set(actions)
        self.Transitions = set(transitions)
        self.eval_fn = eval_fn
        self.effect_fn = effect_fn
        self.g0 = g0

    def add_location(self, *locations: Location) -> "ProgramGraph":
        """Add one or more locations to the program graph."""
        self.Loc.update(locations)
        return self

    def add_action(self, *actions: Action) -> "ProgramGraph":
        """Add one or more actions to the program graph."""
        self.Act.update(actions)
        return self

    def add_transition(self, *transitions: Transition) -> "ProgramGraph":
        """
        Add one or more transitions to the program graph.

        Each transition must be a tuple: (loc_from, condition, action, loc_to).
        """
        for transition in transitions:
            if not isinstance(transition, tuple) or len(transition) != 4:
                raise ValueError(f"Invalid transition format: {transition}. Expected (loc_from, cond, action, loc_to).")
            loc_from, cond, action, loc_to = transition
            if loc_from not in self.Loc:
                raise ValueError(f"Location {loc_from} is not in the program graph.")
            if loc_to not in self.Loc:
                raise ValueError(f"Location {loc_to} is not in the program graph.")
            if action not in self.Act:
                raise ValueError(f"Action {action} is not in the program graph.")

            self.Transitions.add(transition)
        return self

    def add_initial_location(self, *locations: Location) -> "ProgramGraph":
        """Add one or more initial locations to the program graph."""
        for loc in locations:
            if loc not in self.Loc:
                raise ValueError(f"Cannot set initial location {loc}. Location is not in the set of locations.")
            self.Loc0.add(loc)
        return self

    def set_eval_fn(self, eval_fn: Callable[[Condition, Environment], bool]) -> "ProgramGraph":
        """Set the function used to evaluate conditions."""
        self.eval_fn = eval_fn
        return self

    def set_effect_fn(self, effect_fn: Callable[[Action, Environment], Environment]) -> "ProgramGraph":
        """Set the function used to apply actions to environments."""
        self.effect_fn = effect_fn
        return self

    def eval(self, condition: Condition, env: Environment) -> bool:
        """Evaluate a condition string in the given environment."""
        return self.eval_fn(condition, env)

    def effect(self, action: Action, env: Environment) -> Environment:
        """Apply an action to the environment and return the new environment."""
        return self.effect_fn(action, env)


    def valid_transitions(self, loc: Location, env: Environment, action: Action) -> List[Tuple[Location, Action, Location]]:
        """
        Return a list of valid transitions from a given location using the provided environment and action.
        """
        return [(loc_from, act, loc_to) for loc_from, cond, act, loc_to in self.Transitions
                if loc_from == loc and act == action and self.eval(cond, env)]

    def to_transition_system(self, vars: Dict[str, Set[Union[str, bool, int, float]]], labels: Set[Condition]) -> "TransitionSystem":
        """
        Construct and return a Transition System from the program graph.

        :param vars: A dictionary mapping variable names to their finite sets of possible values.
        :param labels: A set of atomic proposition strings to be used for labeling states.
        :return: A TransitionSystem instance corresponding to the program graph.
        """
        envs = generate_combinations(vars)

        states = {(loc,env) for loc in self.Loc for env in envs}
        actions = self.Act
        transitions = {
            ((loc_from, HashableDict(env)), act, (loc_to,HashableDict(self.effect(act, env)))) for env in envs for loc_from, cond, act, loc_to in self.Transitions
            if self.eval(cond, env)}
        initial_states = {(loc,env) for loc in self.Loc0 for env in envs 
                          if self.eval(self.g0, env)}
        atomic_props = self.Loc.union(labels).union({self.g0})
        labeling_map = {state:{label for label in labels if self.eval(label, state[1])}.union({state[0]}) for state in states}        

        ts_full = TransitionSystem(
            states=states,
            actions=actions,
            transitions=transitions,
            initial_states=initial_states,
            atomic_props=atomic_props,
            labeling_map=labeling_map
        )
        reachable_states = ts_full.reach()
        reachable_transitions = {trans for trans in ts_full.Transitions if trans[0] in reachable_states and trans[2] in reachable_states}
        reachable_initial_states = reachable_states.intersection(ts_full.I)
        rechable_labeling_map = {state: ts_full.L(state) for state in reachable_states}
        ts_reachable = TransitionSystem(
            states=reachable_states,
            actions=actions,
            transitions=reachable_transitions,
            initial_states=reachable_initial_states,
            atomic_props=atomic_props,
            labeling_map=rechable_labeling_map
        )
        return ts_reachable
        

    def __repr__(self):
                return (
            f"ProgramGraph(\n"
            f"  Loc: {self.Loc}\n"
            f"  Act: {self.Act}\n"
            f"  Transitions: {self.Transitions}\n"
            f"  Loc0: {self.Loc0}\n"
            f"  g0: {self.g0}\n"
            f")"
        )

    def plot(self):
        """
        Visualize the program graph as a directed graph using networkx and matplotlib.

        Nodes are locations. Edges are labeled with (condition, action).
        """
        G = nx.MultiDiGraph()

        # Add nodes
        for loc in self.Loc:
            G.add_node(loc, color='lightblue' if loc in self.Loc0 else 'white')

        # Add edges with (condition, action) labels
        for (loc_from, cond, action, loc_to) in self.Transitions:
            label = f"{cond} / {action}"
            G.add_edge(loc_from, loc_to, label=label)

        pos = nx.spring_layout(G, seed=42)  # consistent layout

        # Draw nodes with color
        node_colors = [G.nodes[n].get('color', 'white') for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black')
        nx.draw_networkx_labels(G, pos)

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)

        # Draw edge labels
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Display
        plt.title("Program Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()



def generate_combinations(input_dict: Dict[str, Set[Union[str, bool, int, float]]]) -> Set[Dict[str, Union[str, bool, int, float]]]:
    """
    Generate a set of dictionaries for each combination of string-to-value mappings.

    :param input_dict: A dictionary mapping strings to sets of possible values.
    :return: A set of dictionaries, each representing a unique combination of mappings.
    """
    keys = input_dict.keys()
    values = input_dict.values()
    combinations = product(*values)
    
    return {HashableDict(zip(keys, combination)) for combination in combinations}


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))