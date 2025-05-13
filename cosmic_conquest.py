"""
cosmic_conquest.py - 100% Implementation

Complete implementation of Cosmic Conquest:
- Players control star systems (nodes) on a dynamic network, spending resource tokens to move.
- Players can collect resources from controlled nodes each turn.
- Cosmic events regularly alter the connections between systems.
- Win by controlling a target number of star systems or accumulating enough resources.

Features:
- Full game mechanics with resource management
- Advanced AI using Minimax with Alpha-Beta pruning
- Dynamic board with cosmic events
- Multiple game modes and win conditions
- Complete UI with game state visualization
"""

import pygame
import sys
import random
import math
from enum import Enum

# -------------------------
# CONFIGURATION & CONSTANTS
# -------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Graph / Board
NODE_COUNT_OPTIONS = {"small": 8, "medium": 12, "large": 16}
NUM_NODES = NODE_COUNT_OPTIONS["medium"]  # Default size
NODE_TYPES = ["normal", "resource", "strategic", "wormhole"]
NODE_TYPE_WEIGHTS = [0.5, 0.2, 0.2, 0.1]  # Probability distribution for node types
INITIAL_RESOURCE_TOKENS = 20
INITIAL_EDGE_DENSITY = 0.4  # Percentage of possible edges that will be created

# Win condition parameters
TARGET_CONTROL_COUNT = 5  # Win if a player controls >= TARGET_CONTROL_COUNT nodes
TARGET_RESOURCE_COUNT = 50  # Alternative win condition: accumulate resources

# Cosmic event parameters
EVENT_PROBABILITY = 0.3  # Chance to trigger an event after a move
EVENT_TYPES = ["toggle_edge", "bonus_resources", "warp_storm", "node_upgrade"]
EVENT_WEIGHTS = [0.5, 0.2, 0.2, 0.1]  # Probability distribution for event types

# AI parameters
MAX_DEPTH = 3  # Maximum depth for Minimax search
RESOURCE_WEIGHT = 1.0  # Weight for resource value in heuristic
CONTROL_WEIGHT = 3.0  # Weight for controlled nodes in heuristic
POSITION_WEIGHT = 2.0  # Weight for strategic position in heuristic

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
DARKGRAY = (80, 80, 80)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
TEAL = (0, 128, 128)

# UI parameters
NODE_RADIUS = 25
FONT_SMALL = 20
FONT_MEDIUM = 24
FONT_LARGE = 36
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 50

# Game modes
class GameMode(Enum):
    CONQUEST = 1  # Win by controlling target number of nodes
    RESOURCE = 2  # Win by accumulating target number of resources
    ELIMINATION = 3  # Win by eliminating the opponent's ability to move

# Node types
class NodeType(Enum):
    NORMAL = 1  # Standard node
    RESOURCE = 2  # Gives extra resources each turn when controlled
    STRATEGIC = 3  # Gives strategic advantage (cheaper movement)
    WORMHOLE = 4  # Allows long-distance movement


# -------------------------
# DATA STRUCTURES
# -------------------------
class Node:

    def __init__(self, node_id, x, y, node_type=NodeType.NORMAL):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.controlled_by = None  # Can be None, 'Player', or 'AI'
        self.node_type = node_type
        self.resource_value = self._get_resource_value()
        self.name = self._generate_name()

    def _get_resource_value(self):
        if self.node_type == NodeType.RESOURCE:
            return random.randint(2, 4)
        elif self.node_type == NodeType.STRATEGIC:
            return random.randint(1, 2)
        elif self.node_type == NodeType.WORMHOLE:
            return 0
        else:  # NORMAL
            return 1

    def _generate_name(self):
        prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
                    "Iota", "Kappa", "Lambda", "Omega", "Sigma", "Tau"]
        suffixes = ["Prime", "Major", "Minor", "IV", "V", "VI", "VII", "Centauri",
                    "Proxima", "Nova", "Nebula", "Cluster"]
        return f"{random.choice(prefixes)} {random.choice(suffixes)}"

    def contains_point(self, pos):
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        return math.hypot(dx, dy) <= NODE_RADIUS


class Edge:

    def __init__(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b
        self.active = True
        self.travel_cost = self._get_travel_cost()

    def _get_travel_cost(self):
        # Strategic nodes make travel cheaper
        if (self.node_a.node_type == NodeType.STRATEGIC or
            self.node_b.node_type == NodeType.STRATEGIC):
            return 0.7  # 30% discount
        # Wormhole nodes may have variable costs
        elif (self.node_a.node_type == NodeType.WORMHOLE or
              self.node_b.node_type == NodeType.WORMHOLE):
            return random.uniform(0.5, 1.5)  # Variable cost
        else:
            return 1.0  # Standard cost


class GameState:

    def __init__(self, nodes, edges, game_mode=GameMode.CONQUEST):
        self.nodes = nodes
        self.edges = edges
        self.player_position = 0  # Human starts at node 0
        self.ai_position = len(nodes) - 1  # AI starts at last node
        self.player_resources = INITIAL_RESOURCE_TOKENS
        self.ai_resources = INITIAL_RESOURCE_TOKENS
        self.current_player = 'Player'  # 'Player' or 'AI'
        self.game_mode = game_mode
        self.turn_count = 1
        self.last_dice_roll = None
        self.game_log = ["Game started"]
        self.last_event = None

        # Initialize starting positions
        self.control_node(self.player_position, 'Player')
        self.control_node(self.ai_position, 'AI')

    def get_active_neighbors(self, node_id):
        neighbors = []
        for e in self.edges:
            if e.active:
                if e.node_a.node_id == node_id:
                    neighbors.append((e.node_b.node_id, e.travel_cost))
                elif e.node_b.node_id == node_id:
                    neighbors.append((e.node_a.node_id, e.travel_cost))
        return neighbors

    def switch_player(self):
        # Collect resources from controlled nodes
        self.collect_resources()
        self.current_player = 'AI' if self.current_player == 'Player' else 'Player'
        if self.current_player == 'Player':
            self.turn_count += 1

    def control_node(self, node_id, controller):
        self.nodes[node_id].controlled_by = controller

    def collect_resources(self):
        player_resources = 0
        ai_resources = 0

        for node in self.nodes:
            if node.controlled_by == 'Player':
                player_resources += node.resource_value
            elif node.controlled_by == 'AI':
                ai_resources += node.resource_value

        self.player_resources += player_resources
        self.ai_resources += ai_resources

        # Log resource collection
        if player_resources > 0:
            self.game_log.append(f"Player collected {player_resources} resources")
        if ai_resources > 0:
            self.game_log.append(f"AI collected {ai_resources} resources")

    def count_controlled(self, controller):
        return sum(1 for n in self.nodes if n.controlled_by == controller)

    def get_winner(self):
        if self.game_mode == GameMode.CONQUEST:
            if self.count_controlled('Player') >= TARGET_CONTROL_COUNT:
                return 'Player'
            elif self.count_controlled('AI') >= TARGET_CONTROL_COUNT:
                return 'AI'
        elif self.game_mode == GameMode.RESOURCE:
            if self.player_resources >= TARGET_RESOURCE_COUNT:
                return 'Player'
            elif self.ai_resources >= TARGET_RESOURCE_COUNT:
                return 'AI'
        elif self.game_mode == GameMode.ELIMINATION:
            # Check if either player has no legal moves
            player_moves = get_legal_moves(self, 'Player')[0]
            ai_moves = get_legal_moves(self, 'AI')[0]
            if not player_moves and self.current_player == 'Player':
                return 'AI'
            elif not ai_moves and self.current_player == 'AI':
                return 'Player'
        return None

    def clone(self):
        # Create shallow copies of nodes and edges to maintain references
        new_nodes = [Node(n.node_id, n.x, n.y, n.node_type) for n in self.nodes]
        for i, n in enumerate(self.nodes):
            new_nodes[i].controlled_by = n.controlled_by
            new_nodes[i].resource_value = n.resource_value
            new_nodes[i].name = n.name

        new_edges = []
        for e in self.edges:
            node_a = new_nodes[e.node_a.node_id]
            node_b = new_nodes[e.node_b.node_id]
            new_edge = Edge(node_a, node_b)
            new_edge.active = e.active
            new_edge.travel_cost = e.travel_cost
            new_edges.append(new_edge)

        # Create new game state with copied objects
        new_state = GameState(new_nodes, new_edges, self.game_mode)
        new_state.player_position = self.player_position
        new_state.ai_position = self.ai_position
        new_state.player_resources = self.player_resources
        new_state.ai_resources = self.ai_resources
        new_state.current_player = self.current_player
        new_state.turn_count = self.turn_count

        return new_state


# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def create_initial_graph(num_nodes=NUM_NODES):
    nodes = []
    radius = 250
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    # Create nodes in a circle
    for i in range(num_nodes):
        angle = (2 * math.pi / num_nodes) * i
        x = center_x + int(radius * math.cos(angle))
        y = center_y + int(radius * math.sin(angle))

        # Determine node type based on weighted random choice
        node_type = random.choices(
            [NodeType.NORMAL, NodeType.RESOURCE, NodeType.STRATEGIC, NodeType.WORMHOLE],
            weights=NODE_TYPE_WEIGHTS
        )[0]

        # Ensure starting positions are normal nodes
        if i == 0 or i == num_nodes - 1:
            node_type = NodeType.NORMAL

        node = Node(i, x, y, node_type)
        nodes.append(node)

    edges = []
    # Connect all nodes in a ring first (to ensure the graph is connected)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        edges.append(Edge(nodes[i], nodes[j]))

    # Add additional edges based on desired density
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2
    target_edge_count = int(max_possible_edges * INITIAL_EDGE_DENSITY)
    current_edge_count = num_nodes  # We already have the ring connections

    # Add more edges until we reach the target density
    while current_edge_count < target_edge_count:
        # Pick two random nodes
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)

        # Make sure they are different nodes
        if i == j:
            continue

        # Check if edge already exists
        edge_exists = False
        for e in edges:
            if ((e.node_a.node_id == i and e.node_b.node_id == j) or
                (e.node_a.node_id == j and e.node_b.node_id == i)):
                edge_exists = True
                break

        if not edge_exists:
            edges.append(Edge(nodes[i], nodes[j]))
            current_edge_count += 1

    return nodes, edges


def roll_dice():
    return random.randint(1, 6)


def cosmic_event_randomizer(state: GameState):
    # Choose event type based on weighted probabilities
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]

    if event_type == "toggle_edge":
        # Toggle a random edge's active status
        if random.random() < 0.5 and state.edges:
            edge = random.choice(state.edges)
            edge.active = not edge.active
            status = "opened" if edge.active else "closed"
            event_message = f"Cosmic event: Connection between {edge.node_a.node_id} and {edge.node_b.node_id} {status}"
            state.last_event = event_message
            state.game_log.append(event_message)

    elif event_type == "bonus_resources":
        # Give bonus resources to a random player
        if random.random() < 0.5:
            bonus = random.randint(1, 5)
            state.player_resources += bonus
            event_message = f"Cosmic event: Player received {bonus} bonus resources"
        else:
            bonus = random.randint(1, 5)
            state.ai_resources += bonus
            event_message = f"Cosmic event: AI received {bonus} bonus resources"
        state.last_event = event_message
        state.game_log.append(event_message)

    elif event_type == "warp_storm":
        # Make a random node temporarily inaccessible
        nodes_with_multiple_edges = []
        for i, node in enumerate(state.nodes):
            connections = sum(1 for e in state.edges if
                             (e.node_a.node_id == i or e.node_b.node_id == i) and e.active)
            if connections > 1:
                nodes_with_multiple_edges.append(i)

        if nodes_with_multiple_edges:
            affected_node = random.choice(nodes_with_multiple_edges)
            # Find edges connected to this node
            connected_edges = [e for e in state.edges if
                              (e.node_a.node_id == affected_node or
                               e.node_b.node_id == affected_node) and e.active]

            # Temporarily deactivate some connections
            num_to_deactivate = random.randint(1, min(len(connected_edges), 2))
            deactivated = random.sample(connected_edges, num_to_deactivate)

            for edge in deactivated:
                edge.active = False

            event_message = f"Cosmic event: Warp storm affects node {affected_node}, blocking {num_to_deactivate} connection(s)"
            state.last_event = event_message
            state.game_log.append(event_message)

    elif event_type == "node_upgrade":
        # Upgrade a random uncontrolled node
        uncontrolled_nodes = [n for n in state.nodes if n.controlled_by is None]
        if uncontrolled_nodes:
            node = random.choice(uncontrolled_nodes)
            if node.node_type == NodeType.NORMAL:
                # Upgrade to a more valuable type
                node.node_type = random.choice([NodeType.RESOURCE, NodeType.STRATEGIC])
                node.resource_value = node._get_resource_value()  # Recalculate resource value
                event_message = f"Cosmic event: Node {node.node_id} upgraded to {node.node_type.name}"
                state.last_event = event_message
                state.game_log.append(event_message)

    return state.last_event


def get_legal_moves(state: GameState, player=None):
    if player is None:
        player = state.current_player

    dice_result = roll_dice() if state.last_dice_roll is None else state.last_dice_roll

    current_position = state.player_position if player == 'Player' else state.ai_position
    neighbors_with_cost = state.get_active_neighbors(current_position)

    moves = []
    for neighbor_id, edge_cost_modifier in neighbors_with_cost:
        # Calculate movement cost based on dice roll and edge cost modifier
        base_cost = 1
        cost = int(base_cost * edge_cost_modifier * (1 + dice_result/10))
        cost = max(1, cost)  # Ensure minimum cost of 1

        # Special case for wormhole destinations
        if state.nodes[neighbor_id].node_type == NodeType.WORMHOLE:
            # Find all other active wormholes
            wormholes = [n.node_id for n in state.nodes
                        if n.node_type == NodeType.WORMHOLE
                        and n.node_id != neighbor_id]

            # Add wormhole destinations with a premium cost
            for wormhole_id in wormholes:
                wormhole_cost = cost + 2  # Premium for wormhole travel
                if player == 'Player' and state.player_resources >= wormhole_cost:
                    moves.append((wormhole_id, wormhole_cost))
                elif player == 'AI' and state.ai_resources >= wormhole_cost:
                    moves.append((wormhole_id, wormhole_cost))

        # Regular move
        if player == 'Player' and state.player_resources >= cost:
            moves.append((neighbor_id, cost))
        elif player == 'AI' and state.ai_resources >= cost:
            moves.append((neighbor_id, cost))

    return moves, dice_result


def apply_move(state: GameState, move):
    destination, cost = move

    # Log the move
    if state.current_player == 'Player':
        state.player_position = destination
        state.player_resources -= cost
        state.control_node(destination, 'Player')
        state.game_log.append(f"Player moved to {state.nodes[destination].name} (Node {destination}) for {cost} resources")
    else:
        state.ai_position = destination
        state.ai_resources -= cost
        state.control_node(destination, 'AI')
        state.game_log.append(f"AI moved to {state.nodes[destination].name} (Node {destination}) for {cost} resources")


def is_terminal(state: GameState):
    return state.get_winner() is not None


# -------------------------
# AI IMPLEMENTATION USING MINIMAX WITH ALPHA-BETA PRUNING
# -------------------------
def evaluate_state(state: GameState):
    # Count controlled nodes (weighted by type)
    player_control_value = 0
    ai_control_value = 0

    for node in state.nodes:
        node_value = 1
        # Strategic nodes are more valuable
        if node.node_type == NodeType.STRATEGIC:
            node_value = 2
        # Resource nodes are even more valuable
        elif node.node_type == NodeType.RESOURCE:
            node_value = node.resource_value + 1

        if node.controlled_by == 'Player':
            player_control_value += node_value
        elif node.controlled_by == 'AI':
            ai_control_value += node_value

    # Evaluate connectivity and position
    player_position_value = len(state.get_active_neighbors(state.player_position))
    ai_position_value = len(state.get_active_neighbors(state.ai_position))

    # Combine factors with weights
    player_score = (RESOURCE_WEIGHT * state.player_resources +
                   CONTROL_WEIGHT * player_control_value +
                   POSITION_WEIGHT * player_position_value)

    ai_score = (RESOURCE_WEIGHT * state.ai_resources +
               CONTROL_WEIGHT * ai_control_value +
               POSITION_WEIGHT * ai_position_value)

    # Return difference from AI perspective
    return ai_score - player_score


def minimax(state: GameState, depth, alpha, beta, maximizing_player):
    # Check terminal state or depth limit
    if depth == 0 or is_terminal(state):
        return None, evaluate_state(state)

    # Get current player
    current_player = 'AI' if maximizing_player else 'Player'

    # Get legal moves for current player
    legal_moves, _ = get_legal_moves(state, current_player)

    if not legal_moves:
        # No legal moves, return current state evaluation
        return None, evaluate_state(state)

    best_move = None

    if maximizing_player:  # AI's turn
        max_eval = float('-inf')
        for move in legal_moves:
            # Apply move to a cloned state
            new_state = state.clone()
            new_state.current_player = current_player
            apply_move(new_state, move)
            new_state.switch_player()

            # Recursive call
            _, eval_score = minimax(new_state, depth - 1, alpha, beta, False)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break

        return best_move, max_eval
    else:  # Player's turn
        min_eval = float('inf')
        for move in legal_moves:
            # Apply move to a cloned state
            new_state = state.clone()
            new_state.current_player = current_player
            apply_move(new_state, move)
            new_state.switch_player()

            # Recursive call
            _, eval_score = minimax(new_state, depth - 1, alpha, beta, True)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break

        return best_move, min_eval


def choose_ai_move(state: GameState):
    legal_moves, dice_roll = get_legal_moves(state, 'AI')
    state.last_dice_roll = dice_roll

    if not legal_moves:
        return None

    # For random moves (20% chance), choose randomly
    if random.random() < 0.2:
        return random.choice(legal_moves)

    # Use Minimax for strategic decision
    best_move, _ = minimax(state, MAX_DEPTH, float('-inf'), float('inf'), True)

    return best_move


# -------------------------
# PYGAME UI FUNCTIONS
# -------------------------
def draw_node(screen, node, highlight=False, pulse=False):
    # Determine node color based on type
    if node.node_type == NodeType.NORMAL:
        type_color = WHITE
    elif node.node_type == NodeType.RESOURCE:
        type_color = YELLOW
    elif node.node_type == NodeType.STRATEGIC:
        type_color = CYAN
    elif node.node_type == NodeType.WORMHOLE:
        type_color = PURPLE

    # Apply control color overlay
    if node.controlled_by == 'Player':
        fill_color = blend_colors(type_color, BLUE, 0.7)
    elif node.controlled_by == 'AI':
        fill_color = blend_colors(type_color, RED, 0.7)
    else:
        fill_color = type_color

    # Highlight effect for valid moves
    if highlight:
        # Pulsating highlight effect
        if pulse:
            pulse_factor = 0.7 + 0.3 * (math.sin(pygame.time.get_ticks() / 200) + 1) / 2
            outline_color = brighten_color(YELLOW, pulse_factor)
        else:
            outline_color = YELLOW
        outline_width = 4
    else:
        outline_color = BLACK
        outline_width = 2

    # Draw the node
    pygame.draw.circle(screen, fill_color, (node.x, node.y), NODE_RADIUS)
    pygame.draw.circle(screen, outline_color, (node.x, node.y), NODE_RADIUS, outline_width)

    # Draw node ID and resource value
    font = pygame.font.SysFont(None, FONT_SMALL)
    id_text = font.render(str(node.node_id), True, BLACK)
    screen.blit(id_text, (node.x - 8, node.y - 8))

    # Show resource value if it's a resource node or controlled
    if node.node_type == NodeType.RESOURCE or node.controlled_by:
        res_text = font.render(f"+{node.resource_value}", True, BLACK)
        screen.blit(res_text, (node.x - 8, node.y + 8))


def blend_colors(color1, color2, factor=0.5):
    r = int(color1[0] * (1-factor) + color2[0] * factor)
    g = int(color1[1] * (1-factor) + color2[1] * factor)
    b = int(color1[2] * (1-factor) + color2[2] * factor)
    return (r, g, b)


def brighten_color(color, factor=1.2):
    r = min(255, int(color[0] * factor))
    g = min(255, int(color[1] * factor))
    b = min(255, int(color[2] * factor))
    return (r, g, b)


def draw_edge(screen, edge):
    if not edge.active:
        # Draw inactive edge as dashed line
        draw_dashed_line(screen, (edge.node_a.x, edge.node_a.y),
                          (edge.node_b.x, edge.node_b.y), DARKGRAY, 2, 5)
    else:
        # Draw active edge
        cost_factor = edge.travel_cost

        # Color based on travel cost
        if cost_factor < 0.8:  # Cheap
            color = GREEN
        elif cost_factor > 1.2:  # Expensive
            color = ORANGE
        else:  # Normal
            color = GRAY

        pygame.draw.line(screen, color,
                         (edge.node_a.x, edge.node_a.y),
                         (edge.node_b.x, edge.node_b.y), 3)

        # Show cost on edge
        midx = (edge.node_a.x + edge.node_b.x) / 2
        midy = (edge.node_a.y + edge.node_b.y) / 2

        font = pygame.font.SysFont(None, FONT_SMALL)
        cost_text = font.render(f"{edge.travel_cost:.1f}x", True, WHITE)
        screen.blit(cost_text, (midx - 15, midy - 10))


def draw_dashed_line(screen, point1, point2, color, width=1, dash_length=10):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate line length and angle
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)

    # Calculate unit vector components
    unit_x = math.cos(angle)
    unit_y = math.sin(angle)

    # Draw the dashed line
    dash_gap = dash_length
    current_length = 0

    while current_length < length:
        # Calculate start and end points of current dash segment
        start_x = x1 + unit_x * current_length
        start_y = y1 + unit_y * current_length

        # Ensure we don't draw past the end of the line
        segment_length = min(dash_length, length - current_length)
        end_x = start_x + unit_x * segment_length
        end_y = start_y + unit_y * segment_length

        # Draw the segment
        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), width)

        # Move to position for next segment
        current_length += dash_length + dash_gap


def draw_player_marker(screen, node, is_player):
    marker_size = 15

    if is_player:
        color = BLUE
        # Triangle above the node
        pygame.draw.polygon(screen, color,
                           [(node.x, node.y - NODE_RADIUS - marker_size),
                            (node.x - marker_size, node.y - NODE_RADIUS),
                            (node.x + marker_size, node.y - NODE_RADIUS)])
    else:
        color = RED
        # Triangle below the node
        pygame.draw.polygon(screen, color,
                           [(node.x, node.y + NODE_RADIUS + marker_size),
                            (node.x - marker_size, node.y + NODE_RADIUS),
                            (node.x + marker_size, node.y + NODE_RADIUS)])


def draw_board(screen, state, legal_move_ids=None):
    if legal_move_ids is None:
        legal_move_ids = set()

    pulse_effect = True  # Enable pulsating highlight effect

    # Clear screen with space background
    screen.fill(BLACK)

    # Draw background stars
    for _ in range(100):
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT)
        brightness = random.randint(100, 255)
        size = random.randint(1, 3)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), size)

    # Draw edges first (so they appear behind nodes)
    for e in state.edges:
        draw_edge(screen, e)

    # Draw nodes
    for n in state.nodes:
        is_legal_move = n.node_id in legal_move_ids
        draw_node(screen, n, is_legal_move, pulse_effect)

    # Draw player and AI position markers
    player_node = state.nodes[state.player_position]
    ai_node = state.nodes[state.ai_position]
    draw_player_marker(screen, player_node, True)
    draw_player_marker(screen, ai_node, False)

    # Draw node info on hover
    mouse_pos = pygame.mouse.get_pos()
    for n in state.nodes:
        if n.contains_point(mouse_pos):
            draw_node_info(screen, n, mouse_pos)

    # Draw game info panel
    draw_game_info_panel(screen, state)

    # Draw game log
    draw_game_log(screen, state)

    # Draw win condition information
    draw_win_condition(screen, state)


def draw_node_info(screen, node, mouse_pos):
    # Prepare text lines
    info_lines = [
        f"System: {node.name}",
        f"Type: {node.node_type.name}",
        f"Resource: +{node.resource_value}/turn",
        f"Control: {node.controlled_by if node.controlled_by else 'None'}"
    ]

    # Calculate tooltip dimensions
    font = pygame.font.SysFont(None, FONT_SMALL)
    line_height = font.get_height()
    max_width = 0
    for line in info_lines:
        width = font.size(line)[0]
        max_width = max(max_width, width)

    # Calculate tooltip position
    tooltip_width = max_width + 20
    tooltip_height = len(info_lines) * line_height + 10
    tooltip_x = mouse_pos[0] + 15
    tooltip_y = mouse_pos[1] + 15

    # Ensure tooltip stays within screen boundaries
    if tooltip_x + tooltip_width > SCREEN_WIDTH:
        tooltip_x = mouse_pos[0] - tooltip_width - 15
    if tooltip_y + tooltip_height > SCREEN_HEIGHT:
        tooltip_y = mouse_pos[1] - tooltip_height - 15

    # Draw tooltip background
    pygame.draw.rect(screen, (30, 30, 30),
                     (tooltip_x, tooltip_y, tooltip_width, tooltip_height))
    pygame.draw.rect(screen, WHITE,
                     (tooltip_x, tooltip_y, tooltip_width, tooltip_height), 1)

    # Draw text
    y_offset = tooltip_y + 5
    for line in info_lines:
        text = font.render(line, True, WHITE)
        screen.blit(text, (tooltip_x + 10, y_offset))
        y_offset += line_height


def draw_game_info_panel(screen, state):
    panel_width = 280
    panel_height = 200
    panel_x = SCREEN_WIDTH - panel_width - 10
    panel_y = 10

    # Draw panel background
    pygame.draw.rect(screen, (20, 20, 40, 200),
                     (panel_x, panel_y, panel_width, panel_height))
    pygame.draw.rect(screen, GRAY,
                     (panel_x, panel_y, panel_width, panel_height), 1)

    # Draw game information
    font = pygame.font.SysFont(None, FONT_MEDIUM)

    # Game state info
    info_lines = [
        f"Turn: {state.turn_count}",
        f"Current Player: {state.current_player}",
        f"Game Mode: {state.game_mode.name}",
        "",
        f"Player Resources: {state.player_resources}",
        f"AI Resources: {state.ai_resources}",
        "",
        f"Player Controls: {state.count_controlled('Player')}",
        f"AI Controls: {state.count_controlled('AI')}"
    ]

    # Last dice roll if available
    if state.last_dice_roll:
        info_lines.append("")
        info_lines.append(f"Last Dice Roll: {state.last_dice_roll}")

    # Display info lines
    y_offset = panel_y + 10
    for line in info_lines:
        text = font.render(line, True, WHITE)
        screen.blit(text, (panel_x + 10, y_offset))
        y_offset += font.get_height()


def draw_game_log(screen, state):
    log_width = 280
    log_height = 150
    log_x = SCREEN_WIDTH - log_width - 10
    log_y = 220

    # Draw log background
    pygame.draw.rect(screen, (20, 20, 40, 200),
                     (log_x, log_y, log_width, log_height))
    pygame.draw.rect(screen, GRAY,
                     (log_x, log_y, log_width, log_height), 1)

    # Draw header
    font_header = pygame.font.SysFont(None, FONT_MEDIUM)
    header_text = font_header.render("Event Log", True, WHITE)
    screen.blit(header_text, (log_x + 10, log_y + 5))

    # Draw log entries (show last 5 entries)
    font = pygame.font.SysFont(None, FONT_SMALL)
    line_height = font.get_height()

    # Get last 5 log entries
    log_entries = state.game_log[-5:] if len(state.game_log) > 5 else state.game_log

    y_offset = log_y + 30
    for entry in log_entries:
        # Wrap long text
        words = entry.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if font.size(test_line)[0] > log_width - 20:
                text = font.render(line, True, WHITE)
                screen.blit(text, (log_x + 10, y_offset))
                y_offset += line_height
                line = word + " "
            else:
                line = test_line

        if line:
            text = font.render(line, True, WHITE)
            screen.blit(text, (log_x + 10, y_offset))
            y_offset += line_height


def draw_win_condition(screen, state):
    panel_width = 280
    panel_height = 80
    panel_x = SCREEN_WIDTH - panel_width - 10
    panel_y = 380

    # Draw panel background
    pygame.draw.rect(screen, (20, 20, 40, 200),
                     (panel_x, panel_y, panel_width, panel_height))
    pygame.draw.rect(screen, GRAY,
                     (panel_x, panel_y, panel_width, panel_height), 1)

    # Draw win condition information based on game mode
    font = pygame.font.SysFont(None, FONT_MEDIUM)
    header_text = font.render("Win Condition:", True, WHITE)
    screen.blit(header_text, (panel_x + 10, panel_y + 10))

    condition_text = ""
    if state.game_mode == GameMode.CONQUEST:
        condition_text = f"Control {TARGET_CONTROL_COUNT} systems"
    elif state.game_mode == GameMode.RESOURCE:
        condition_text = f"Accumulate {TARGET_RESOURCE_COUNT} resources"
    elif state.game_mode == GameMode.ELIMINATION:
        condition_text = "Eliminate opponent's movement"

    condition_rendered = font.render(condition_text, True, YELLOW)
    screen.blit(condition_rendered, (panel_x + 10, panel_y + 40))


def draw_button(screen, text, x, y, width, height, inactive_color, active_color):
    mouse_pos = pygame.mouse.get_pos()
    mouse_clicked = pygame.mouse.get_pressed()[0]

    rect = pygame.Rect(x, y, width, height)
    hover = rect.collidepoint(mouse_pos)

    # Draw button
    color = active_color if hover else inactive_color
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, WHITE, rect, 2)

    # Draw text
    font = pygame.font.SysFont(None, FONT_MEDIUM)
    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

    # Return True if clicked
    return hover and mouse_clicked


def draw_menu(screen):
    # Fill background with space theme
    screen.fill(BLACK)

    # Draw stars
    for _ in range(200):
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT)
        brightness = random.randint(100, 255)
        size = random.randint(1, 3)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), size)

    # Draw title
    font_title = pygame.font.SysFont(None, 80)
    title_text = font_title.render("COSMIC CONQUEST", True, CYAN)
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, 120))
    screen.blit(title_text, title_rect)

    # Draw subtitle
    font_subtitle = pygame.font.SysFont(None, FONT_MEDIUM)
    subtitle_text = font_subtitle.render("Interstellar Strategy Board Game", True, WHITE)
    subtitle_rect = subtitle_text.get_rect(center=(SCREEN_WIDTH//2, 170))
    screen.blit(subtitle_text, subtitle_rect)

    # Draw buttons
    button_width = BUTTON_WIDTH
    button_height = BUTTON_HEIGHT
    button_x = SCREEN_WIDTH//2 - button_width//2

    # Game mode buttons
    mode_buttons = [
        ("Conquest Mode", GameMode.CONQUEST, 240),
        ("Resource Mode", GameMode.RESOURCE, 300),
        ("Elimination Mode", GameMode.ELIMINATION, 360)
    ]

    clicked_mode = None
    for text, mode, y in mode_buttons:
        if draw_button(screen, text, button_x, y, button_width, button_height, (50, 50, 120), (80, 80, 160)):
            clicked_mode = mode

    # Board size buttons
    size_buttons = [
        ("Small Board", "small", 440),
        ("Medium Board", "medium", 500),
        ("Large Board", "large", 560)
    ]

    clicked_size = None
    for text, size, y in size_buttons:
        if draw_button(screen, text, button_x, y, button_width, button_height, (50, 120, 50), (80, 160, 80)):
            clicked_size = size

    # Draw start button
    start_button_y = 620
    start_clicked = draw_button(screen, "START GAME", button_x, start_button_y, button_width, button_height, (120, 50, 50), (160, 80, 80))

    return clicked_mode, clicked_size, start_clicked


def highlight_legal_moves(state: GameState):
    legal_moves, dice_roll = get_legal_moves(state)
    state.last_dice_roll = dice_roll
    legal_move_ids = {move[0] for move in legal_moves}
    return legal_moves, dice_roll, legal_move_ids


def draw_game_over(screen, winner):
    # Transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # Semi-transparent black
    screen.blit(overlay, (0, 0))

    # Draw winner announcement
    font_title = pygame.font.SysFont(None, FONT_LARGE * 2)
    font_subtitle = pygame.font.SysFont(None, FONT_LARGE)

    # Title text
    title_text = font_title.render("GAME OVER", True, WHITE)
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
    screen.blit(title_text, title_rect)

    # Winner text with appropriate color
    winner_color = BLUE if winner == 'Player' else RED
    winner_text = font_subtitle.render(f"{winner} Wins!", True, winner_color)
    winner_rect = winner_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20))
    screen.blit(winner_text, winner_rect)

    # Play again button
    button_width = BUTTON_WIDTH
    button_height = BUTTON_HEIGHT
    button_x = SCREEN_WIDTH//2 - button_width//2
    button_y = SCREEN_HEIGHT//2 + 100

    play_again = draw_button(screen, "Play Again", button_x, button_y, button_width, button_height, (50, 120, 50), (80, 160, 80))

    # Exit button
    exit_button_y = button_y + button_height + 20
    exit_game = draw_button(screen, "Exit Game", button_x, exit_button_y, button_width, button_height, (120, 50, 50), (160, 80, 80))

    return play_again, exit_game


# -------------------------
# MAIN GAME LOOP
# -------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Cosmic Conquest")
    clock = pygame.time.Clock()

    # Game states
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2

    current_state = MENU
    game_state = None

    # Menu selections
    selected_mode = GameMode.CONQUEST
    selected_size = "medium"

    # For player's turn, store legal moves (destination and cost)
    player_legal_moves = []
    legal_move_ids = set()
    waiting_for_click = False  # Flag to wait for player's selection

    winner = None

    running = True
    while running:
        clock.tick(FPS)

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Process mouse clicks for player's turn
            if event.type == pygame.MOUSEBUTTONDOWN and current_state == PLAYING:
                if game_state.current_player == 'Player' and waiting_for_click:
                    pos = pygame.mouse.get_pos()
                    for node in game_state.nodes:
                        if node.contains_point(pos) and node.node_id in legal_move_ids:
                            # Find the corresponding move (node_id, cost)
                            selected_move = next((move for move in player_legal_moves if move[0] == node.node_id), None)
                            if selected_move:
                                # Valid move selected; apply move
                                apply_move(game_state, selected_move)

                                # Cosmic event chance
                                if random.random() < EVENT_PROBABILITY:
                                    cosmic_event_randomizer(game_state)

                                waiting_for_click = False
                                game_state.switch_player()
                                break

        # Main menu state
        if current_state == MENU:
            clicked_mode, clicked_size, start_clicked = draw_menu(screen)

            # Update selections
            if clicked_mode:
                selected_mode = clicked_mode
            if clicked_size:
                selected_size = clicked_size

            # Start new game if Start button clicked
            if start_clicked:
                num_nodes = NODE_COUNT_OPTIONS[selected_size]
                nodes, edges = create_initial_graph(num_nodes)
                game_state = GameState(nodes, edges, selected_mode)
                current_state = PLAYING

        # Game playing state
        elif current_state == PLAYING:
            # Check for terminal state
            winner = game_state.get_winner()
            if winner:
                current_state = GAME_OVER

            # Process player turn
            if game_state.current_player == 'Player' and not waiting_for_click:
                player_legal_moves, dice_roll, legal_move_ids = highlight_legal_moves(game_state)
                if player_legal_moves:
                    waiting_for_click = True
                    game_state.game_log.append(f"Player rolled {dice_roll}. Select a move.")
                else:
                    game_state.game_log.append("Player has no valid moves! Turn skipped.")
                    game_state.switch_player()

            # Process AI turn
            elif game_state.current_player == 'AI':
                ai_move = choose_ai_move(game_state)
                if ai_move:
                    # Short delay to simulate thinking
                    pygame.time.delay(500)
                    apply_move(game_state, ai_move)

                    # Cosmic event chance
                    if random.random() < EVENT_PROBABILITY:
                        cosmic_event_randomizer(game_state)
                else:
                    game_state.game_log.append("AI has no valid moves! Turn skipped.")

                game_state.switch_player()

            # Draw game board
            draw_board(screen, game_state, legal_move_ids if game_state.current_player == 'Player' else set())

        # Game over state
        elif current_state == GAME_OVER:
            # Draw board in background
            draw_board(screen, game_state)

            # Draw game over screen
            play_again, exit_game = draw_game_over(screen, winner)

            if play_again:
                current_state = MENU
            elif exit_game:
                running = False

        # Update display
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
