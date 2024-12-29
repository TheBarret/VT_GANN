from dataclasses import dataclass
import json
import uuid
import pygame
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

from toolkit import Network, Graphics, softmax

@dataclass
class MemoryRecord:
    rocket_id: str
    state: np.ndarray
    timestamp: int
    influence_score: float

@dataclass
class Config:
    WINDOW_SIZE         = (1024, 600)
    FPS                 = 30
    POPULATION_SIZE     = 25
    MUTATION_RATE       = 0.30
    MUTATION_STRENGTH   = 0.40
    GRAVITY             = 0.30
    DRAG                = 0.07
    GENERATION_TIMEOUT  = FPS * 15
    
    # Rocket config
    SENSOR_RANGE        = 50   # max sensor
    SENSOR_RANGE_H      = SENSOR_RANGE // 2 # half range
    MAX_VELOCITY        = 10.0 # max speed
    MAX_THRUST          = 7.0  # max thrust
    NOZZLE_MAX          = 30.0 # max angle
    NOZZLE_RSPD         = 0.30 # max angle rotation speed
    MAX_FUEL            = 1024 # max fuel
    WEIGHT              = 10.0 # dry weight
    WEIGHT_RATIO        = 0.8  # fuel weight ratio
    
    NETWORK_ARCH        = (12, 10, 3) # network architecture
   
    COLORS = {
        'WHITE'     : (255, 255, 255),
        'RED'       : (255, 0, 0),
        'YELLOW'    : (255, 255, 0),
        'BLACK'     : (0, 0, 0)
    }
    TARGET_POS = np.array([WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 3], dtype=float)

class Rocket:
    def __init__(self, network: Optional[Network] = None, config: Config = Config):
        self.config = config
        self.alive = True
        self.id = uuid.uuid4()
        self.vel = np.zeros(2)
        self.group = []
        self.detected = []
        self.network = network or Network(config.NETWORK_ARCH)
        self.pos = np.array([config.WINDOW_SIZE[0] // 2, config.WINDOW_SIZE[1] - 200], dtype=float)
        self.fitness = 0
        self.thrust = 0
        self.impacts = 0
        self.comm_dens = 0
        self.comm_open = 0
        self.alignment = 0
        self.nozzle_angle = 0.0
        self.fuel = config.MAX_FUEL
        self.dry_mass = config.WEIGHT
        self.fuel_ratio = config.WEIGHT_RATIO
        
    @property
    def mass(self) -> float:
        fuel_mass = (self.fuel / self.config.MAX_FUEL) * self.fuel_ratio * self.dry_mass
        return self.dry_mass + fuel_mass

    def get_state(self) -> np.ndarray:
        rel_pos = self.config.TARGET_POS - self.pos
        distance = np.linalg.norm(rel_pos)
        return np.array([
            self.mass / (self.dry_mass * (1 + self.fuel_ratio)),
            self.fuel / self.config.MAX_FUEL,
            distance / np.linalg.norm(self.config.WINDOW_SIZE),
            *rel_pos / self.config.WINDOW_SIZE,
            *self.vel / self.config.MAX_VELOCITY,
            self.nozzle_angle / self.config.NOZZLE_MAX,
            (self.config.WINDOW_SIZE[1] - self.pos[1]) / self.config.WINDOW_SIZE[1],
            self.alignment,                     # group alignment
            self.comm_dens,                     # comm density
            self.comm_open,                     # comm channel (must stay last)
        ])

    def update(self, group) -> None:
        if not self.alive:
            return
       
        # get environment
        self.group = group
        self.detected = self.get_sensor();
        
        # get initial state
        state = self.get_state()
        
        # handle data
        if len(self.detected) > 0:
            comm_sum = 0
            self.alignment = self._calculate_group_alignment()
            self.comm_dens = len(self.detected) / len(self.group)
            for rocket in self.detected:
                 comm_sum += rocket.network.forward(state)[-1]
            self.comm_open = np.tanh(self.comm_open + (comm_sum / len(self.detected)))
        else:
            self.alignment = 0
            self.comm_dens = 0
            self.comm_open = 0
       
        # forward state
        output = self.network.forward(state)
        
        # Commit output
        self._update_thrust(output[0])
        self._update_nozzle_angle(output[1])
        self._consume_fuel()
        self._update_position_and_velocity()
        
        if not (0 <= self.pos[1] <= self.config.WINDOW_SIZE[1]):
            self.terminate()
        else:
            self.update_fitness()

    def terminate(self):
        self.alive = False
        self.fitness = 0
        self.impacts = 0
        self.fuel = 0
        self.thrust = 0
        self.alignment = 0
        self.comm_dens = 0
        self.comm_open = 0

    def get_sensor(self):
        rockets_in_range = []
        for rocket in self.group:
            if rocket is not self and rocket.alive:
                distance = np.linalg.norm(rocket.pos - self.pos)
                if distance < self.config.SENSOR_RANGE:
                    if distance < self.config.SENSOR_RANGE_H:
                        self.impacts += 1
                    else:
                        rockets_in_range.append(rocket)
                    
        return rockets_in_range

    def _calculate_group_alignment(self) -> float:
        if len(self.detected) <= 1:
            return 0.0
        avg_velocity = np.mean([rocket.vel for rocket in self.detected if rocket.alive], axis=0)
        norm_avg_velocity = avg_velocity / np.linalg.norm(avg_velocity) if np.linalg.norm(avg_velocity) > 0 else np.zeros(2)
        norm_own_velocity = self.vel / np.linalg.norm(self.vel) if np.linalg.norm(self.vel) > 0 else np.zeros(2)
        alignment = np.dot(norm_avg_velocity, norm_own_velocity)
        return (alignment + 1) / 2

    def _update_thrust(self, thrust_output: float) -> None:
        self.thrust = max(0, thrust_output) * self.config.MAX_THRUST

    def _update_nozzle_angle(self, angle_output: float) -> None:
        target_angle = angle_output * self.config.NOZZLE_MAX
        angle_diff = target_angle - self.nozzle_angle
        self.nozzle_angle += np.clip(angle_diff, -self.config.NOZZLE_RSPD, self.config.NOZZLE_RSPD)

    def _consume_fuel(self) -> None:
        if self.fuel > 0:
            self.fuel -= self.thrust / self.config.MAX_THRUST
            if self.fuel < 0:
                self.terminate()

    def _update_position_and_velocity(self) -> None:
        self.pos, self.vel = Physics.apply_forces(
            self.pos, self.vel, self.thrust, self.nozzle_angle, self.mass, self.config
        )

    def update_fitness(self) -> None:
        if not self.alive:
            return
        # Reward traveling to target 
        distance = np.linalg.norm(self.pos - self.config.TARGET_POS)
        # Penalize high velocity 
        velocity_penalty = np.linalg.norm(self.vel) * 0.08
        # Penalize collision
        impact_penalty = self.impacts * 0.5
        # Penalize fuel consumption
        fuel_efficiency = (self.config.MAX_FUEL - self.fuel) / self.config.MAX_FUEL * 0.04
        # Calculate sum
        self.fitness += (1.0 / (distance + 1)) / (1 + velocity_penalty + impact_penalty + fuel_efficiency) * 8

    def draw(self, screen: pygame.Surface, font, extended: bool = False) -> None:
        if not self.alive:
            return

        rocket_shape = Graphics.get_rocket_shape(self.pos)
        thrust_visual = self.network.forward(self.get_state())[0]
        Graphics.draw_rocket(screen, rocket_shape, self.nozzle_angle, thrust_visual)

        if extended:
            #self._draw_network_graph(screen, font) # draws network layer
            if self.comm_open > 0:
                fx = int(self.comm_open * 255)
                radius = self.comm_open * self.config.SENSOR_RANGE_H
                pygame.draw.circle(screen, (fx,0,0), self.pos.astype(int), radius, 1)
        
    def _draw_network_graph(self, screen: pygame.Surface,
                                    font: pygame.font.Font, 
                                    node_size = 4,
                                    node_space = 1,
                                    height_offset = 50) -> None:
        # Get the specified layer from the network
        layer_data = self.network.get_layer(self.get_state(), 0)

        # Apply softmax activation to normalize the values for visualization
        exp_values = np.exp(layer_data - np.max(layer_data))
        node_states = exp_values / np.sum(exp_values)

        # Determine the grid layout
        num_nodes = len(node_states)
        grid_cols = int(np.ceil(np.sqrt(num_nodes)))
        grid_rows = int(np.ceil(num_nodes / grid_cols))

        # Set up the grid's visual dimensions
        grid_width = grid_cols * (node_size + spacing)
        grid_height = grid_rows * (node_size + spacing)

        # Starting position [centering above the rocket]
        center_x, center_y = self.pos
        start_x = center_x - grid_width // 2
        start_y = center_y - height_offset - grid_height

        # Draw each node in the grid
        for i in range(num_nodes):
            col = i % grid_cols
            row = i // grid_cols
            x = start_x + col * (node_size + spacing)
            y = start_y + row * (node_size + spacing)
            # Determine the node color based on its state
            color_intensity = int(node_states[i] * 255)
            color = (color_intensity, 10, 10) if color_intensity > 0.5 else (10, 10, 10)  # Red for active, gray for inactive
            # Draw the node as a rectangle
            pygame.draw.rect(screen, color, pygame.Rect(x, y, node_size, node_size))
            pygame.draw.rect(screen, (127,127,127), pygame.Rect(x, y, node_size, node_size),1)

class Farm:
    def __init__(self, config: Config = Config):
        self.config = config
        self.generation = 1
        self.best_fitness = 0
        self.best_specimen = None
        
        loaded_network = self.load_network()
        if loaded_network:
            self.population = [Rocket(loaded_network, config)]
            self.population.extend(Rocket(config=config) for _ in range(config.POPULATION_SIZE - 1))
        else:
            self.population = [Rocket(config=config) for _ in range(config.POPULATION_SIZE)]
    
    def load_network(self, filename: str = "snapshot.json") -> Optional[Network]:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                network = Network(self.config.NETWORK_ARCH)
                network.set_weights(np.array(data['weights']))
                return network
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def evolve(self) -> None:
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_specimen = Rocket(self.population[0].network, self.config)
            self.save_network()
        
        # Use current best from population if no best_specimen exists yet
        parent_network = self.best_specimen.network if self.best_specimen else self.population[0].network
        new_population = [Rocket(parent_network, self.config)]
        
        while len(new_population) < self.config.POPULATION_SIZE:
            parents = self.tournament_selection(2)
            child = self.create_offspring(parents)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
    
    def tournament_selection(self, count: int, tournament_size: int = 3) -> List[Rocket]:
        return [max(np.random.choice(self.population, tournament_size),
                   key=lambda x: x.fitness) for _ in range(count)]
    
    def create_offspring(self, parents: List[Rocket]) -> Rocket:
        weights = self.crossover([p.network.get_weights() for p in parents])
        self.mutate(weights)
        network = Network(self.config.NETWORK_ARCH)
        network.set_weights(weights)
        return Rocket(network, self.config)
    
    @staticmethod
    def crossover(weights_list: List[np.ndarray]) -> np.ndarray:
        point = np.random.randint(len(weights_list[0]))
        return np.concatenate((weights_list[0][:point], weights_list[1][point:]))
    
    def mutate(self, weights: np.ndarray) -> None:
        mask = np.random.random(weights.shape) < self.config.MUTATION_RATE
        weights[mask] += np.random.normal(0, self.config.MUTATION_STRENGTH, mask.sum())
    
    def save_network(self, filename: str = "snapshot.json") -> None:
        if self.best_specimen:
            data = {'weights': self.best_specimen.network.get_weights().tolist()}
            with open(filename, 'w') as f:
                json.dump(data, f)

class Physics:
    @staticmethod
    def apply_forces(
        position: np.ndarray,
        velocity: np.ndarray,
        thrust: float,
        angle: float,
        mass: float,
        config: Config
    ) -> Tuple[np.ndarray, np.ndarray]:
        thrust_angle_rad = np.radians(angle + 90)
        thrust_force = (
            np.array([np.cos(thrust_angle_rad), -np.sin(thrust_angle_rad)]) * thrust
            if thrust > 0
            else np.zeros(2)
        )
        # Compute forces
        gravity_force = np.array([0.0, config.GRAVITY]) * mass # Downward gravity
        drag_force = -velocity * np.linalg.norm(velocity) * config.DRAG # Quadratic drag
        # Calculate net acceleration
        net_acceleration = thrust_force + gravity_force + drag_force
        # Update velocity
        updated_velocity = velocity + net_acceleration
        max_velocity = config.MAX_VELOCITY
        if np.linalg.norm(updated_velocity) > max_velocity:
            updated_velocity *= max_velocity / np.linalg.norm(updated_velocity)
        # Update position with wrapping on x-axis
        updated_position = position + updated_velocity
        updated_position[0] %= config.WINDOW_SIZE[0]  # Wrap around horizontally
        return updated_position, updated_velocity

class Game:
    def __init__(self, config: Config = Config):
        pygame.init()
        self.screen = pygame.display.set_mode(config.WINDOW_SIZE)
        pygame.display.set_caption("Neural Thrust Trainer")
        self.clock = pygame.time.Clock()
        self.config = config
        self.farm = Farm(config)
        self.font = pygame.font.Font(None, 24)
        self.fontS = pygame.font.Font(None, 12)
        self.frame_count = 0
        
    def run(self) -> None:
        runetworking = True
        while runetworking:
            if pygame.event.get(pygame.QUIT):
                runetworking = False
                
            self.screen.fill(self.config.COLORS['BLACK'])
            self.draw_target()
            self.draw_stats()
            
            if self.update_population():
                self.farm.evolve()
                self.frame_count = 0
                
            pygame.display.flip()
            self.clock.tick(self.config.FPS)
            self.frame_count += 1
            
        pygame.quit()
    
    def update_population(self) -> bool:
        all_dead = True
        for rocket in self.farm.population:
            if rocket.alive:
                all_dead = False
                rocket.update(self.farm.population)
                rocket.draw(self.screen, self.fontS, True)
        return all_dead or self.frame_count >= self.config.GENERATION_TIMEOUT
    
    def draw_target(self) -> None:
        pos = self.config.TARGET_POS.astype(int)
        pygame.draw.circle(self.screen, self.config.COLORS['RED'], pos, 10)
        pygame.draw.circle(self.screen, self.config.COLORS['YELLOW'], pos, 25, 1)
    
    def draw_stats(self) -> None:
        text = self.font.render(
            f'Fitness: {self.farm.best_fitness:.2f} Gen: {self.farm.generation}',
            True, self.config.COLORS['WHITE']
        )
        self.screen.blit(text, (10, 10))

if __name__ == '__main__':
    Game().run()