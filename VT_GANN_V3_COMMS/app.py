
import json
import uuid
import time
import random
import pygame
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# local models
from toolkit import Network, Graphics

@dataclass
class Config:
    WINDOW_SIZE         = (1024, 600)
    FPS                 = 30
    POPULATION_SIZE     = 25
    MUTATION_RATE       = 0.45
    MUTATION_STRENGTH   = 0.30
    GRAVITY             = 0.30
    DRAG                = 0.07
    GENERATION_TIMEOUT  = FPS * 15
    
    
    # Rocket config
    HEALTH              = 100
    SENSOR              = 75   # max sensor
    SENSOR_H            = 25   # collision distance
    ZONES               = 4
    MAX_VELOCITY        = 10.0 # max speed
    MAX_THRUST          = 7.0  # max thrust
    NOZZLE_MAX          = 30.0 # max angle
    NOZZLE_RSPD         = 0.50 # max angle rotation speed
    MAX_FUEL            = 512 # max fuel
    WEIGHT              = 10.0 # dry weight
    WEIGHT_RATIO        = 0.8  # fuel weight ratio
    
    NETWORK_ARCH        = (ZONES + 12, ZONES + 5, 5, 3) # network architecture
   
    COLORS = {
        'WHITE'     : (255, 255, 255),
        'RED'       : (255, 0, 0),
        'YELLOW'    : (255, 255, 0),
        'BLACK'     : (0, 0, 0)
    }
    TARGET_POS = np.array([WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 3], dtype=float)

class Rocket:
    def __init__(self, network: Optional[Network] = None, config: Config = Config):
        self.alive = True
        self.config = config
        self.network = network or Network(config.NETWORK_ARCH)
        self.pos = np.array([random.randrange(100, config.WINDOW_SIZE[0]-100), config.WINDOW_SIZE[1] - 200], dtype=float)
        self.id = uuid.uuid4()
        self.vel = np.zeros(2)
        self.group = []
        self.detected = []
        self.output = []
        self.state = []
        self.contact = []
        self.fitness = 0
        self.thrust = 0
        self.impacts = 0
        self.density = 0
        self.nozzle = 0.0
        self.polarity = 0
        self.interactions = 0
        self.in_range = False
        self.fuel = config.MAX_FUEL
        self.dry_mass = config.WEIGHT
        self.fuel_ratio = config.WEIGHT_RATIO
        self.target_rel = []
        self.target_dis = (config.WINDOW_SIZE[0] * config.WINDOW_SIZE[1]) / 2
        
    @property
    def mass(self) -> float:
        fuel_mass = (self.fuel / self.config.MAX_FUEL) * self.fuel_ratio * self.dry_mass
        return self.dry_mass + fuel_mass

    @property
    def state_info(self) -> dict:
        if len(self.state) > 0:
            return {
                "Altitude"  : f"{self.state[0]:.2f}",
                "Mass"      : f"{self.state[1]:.2f},",
                "Fuel"      : f"{self.state[2]:.2f}",
                "Rpos"      : f"{self.state[3]:.2f}, {self.state[4]:.2f}",
                "Rvel"      : f"{self.state[5]:.2f}, {self.state[6]:.2f}",
                "Nozzle"    : f"{self.state[7]:.2f}",
                "Target"    : f"{self.state[8]:.2f}",
                "Density"   : f"{self.state[9]:.2f}",
                # skip zones
                "Health"    : f"{self.state[14]:.2f}",
                "Polarity"  : f"{self.polarity:.2f}",
                "Fitness"   : f"{self.fitness:.2f}",
            }
        else:
            return {}

    def get_state(self) -> np.ndarray:
        self.target_rel = self.config.TARGET_POS - self.pos
        self.target_dis = np.linalg.norm(self.target_rel)
        return np.array([
            self.conv_v((self.config.WINDOW_SIZE[1] - self.pos[1]) / self.config.WINDOW_SIZE[1]),
            self.conv_v((self.mass / (self.dry_mass * (1 + self.fuel_ratio)))),
            self.conv_v((self.fuel / self.config.MAX_FUEL)),
            *self.target_rel / self.config.WINDOW_SIZE,
            *self.vel / self.config.MAX_VELOCITY,
            self.conv_v((self.nozzle / self.config.NOZZLE_MAX)),
            self.conv_v((self.target_dis / np.linalg.norm(self.config.WINDOW_SIZE))),
            self.conv_v(self.density),
            *self.contact,
            self.conv_v(self.impacts / self.config.HEALTH),
            self.polarity,
        ])

    # prepare value to -1 to 1 for neural network
    def conv_v(self, value: float, in_min: float = 0, in_max: float = 1, out_min: float = -1, out_max: float = 1) -> float:
        return max(-1, min(1, (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min))

    def update(self, group) -> None:
        if not self.alive:
            return
        # gather information
        self.group = group
        self.detected = self.get_sensor();
        
        # terminate if reached threshold
        if self.impacts > self.config.HEALTH:
            self.terminate(True)
            return
        
        # service refitting when in proximity
        self.in_range = self.target_dis < self.config.SENSOR
        if self.in_range:
            self.fuel = min(self.config.MAX_FUEL, self.fuel * 1.02)
            self.impacts = min(0, self.fuel * 0.99)
            
        # get range data
        self.contact = self._calculate_polar_field(self.config.ZONES, 2)
        
        # get output response
        self.state = self.get_state()
        self.output = self.network.forward(self.state)
            
        # Commit output
        self._update_thrust(self.output[0])
        self._update_nozzle(self.output[1])
        
        if len(self.detected) > 0:
            self.interactions += 1
            self.polarity = Network.get_diff(self.network, self.detected[0].network)
            self.density = len(self.detected) / len(self.group)
        else:
            self.polarity = 0
            self.density = 0

        self._consume_fuel()
        self._update_position_and_velocity()
        
        if not (0 <= self.pos[1] <= self.config.WINDOW_SIZE[1]):
            self.terminate(True)
        else:
            self.update_fitness()

    def terminate(self, penalize = False):
        self.alive = False
        if penalize:
            self.fitness *= 0.5
        
    def get_sensor(self):
        rockets_in_range = []
        for rocket in self.group:
            if rocket is not self and rocket.alive:
                distance = np.linalg.norm(rocket.pos - self.pos)
                if distance < self.config.SENSOR:
                    rockets_in_range.append((distance, rocket))
                    if distance < self.config.SENSOR_H and self.thrust > 0:
                        self.impacts += 1
                        
        # sort by distance
        return [rocket for _, rocket in sorted(rockets_in_range, key=lambda x: x[0])]

    def _calculate_polar_field(self, sectors = 8, width = 2) -> np.ndarray:
        sector_width = width * np.pi / sectors
        field = np.zeros(sectors)
        for rocket in self.detected:
            relative_pos = rocket.pos - self.pos
            angle = np.arctan2(relative_pos[1], relative_pos[0]) % (2 * np.pi)
            sector = int(angle // sector_width)
            distance = np.linalg.norm(relative_pos) / self.config.WINDOW_SIZE[0]
            # weighted by inverse distance
            field[sector] += 1 / (distance + 1e-5)
        # normalize to [-1, 1]
        max_value = np.max(field) + 1e-5
        mean_value = np.mean(field)
        field = (field - mean_value) / max_value
        field = np.clip(field, -1, 1)
        return field

    def _update_thrust(self, thrust_output: float) -> None:
        self.thrust = max(0, thrust_output) * self.config.MAX_THRUST

    def _update_nozzle(self, angle_output: float) -> None:
        target_angle = angle_output * self.config.NOZZLE_MAX
        angle_diff = target_angle - self.nozzle
        self.nozzle += np.clip(angle_diff, -self.config.NOZZLE_RSPD, self.config.NOZZLE_RSPD)

    def _consume_fuel(self, idle = 0.2, throttle = 0.8) -> None:
        thrust_consumption = (self.thrust / self.config.MAX_THRUST) * throttle
        self.fuel -= idle + thrust_consumption
        if self.fuel < 0:
            self.thrust = 0
            self.fuel = 0

    def _update_position_and_velocity(self) -> None:
        self.pos, self.vel = Physics.apply_forces(self.pos, self.vel, self.thrust, self.nozzle, self.mass, self.config)

    def update_fitness(self) -> None:
        if not self.alive:
            return
        extra_score = self.interactions / self.config.GENERATION_TIMEOUT
        velocity_penalty = np.linalg.norm(self.vel) * 0.5
        impact_penalty = self.impacts * 0.9
        fuel_efficiency = (self.config.MAX_FUEL - self.fuel) / self.config.MAX_FUEL * 0.5
        self.fitness += extra_score + (1.0 / (self.target_dis + 1)) / (1 + velocity_penalty + impact_penalty + fuel_efficiency)

    def draw(self, screen: pygame.Surface, font) -> None:
        if not self.alive:
            return
        fuel_norm = self.fuel / self.config.MAX_FUEL
        rocket_shape = Graphics.get_rocket_shape(self.pos)
        #self._draw_network_graph_index(screen, font, 0)
        #self._draw_network_graph_index(screen, font, 1)
        Graphics.draw_rocket(screen, rocket_shape, self.nozzle, self.output[0], fuel_norm, (255,255,255))
        if self.in_range and self.thrust > 0:
            pygame.draw.line(screen, (10, 127, 90), self.pos, self.config.TARGET_POS, 2)
            
    def get_layer_matrix(self, index):
        layer_data = self.network.get_layer(self.get_state(), index)
        exp_values = np.exp(layer_data - np.max(layer_data))
        node_states = exp_values / np.sum(exp_values)
        return node_states
        
    def _draw_network_graph_index(self, screen: pygame.Surface, font: pygame.font.Font, index = 0) -> None:
        layer_data = self.network.get_layer(self.get_state(), index)
        exp_values = np.exp(layer_data - np.max(layer_data))
        node_states = exp_values / np.sum(exp_values)
        num_nodes = len(node_states)
        for i in range(num_nodes):
            if node_states[i] > 0.1:
                color = (255, 255, 255)
                color_i = int(node_states[i] * 255)
                color_r = int(node_states[i] * self.config.SENSOR)
                #radius = 10 + node_states[i] * 10
                radius = i * 1.4
                if i % 1 == 0: color = (color_i, 255, 255)
                if i % 2 == 0: color = (255, color_i, 255)
                if i % 3 == 0: color = (255, 255, color_i)
                pygame.draw.circle(screen, color, (self.pos[0], self.pos[1] - 20), radius, 1)

class Farm:
    def __init__(self, config: Config = Config):
        self.config = config
        self.generation = 1
        self.best_fitness = 0
        self.best_specimen = None
        self.panel_width = 100
        self.panel_height = self.config.WINDOW_SIZE[1]
        self.padding = 1
        self.offset = 3
        
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
                print(f"Loading {filename}...")
                return network
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
            
    def render_debug(self, screen, font) -> None:
        self.offset = 3
        pygame.draw.rect(screen,(50, 50, 50),pygame.Rect(0, 0, self.panel_width, self.panel_height))
        for rocket in self.population:
            if not rocket.alive:
                continue
            debug_texts = [f"{key}: {value}" for key, value in rocket.state_info.items()]
            if len(rocket.contact):
                layer0 = rocket.get_layer_matrix(0)
                layer1 = rocket.get_layer_matrix(1)
                Graphics.draw_polar_grid(screen, (100, self.offset + 20), (10, 10), layer0)
                Graphics.draw_polar_grid(screen, (100, self.offset + 30), (10, 10), layer1)
            for line in debug_texts:
                text_surface = font.render(line, True, self.config.COLORS["WHITE"])
                screen.blit(text_surface, (self.padding, self.offset))
                self.offset += text_surface.get_height()
            self.offset += 5

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
                print(f"Saved {filename}...")

class Physics:
    @staticmethod
    def apply_forces(position: np.ndarray, velocity: np.ndarray, thrust: float, angle: float,mass: float, config: Config) -> Tuple[np.ndarray, np.ndarray]:
        thrust_angle_rad = np.radians(angle + 90)
        thrust_force = (np.array([np.cos(thrust_angle_rad), -np.sin(thrust_angle_rad)]) * thrust
            if thrust > 0
            else np.zeros(2)
        )
        gravity_force = np.array([0.0, config.GRAVITY]) * mass          # Downward gravity
        drag_force = -velocity * np.linalg.norm(velocity) * config.DRAG # Quadratic drag
        net_acceleration = thrust_force + gravity_force + drag_force
        updated_velocity = velocity + net_acceleration
        max_velocity = config.MAX_VELOCITY
        if np.linalg.norm(updated_velocity) > max_velocity:
            updated_velocity *= max_velocity / np.linalg.norm(updated_velocity)
        updated_position = position + updated_velocity
        # Wrap around horizontally
        updated_position[0] %= config.WINDOW_SIZE[0]
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
        self.elapsed = 0
         
    def run(self) -> None:
        running = True
        while running:
            if pygame.event.get(pygame.QUIT):
                running = False

            time_in = time.time()
            self.screen.fill(self.config.COLORS['BLACK'])
            self.farm.render_debug(self.screen , self.fontS)
            self.draw_target()
            self.draw_stats()
            
            if self.update_population():
                self.farm.evolve()
                self.frame_count = 0
            
            self.elapsed = round((time.time() - time_in) * 1000,2)
            
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
                rocket.draw(self.screen, self.fontS)
        return all_dead or self.frame_count >= self.config.GENERATION_TIMEOUT
    
    def draw_target(self) -> None:
        pos = self.config.TARGET_POS.astype(int)
        pygame.draw.circle(self.screen, self.config.COLORS['RED'], pos, 10)
        pygame.draw.circle(self.screen, self.config.COLORS['YELLOW'], pos, 25, 1)
    
    def draw_stats(self) -> None:
        text = self.font.render(f'Score: {self.farm.best_fitness:.2f} Iteration: {self.farm.generation}',True, self.config.COLORS['WHITE'])
        text2 = self.font.render(f'{self.elapsed}ms',True, self.config.COLORS['WHITE'])
        self.screen.blit(text, (self.config.WINDOW_SIZE[0] // 2, 10))
        self.screen.blit(text2, (self.config.WINDOW_SIZE[0] // 2, 30))

if __name__ == '__main__':
    Game().run()