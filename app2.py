import json
import uuid
import pygame
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple

# Game Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 600
FPS                     = 30
POPULATION_SIZE         = 50

# Physics Settings
MAX_VELOCITY            = 9.0
MAX_THRUST              = 3.0
NOZZLE_LEN              = 15.0
MUTATION_MULT           = 0.7
MUTATION_RATE           = 0.3
GRAVITY                 = 0.3
DRAG_COEFFICIENT        = 0.05
MAX_NOZZLE_ANGLE        = 45.0
NOZZLE_ROTATION_SPEED   = 4.0
# 10 seconds per generation
MAX_GENERATION_FRAMES   = 10 * FPS

# Colors
WHITE                   = (255, 255, 255)
RED                     = (255, 0, 0)
YELLOW                  = (255, 255, 0)
BLACK                   = (0, 0, 0)
BLUE                    = (0, 0, 255)

# Target
TARGET_POS = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3], dtype=float)

# Network Settings
NN_INPUT, NN_HIDDEN, NN_OUTPUT = 8, 12, 2

pygame.init()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        return np.tanh(z2)

    def get_weights(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_weights(self, weights):
        split1, split2, split3 = self.W1.size, self.W1.size + self.b1.size, self.W1.size + self.b1.size + self.W2.size
        self.W1, self.b1 = weights[:split1].reshape(self.W1.shape), weights[split1:split2]
        self.W2, self.b2 = weights[split2:split3].reshape(self.W2.shape), weights[split3:]

class Boid:
    def __init__(self, nn=None):
        #self.sprite = pygame.image.load('')
        self.id = uuid.uuid4()
        self.pos = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100], dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.nozzle_angle = 0.0
        self.nn = nn if nn else NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT)
        self.alive = True
        self.fitness = 0
        self.fuel = 512
        self.best_distance = float('inf')
        
    def calculate_inputs(self):
        rel_pos = TARGET_POS - self.pos
        distance = np.linalg.norm(rel_pos)
        if (distance < 10):
            self.fitness += 10
            #TARGET_POS = np.array([random.randrange(50, WINDOW_WIDTH - 50),
            #                       random.randrange(50, WINDOW_HEIGHT - 50)], dtype=float)
        return np.array([
            self.fuel / 512,
            distance / np.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2),
            rel_pos[0] / WINDOW_WIDTH,
            rel_pos[1] / WINDOW_HEIGHT,
            self.vel[0] / MAX_VELOCITY,
            self.vel[1] / MAX_VELOCITY,
            self.nozzle_angle / MAX_NOZZLE_ANGLE,
            (WINDOW_HEIGHT - self.pos[1]) / WINDOW_HEIGHT
        ])

    def calculate_fitness(self):
        distance = np.linalg.norm(self.pos - TARGET_POS)
        self.best_distance = min(self.best_distance, distance)
        distance_reward = 1.0 / (distance + 1)
        velocity_penalty = np.linalg.norm(self.vel) * 0.1
        nozzle_penalty = abs(self.nozzle_angle) / MAX_NOZZLE_ANGLE * 0.1
        self.fitness += (distance_reward / (1 + velocity_penalty + nozzle_penalty) * 10)

    def get_nozzle_length(self, min_length=5, max_length=15):
        # Get the neural network output for the boid's current state
        # Thrust magnitude is output[0], scaled by MAX_THRUST
        # Normalize thrust magnitude to a value between 0 and 1
        # Scale nozzle length based on normalized thrust
        inputs = self.calculate_inputs()
        output = self.nn.forward(inputs)
        thrust_magnitude = max(0, output[0]) * MAX_THRUST
        normalized_thrust = thrust_magnitude / MAX_THRUST
        nozzle_length = min_length + normalized_thrust * (max_length - min_length)
        return nozzle_length

    def update(self):
        if not self.alive:
            return

        inputs = self.calculate_inputs()
        output = self.nn.forward(inputs)
        thrust_magnitude = max(0, output[0]) * MAX_THRUST
        desired_nozzle_angle = output[1] * MAX_NOZZLE_ANGLE

        angle_diff = desired_nozzle_angle - self.nozzle_angle
        self.nozzle_angle += np.clip(angle_diff, -NOZZLE_ROTATION_SPEED, NOZZLE_ROTATION_SPEED)
        self.nozzle_angle = np.clip(self.nozzle_angle, -MAX_NOZZLE_ANGLE, MAX_NOZZLE_ANGLE)

        if (self.fuel > 0):
            thrust_angle_rad = math.radians(90 + self.nozzle_angle)
            thrust_force = np.array([math.cos(thrust_angle_rad), -math.sin(thrust_angle_rad)]) * thrust_magnitude
            self.fuel -= 1
            if (self.fuel < 0): self.fuel = 0
        else:
            thrust_angle_rad = 90
            thrust_force = 0
            
        gravity_force = np.array([0.0, GRAVITY])
        drag_force = -self.vel * np.linalg.norm(self.vel) * DRAG_COEFFICIENT

        acceleration = thrust_force + gravity_force + drag_force
        self.vel += acceleration

        if np.linalg.norm(self.vel) > MAX_VELOCITY:
            self.vel *= MAX_VELOCITY / np.linalg.norm(self.vel)

        self.pos += self.vel

        # Wrap around left and right
        if self.pos[0] < 0:
            self.pos[0] = WINDOW_WIDTH
        elif self.pos[0] > WINDOW_WIDTH:
            self.pos[0] = 0
        # Terminate if out of bounds top and bottom
        if self.pos[1] < 0 or self.pos[1] > WINDOW_HEIGHT:
            self.alive = False
            self.fitness = 0
            return

        self.calculate_fitness()

    def draw(self, screen):
        if not self.alive:
            return
        body_surface = pygame.Surface((10, 30), pygame.SRCALPHA)
        pygame.draw.rect(body_surface, (255,255,255), (0, 0, 10, 10))
        screen.blit(body_surface, (self.pos[0] - 5, self.pos[1] - 15))

        nozzle_start = (self.pos[0], self.pos[1] - 5)
        nozzle_angle_rad = math.radians(self.nozzle_angle)
        nozzle_end = (
            nozzle_start[0] + math.sin(nozzle_angle_rad) * self.get_nozzle_length(1,NOZZLE_LEN),
            nozzle_start[1] + math.cos(nozzle_angle_rad) * self.get_nozzle_length(1,NOZZLE_LEN)
        )
        pygame.draw.line(screen, (90,90,90), nozzle_start, nozzle_end, 5)

class GA:
    def __init__(self):
        self.population = [Boid() for _ in range(POPULATION_SIZE)]
        self.generation = 1
        self.best_fitness = 0
        self.best_boid = None

    def create_next_generation(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_ever = Boid(self.population[0].nn)
            save_neural_network(self.best_ever.nn)
        else:
            print(f'! {self.population[0].fitness} < {self.best_fitness}')

        new_population = [Boid(self.population[0].nn)]
                
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = self.select_parent(), self.select_parent()
            child_weights = self.crossover(parent1.nn.get_weights(), parent2.nn.get_weights())
            self.mutate(child_weights)
            child_nn = NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT)
            child_nn.set_weights(child_weights)
            new_population.append(Boid(child_nn))

        self.population = new_population
        self.generation += 1

    def select_parent(self):
        return max(random.sample(self.population, 3), key=lambda x: x.fitness)

    def crossover(self, weights1, weights2):
        return np.where(np.random.rand(len(weights1)) < 0.5, weights1, weights2)

    def mutate(self, weights):
        mutation_mask = np.random.rand(len(weights)) < MUTATION_RATE
        weights[mutation_mask] += np.random.normal(0, MUTATION_MULT, weights[mutation_mask].shape)

def save_neural_network(nn, filename="snapshot.json"):
    nn_data = {
        'W1': nn.W1.tolist(),
        'b1': nn.b1.tolist(),
        'W2': nn.W2.tolist(),
        'b2': nn.b2.tolist()
    }
    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(nn_data, f)
    print(f"! Snapshot saved to {filename}")

def load_neural_network(filename="snapshot.json", strength_multiplier=1.0):
    with open(filename, "r") as f:
        nn_data = json.load(f)
    # Create a new neural network and set scaled weights and biases
    nn = NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT)
    nn.W1 = np.array(nn_data['W1']) * strength_multiplier
    nn.b1 = np.array(nn_data['b1']) * strength_multiplier
    nn.W2 = np.array(nn_data['W2']) * strength_multiplier
    nn.b2 = np.array(nn_data['b2']) * strength_multiplier
    print(f"Loaded '{filename}' [{strength_multiplier}]")
    return nn

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("GA/NN - Vector Thrust Trainer")
    clock = pygame.time.Clock()
    ga = GA()
    
    if Path('snapshot.json').is_file():
        best_nn = load_neural_network('snapshot.json')
        # Use it for the first boid in the population
        ga.population[0] = Boid(best_nn)
        print("! Using old snapshot...")
    
    running = True
    elapsed_frames = 0
    # Text
    font = pygame.font.Font(None, 24)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        pygame.draw.circle(screen, RED, TARGET_POS.astype(int), 10)
        pygame.draw.circle(screen, YELLOW, TARGET_POS.astype(int), 25, 1)
        
        # Debug boid
        text = font.render(f'Fitness: {round(ga.best_fitness)}', True, WHITE)
        screen.blit(text, (10,10))
        
        all_dead = True
        for boid in ga.population:
            if boid.alive:
                all_dead = False
                boid.update()
                boid.draw(screen)

        if all_dead or elapsed_frames >= MAX_GENERATION_FRAMES:
            ga.create_next_generation()
            elapsed_frames = 0

        pygame.display.flip()
        clock.tick(FPS)
        elapsed_frames += 1

    pygame.quit()

if __name__ == '__main__':
    main()
