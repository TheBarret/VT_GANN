import json
import uuid
import pygame
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple

# Constants (Moved to a separate section for better organization)
WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 600
FPS = 30
POPULATION_SIZE = 50
MAX_VELOCITY = 9.0
MAX_THRUST = 3.0
NOZZLE_LEN = 10
MUTATION_MULT = 0.7
MUTATION_RATE = 0.3
GRAVITY = 0.3
DRAG_COEFFICIENT = 0.05
MAX_NOZZLE_ANGLE = 45.0
NOZZLE_ROTATION_SPEED = 4.0
MAX_FUEL = 1024
MAX_GENERATION_FRAMES = 10 * FPS
TARGET_POS = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3], dtype=float)
NN_INPUT, NN_HIDDEN, NN_OUTPUT = 8, 5, 2
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# Target
TARGET_POS = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3], dtype=float)

# Network Settings
NN_INPUT, NN_HIDDEN, NN_OUTPUT = 8, 5, 2

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
        self.id = uuid.uuid4()
        self.pos = np.array([WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100], dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.nozzle_angle = 0.0
        self.nn = nn if nn else NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT)
        self.alive = True
        self.fitness = 0
        self.fuel = MAX_FUEL
        self.best_distance = float('inf')

    def calculate_inputs(self):
        rel_pos = TARGET_POS - self.pos
        distance = np.linalg.norm(rel_pos)
        return np.array([
            self.fuel / MAX_FUEL,
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
        fuel_penalty = (MAX_FUEL - self.fuel) / MAX_FUEL * 0.05 # Penalize fuel consumption
        self.fitness += (distance_reward / (1 + velocity_penalty + fuel_penalty) * 10)

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

        if self.fuel > 0 and thrust_magnitude > 0: # Only consume fuel if thrust is applied
            thrust_angle_rad = math.radians(90 + self.nozzle_angle)
            thrust_force = np.array([math.cos(thrust_angle_rad), -math.sin(thrust_angle_rad)]) * thrust_magnitude
            self.fuel -= (thrust_magnitude / MAX_THRUST) * 0.5 # Adjusted fuel consumption
            if self.fuel < 0:
                self.fuel = 0
        else:
            thrust_force = np.zeros(2) # No thrust if out of fuel

        gravity_force = np.array([0.0, GRAVITY])
        drag_force = -self.vel * np.linalg.norm(self.vel) * DRAG_COEFFICIENT

        acceleration = thrust_force + gravity_force + drag_force
        self.vel += acceleration

        if np.linalg.norm(self.vel) > MAX_VELOCITY:
            self.vel *= MAX_VELOCITY / np.linalg.norm(self.vel)

        self.pos += self.vel

        # Boundary conditions (Simplified)
        self.pos[0] %= WINDOW_WIDTH  # Wrap around horizontally
        if self.pos[1] < 0 or self.pos[1] > WINDOW_HEIGHT:
            self.alive = False
            self.fitness = 0
            return

        self.calculate_fitness()

    def draw(self, screen):
        if not self.alive:
            return

        # Rocket Body (Trapezoid with rounded top)
        body_width = 10
        body_height = 20
        body_points = [
            (self.pos[0] - body_width // 2, self.pos[1] + body_height // 2),  # Bottom Left
            (self.pos[0] + body_width // 2, self.pos[1] + body_height // 2),  # Bottom Right
            (self.pos[0] + body_width // 3, self.pos[1] - body_height // 2),  # Top Right
            (self.pos[0] - body_width // 3, self.pos[1] - body_height // 2)   # Top Left
        ]
        pygame.draw.polygon(screen, (150, 150, 150), body_points)  # Gray body

        # Fins (Triangles)
        fin_width = 4
        fin_height = 6
        fin_points_left = [
            (self.pos[0] - body_width // 2, self.pos[1] + body_height // 2),
            (self.pos[0] - body_width // 2 - fin_width, self.pos[1] + body_height // 2 + fin_height),
            (self.pos[0] - body_width // 2, self.pos[1] + body_height // 2 + fin_height)
        ]
        fin_points_right = [
            (self.pos[0] + body_width // 2, self.pos[1] + body_height // 2),
            (self.pos[0] + body_width // 2 + fin_width, self.pos[1] + body_height // 2 + fin_height),
            (self.pos[0] + body_width // 2, self.pos[1] + body_height // 2 + fin_height)
        ]
        pygame.draw.polygon(screen, (100,100,100), fin_points_left)
        pygame.draw.polygon(screen, (100,100,100), fin_points_right)

        # Nozzle (Trapezoid)
        nozzle_width = 4
        nozzle_height = 8
        nozzle_start = (self.pos[0], self.pos[1] + body_height // 2)
        nozzle_angle_rad = math.radians(self.nozzle_angle)

        nozzle_points = [
            nozzle_start,
            (nozzle_start[0] + math.sin(nozzle_angle_rad - 0.2) * nozzle_height, nozzle_start[1] + math.cos(nozzle_angle_rad - 0.2) * nozzle_height),
            (nozzle_start[0] + math.sin(nozzle_angle_rad + 0.2) * nozzle_height, nozzle_start[1] + math.cos(nozzle_angle_rad + 0.2) * nozzle_height)
        ]
        pygame.draw.polygon(screen, (50, 50, 50), nozzle_points)

        # Exhaust (Only when thrust is applied)
        inputs = self.calculate_inputs()
        output = self.nn.forward(inputs)
        thrust_magnitude = max(0, output[0]) * MAX_THRUST

        if thrust_magnitude > 0:
            exhaust_len = thrust_magnitude * 3 # Adjust length based on thrust
            exhaust_points = [
                nozzle_start,
                (nozzle_start[0] + math.sin(nozzle_angle_rad - 0.1) * (nozzle_height + exhaust_len), nozzle_start[1] + math.cos(nozzle_angle_rad - 0.1) * (nozzle_height + exhaust_len)),
                (nozzle_start[0] + math.sin(nozzle_angle_rad + 0.1) * (nozzle_height + exhaust_len), nozzle_start[1] + math.cos(nozzle_angle_rad + 0.1) * (nozzle_height + exhaust_len))
            ]
            pygame.draw.polygon(screen, (255, 165, 0, 128), exhaust_points) # Orange with alpha

class GA:
    def __init__(self):
        self.population = [Boid() for _ in range(POPULATION_SIZE)]
        self.generation = 1
        self.best_fitness = 0
        self.best_ever = None

    def create_next_generation(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_ever = Boid(self.population[0].nn)
            save_neural_network(self.best_ever.nn)
        else:
            print(f'! {self.population[0].fitness} < {self.best_fitness}')

        # Elitism: Keep the best individual
        new_population = [Boid(self.best_ever.nn)]

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = self.tournament_selection(), self.tournament_selection()
            child_weights = self.crossover(parent1.nn.get_weights(), parent2.nn.get_weights())
            self.mutate(child_weights)
            child_nn = NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT)
            child_nn.set_weights(child_weights)
            new_population.append(Boid(child_nn))

        self.population = new_population
        self.generation += 1

    def tournament_selection(self, tournament_size=3):
        return max(random.sample(self.population, tournament_size), key=lambda x: x.fitness)

    def crossover(self, weights1, weights2):
        crossover_point = random.randint(0, len(weights1) - 1)
        return np.concatenate((weights1[:crossover_point], weights2[crossover_point:]))

    def mutate(self, weights):
        mutation_mask = np.random.rand(len(weights)) < MUTATION_RATE
        weights[mutation_mask] += np.random.normal(0, MUTATION_MULT, weights[mutation_mask].shape)

# Save/Load Functions
def save_neural_network(nn, filename="snapshot.json"):
    nn_data = {
        'W1': nn.W1.tolist(),
        'b1': nn.b1.tolist(),
        'W2': nn.W2.tolist(),
        'b2': nn.b2.tolist()
    }
    with open(filename, "w") as f:
        json.dump(nn_data, f)
    print(f"! Snapshot saved to {filename}")

def load_neural_network(filename="snapshot.json"):
    try:
        with open(filename, "r") as f:
            nn_data = json.load(f)
        nn = NeuralNetwork(NN_INPUT, NN_HIDDEN, NN_OUTPUT)
        nn.W1 = np.array(nn_data['W1'])
        nn.b1 = np.array(nn_data['b1'])
        nn.W2 = np.array(nn_data['W2'])
        nn.b2 = np.array(nn_data['b2'])
        print(f"Loaded '{filename}'")
        return nn
    except FileNotFoundError:
        print(f"File '{filename}' not found. Starting with a new network.")
        return None  # Return None if the file doesn't exist

# Main Function
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("GA/NN - Vector Thrust Trainer")
    clock = pygame.time.Clock()
    ga = GA()

    best_nn = load_neural_network() # Load or create new
    if best_nn:
        ga.population[0] = Boid(best_nn)
        print("! Using old snapshot...")

    running = True
    elapsed_frames = 0
    font = pygame.font.Font(None, 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        pygame.draw.circle(screen, RED, TARGET_POS.astype(int), 10)
        pygame.draw.circle(screen, YELLOW, TARGET_POS.astype(int), 25, 1)

        text = font.render(f'Fitness: {round(ga.best_fitness, 2)} Gen: {ga.generation}', True, WHITE)
        screen.blit(text, (10, 10))

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
