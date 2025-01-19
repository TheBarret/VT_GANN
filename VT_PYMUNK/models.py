import math
import random
import pygame
import numpy as np

from typing import Tuple, Union, List, Optional


class Network:
    def __init__(self, architecture: Union[Tuple[int, ...], Tuple[int, List[int], int]]):
        # can be flat (e.g., (2, 10, 10, 2)) or nested (e.g., (2, [10, 10], 2))
        if any(isinstance(layer, list) for layer in architecture):
            flat_architecture = [architecture[0]] + [n for layers in architecture[1:-1] for n in (layers if isinstance(layers, list) else [layers])] + [architecture[-1]]
        else:
            flat_architecture = architecture
        
        self.layers = []
        for i in range(len(flat_architecture) - 1):
            self.layers.append({
                'W': np.random.randn(flat_architecture[i + 1], flat_architecture[i]) * np.sqrt(2 / flat_architecture[i]),
                'b': np.zeros(flat_architecture[i + 1])
            })
            
    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x
        for i, layer in enumerate(self.layers):
            z = np.dot(layer['W'], current) + layer['b']
            # input(tanh) -> *hidden(ReLU) -> output(tanh)
            if i < len(self.layers) - 1:
                #current = np.maximum(0, z)  # ReLU
                current = np.maximum(0.01 * z, z)  # Leaky ReLU
            else:
                current = np.tanh(z)        # Output layer
        return current
        
    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            np.concatenate([l['W'].flatten(), l['b']])
            for l in self.layers
        ])
        
    def set_weights(self, weights: np.ndarray) -> None:
        start = 0
        for layer in self.layers:
            w_size = layer['W'].size
            b_size = layer['b'].size
            layer['W'] = weights[start:start + w_size].reshape(layer['W'].shape)
            layer['b'] = weights[start + w_size:start + w_size + b_size]
            start += w_size + b_size
    
    def combine_weights(self, weights1, weights2, fitness1, fitness2):
        ratio = fitness1 / (fitness1 + fitness2)
        return ratio * weights1 + (1 - ratio) * weights2
        
    def get_layer(self, x: np.ndarray, index: int) -> np.ndarray:
        current = x
        for i, layer in enumerate(self.layers):
            z = np.dot(layer['W'], current) + layer['b']
            current = np.tanh(z)
            if i == index:
                return z
        return current
        
    def get_diff(self, other) -> float:
        # 1 positive correlation, -1 negative correlation, 0 no correlation
        weights1 = self.get_weights()
        weights2 = other.get_weights()
        return np.corrcoef(weights1, weights2)[0, 1]

class Smoke:
    def __init__(self, gravity: float):
        self.gravity = gravity
        self.particles = []
        self.config = {
            'size_range': (3, 3),
            'lifetime_range': (10, 30),
            'emission_rate': 2,
            'speed_range': (-1.0, 0.8),
            'spread_angle': 360,
            'color_map': {
                    1: (255, 40, 40),
                    2: (255, 30, 30),
                    3: (255, 20, 20),
                    4: (255, 10, 10),
                    5: (255, 0, 0)
            }
        }
        
    def stop(self):
        self.particles = []  
        
    def create(self, pos: np.ndarray):
        for _ in range(self.config['emission_rate']):
            size        = random.uniform(*self.config['size_range'])
            lifetime    = random.randint(*self.config['lifetime_range'])
            speed       = random.uniform(*self.config['speed_range'])
            angle       = random.uniform(-self.config['spread_angle'], self.config['spread_angle'])
            direction   = np.array([np.cos(np.radians(angle)) * speed, -np.sin(np.radians(angle)) * speed])
            # interpolate color by size
            color = self.get_color(size)
            self.particles.append({
                'pos': np.array([pos[0], pos[1]]),
                'vel': direction,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'max_lifetime': lifetime
            })
    def update(self):
        updated_particles = []
        for p in self.particles:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                continue
            p['vel'][1] += self.gravity * 0.3
            p['pos'] += p['vel']
            # spread angle decay
            p['vel'] *= 0.98
            fade = p['lifetime'] / p['max_lifetime']
            p['color'] = tuple(int(c * fade) for c in p['color'])
            p['size'] *= 0.99  # size decay
            updated_particles.append(p)
        self.particles = updated_particles
        
    def emit(self, screen, pos: np.ndarray):
        self.create(pos)
        self.update()
        for p in self.particles:
            pygame.draw.circle(screen, p['color'], p['pos'].astype(int), int(p['size'] * (p['lifetime'] / p['max_lifetime'])))
        
    def get_color(self, size):
        color_map = self.config['color_map']
        sizes = sorted(color_map.keys())
        # handle edge cases
        if size <= sizes[0]: return color_map[sizes[0]]
        if size >= sizes[-1]: return color_map[sizes[-1]]
        # find bracketing sizes
        for i in range(len(sizes) - 1):
            if sizes[i] <= size <= sizes[i + 1]:
                t = (size - sizes[i]) / (sizes[i + 1] - sizes[i])
                c1 = np.array(color_map[sizes[i]])
                c2 = np.array(color_map[sizes[i + 1]])
                return tuple(map(int, c1 + t * (c2 - c1)))