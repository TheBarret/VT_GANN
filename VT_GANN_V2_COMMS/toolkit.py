import pygame
import numpy as np
from typing import List, Optional, Tuple

def softmax(x):
    # subtract maximum to avoid overflow
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class Network:
    def __init__(self, architecture: Tuple[int, ...]):
        self.layers = []
        for i in range(len(architecture) - 1):
            self.layers.append({
                'W': np.random.randn(architecture[i+1], architecture[i]) * np.sqrt(2/architecture[i]),
                'b': np.zeros(architecture[i+1])
            })
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x
        for layer in self.layers:
            current = np.tanh(np.dot(layer['W'], current) + layer['b'])
        return current
    
    def get_weights(self) -> np.ndarray:
        return np.concatenate([np.concatenate([l['W'].flatten(), l['b']]) for l in self.layers])
    
    def set_weights(self, weights: np.ndarray) -> None:
        start = 0
        for layer in self.layers:
            w_size = layer['W'].size
            b_size = layer['b'].size
            layer['W'] = weights[start:start + w_size].reshape(layer['W'].shape)
            layer['b'] = weights[start + w_size:start + w_size + b_size]
            start += w_size + b_size
            
    def get_layer(self, x: np.ndarray, index: int) -> np.ndarray:
        current = x
        # Get hidden layer output
        for i, layer in enumerate(self.layers):
            z = np.dot(layer['W'], current) + layer['b']
            current = np.tanh(z)
            # Return first hidden layer
            if i == index:
                return z
        return current
    
class Graphics:
    @staticmethod
    def get_rocket_shape(pos: np.ndarray) -> dict:
        return {
            'body': np.array([
                [-5, 10], [5, 10],
                [3, -10], [-3, -10]
            ]) + pos,
            'fins': (
                np.array([[-5, 10], [-9, 16], [-5, 16]]) + pos,
                np.array([[5, 10], [9, 16], [5, 16]]) + pos
            )
        }
    
    @staticmethod
    def draw_rocket(screen: pygame.Surface, shape: dict, angle: float, thrust: float) -> None:
        pygame.draw.polygon(screen, (150, 150, 150), shape['body'])
        for fin in shape['fins']:
            pygame.draw.polygon(screen, (100, 100, 100), fin)
            
        if thrust > 0:
            nozzle_pos = shape['body'][0] + (shape['body'][1] - shape['body'][0]) / 2
            thrust_length = thrust * 25
            thrust_angle = np.radians(-angle + 90)
            end_pos = nozzle_pos + thrust_length * np.array([
                np.cos(thrust_angle),
                np.sin(thrust_angle)
            ])
            pygame.draw.line(screen, (255, 165, 0), nozzle_pos, end_pos, 3)