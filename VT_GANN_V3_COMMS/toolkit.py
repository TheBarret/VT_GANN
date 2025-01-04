import math
import random
import pygame
import numpy as np
from typing import Tuple, Union, List, Optional

class Network:
    def __init__(self, architecture: Union[Tuple[int, ...], Tuple[int, List[int], int]]):
        #:param architecture: Tuple defining the number of neurons in each layer.
        #                     Can be flat (e.g., (2, 10, 10, 2)) or nested (e.g., (2, [10, 10], 2)).
        # Flatten the architecture if it includes nested lists
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
        for layer in self.layers:
            current = np.tanh(np.dot(layer['W'], current) + layer['b'])
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

    def get_layer(self, x: np.ndarray, index: int) -> np.ndarray:
        current = x
        for i, layer in enumerate(self.layers):
            z = np.dot(layer['W'], current) + layer['b']
            current = np.tanh(z)
            if i == index:
                return z
        return current

    def get_diff(self, other) -> float:
        # Get the weights of both networks
        weights1 = self.get_weights()
        weights2 = other.get_weights()
        # Calculate the correlation coefficient between the two network weights
        #  1 positive correlation
        # -1 negative correlation 
        #  0 no correlation
        return np.corrcoef(weights1, weights2)[0, 1]
    
class Graphics:
    smoke_particles = []
    
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
            ),
            'fuel_bar': [pos + np.array([0, 5]), pos + np.array([0, -5])]
        }
    
    @staticmethod
    def draw_rocket(screen: pygame.Surface,
                    shape: dict, angle: float, thrust: float, fuel: float,
                    body=(150, 150, 150), cockpit=(200, 200, 200),
                    fins=(100, 100, 100), flame=(255, 165, 0)) -> None:
        Graphics._draw_flame(screen, shape['body'], angle, thrust, fuel, flame)
        Graphics._draw_body(screen, shape['body'], body)
        Graphics._draw_fins(screen, shape['fins'], fins)
        Graphics._draw_cockpit(screen, shape['body'], cockpit)
        Graphics._draw_fuel_bar(screen, shape['fuel_bar'], fuel)

    @staticmethod
    def _draw_body(screen: pygame.Surface, body_shape: np.ndarray, color: tuple) -> None:
        pygame.draw.polygon(screen, color, body_shape)

    @staticmethod
    def _draw_fins(screen: pygame.Surface, fins: tuple, color: tuple) -> None:
        for fin in fins:
            pygame.draw.polygon(screen, color, fin)

    @staticmethod
    def _draw_cockpit(screen: pygame.Surface, body_shape: np.ndarray, color: tuple) -> None:
        cockpit_pos = (body_shape[2] + body_shape[3]) / 2
        pygame.draw.circle(screen, color, cockpit_pos.astype(int), 4)

    @staticmethod
    def _draw_fuel_bar(screen: pygame.Surface, fuel_bar: tuple, fuel: float) -> None:
        fuel_bar_start, fuel_bar_end = fuel_bar
        fuel_bar_height = (fuel_bar_start - fuel_bar_end) * fuel
        current_fuel_end = fuel_bar_start - fuel_bar_height
        pygame.draw.line(screen, (0, 50, 50), fuel_bar_start, fuel_bar_end, 4)
        pygame.draw.line(screen, (0, 255, 0), fuel_bar_start, current_fuel_end, 3)
    
    @staticmethod
    def _draw_flame(screen: pygame.Surface, body_shape: np.ndarray, angle: float, thrust: float, fuel: float, color: tuple) -> None:
        if fuel > 0 and thrust > 0:
            nozzle_pos = body_shape[0] + (body_shape[1] - body_shape[0]) / 2
            base_length = thrust * 15
            flicker = random.uniform(-3, 3)
            thrust_length = base_length + flicker
            thrust_angle = np.radians(-angle + 90)
            end_pos = nozzle_pos + thrust_length * np.array([np.cos(thrust_angle), np.sin(thrust_angle)])
            # Draw flame with fading effect
            pygame.draw.line(screen, (255, 0, 0), nozzle_pos, end_pos, 7)     # Base flame
            pygame.draw.line(screen, color, nozzle_pos, end_pos, 3)             # Inner flame
    
    @staticmethod
    def draw_polar_grid(screen: pygame.Surface, start_pos: tuple, cell_size: tuple, polar_field: np.ndarray, colors: tuple = ((0, 30, 30), (0, 255, 0))) -> None:
        x, y = start_pos
        cell_width, cell_height = cell_size

        for i, value in enumerate(polar_field):
            # intensity [-1, 1] to [0, 1]
            intensity = (value + 1) / 2  
            tint = int(intensity * 255)
            
            if value > 0.5:
                color = (0,tint,0)
            elif value < 0.5:
                color = (tint,0,0)
            else:
                color = (0,0,0)

            # Draw the cell
            pygame.draw.rect(
                screen, color, pygame.Rect(x + i * cell_width, y, cell_width, cell_height)
            )
            pygame.draw.rect(
                screen, (50, 50, 50), pygame.Rect(x + i * cell_width, y, cell_width, cell_height), 1
            )