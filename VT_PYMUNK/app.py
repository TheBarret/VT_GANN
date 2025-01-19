import random
import pygame
import pymunk
import numpy as np
from typing import Tuple, Union, List, Optional
from dataclasses import dataclass

from models import Smoke, Network

@dataclass
class Config:
    fps: int = 30
    width: int = 800
    height: int = 600
    gravity: float = 9.8
    dt: float = 1 / fps
    max_thrust: int = 500
    network = (4, 5, 2)

class Vehicle:
    def __init__(self, space, x, y, size, config, network: Optional[Network] = None):
        self.config = config
        self.enabled = True
        self.size = (size, size + (size // 2))
        self.mass = 5
        self.color = (100, 100, 100)
        self.thruster_angle = 0.0
        self.thruster_strength = 0.0
        self.network = network or Network(config.network)
        # pymunk body and shape
        self.body = pymunk.Body(self.mass, float('inf'))
        self.body.position = (x, y)
        self.shape = pymunk.Poly.create_box(self.body, self.size)
        self.shape.elasticity = 0.3
        self.space = space
        space.add(self.body, self.shape)
        # particle effect
        self.particles = Smoke(self.config.gravity)
        
    def terminate(self):
        self.enabled = False
        self.space.remove(self.body, self.shape)

    def update(self):
        # test
        self.apply_thruster(-0.50, 0.98)
        x, y = self.body.position
        angle = self.body.angle
        w, h = self.size
        # Create the rectangle's corners in local space
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        # Rotate and translate the corners
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        transformed_corners = corners @ rotation_matrix.T + np.array([x, y])
        self.corners = [tuple(corner) for corner in transformed_corners]

    def draw(self, screen):
        if self.enabled:
            self.draw_chassis(screen)
            self.draw_thruster(screen)
            self.draw_vectors(screen, 25)
            
    def draw_chassis(self, screen):
        pygame.draw.polygon(screen, self.color, self.corners)
        
    def apply_thruster(self, angle: float, strength: float):
        self.thruster_angle = np.clip(angle, -1, 1) * np.pi
        self.thruster_strength = np.clip(strength, -1, 1)
        # Calculate thrust vector
        thrust_force = min(self.config.max_thrust, self.config.max_thrust * abs(self.thruster_strength))
        thrust_direction = np.array([np.cos(self.thruster_angle), np.sin(self.thruster_angle)])
        force = thrust_force * thrust_direction
        # calculate application point, bottom center of the vehicle
        application_point = (0, self.size[1] / 2)
        # apply force body
        self.body.apply_force_at_local_point((force[0], force[1]), application_point)
        # calculate torque and apply it
        lever_arm = np.array(application_point)  # Already in local coordinates
        # torque
        torque = lever_arm[0] * force[1] - lever_arm[1] * force[0]  # 2D cross product
        self.body.torque += torque

    def draw_thruster(self, screen: pygame.Surface):
        if self.thruster_strength > 0:
            # get the rear nozzle position
            body_shape = np.array(self.corners)
            # midpoint of bottom edge
            nozzle_pos = body_shape[2] + (body_shape[3] - body_shape[2]) / 2
            # calculate flame length and flicker
            flicker = random.uniform(-3, 3)
            thrust_length = (self.thruster_strength * 10) + flicker
            # compute flame end position
            thrust_angle = self.body.angle + self.thruster_angle + np.pi
            end_pos = nozzle_pos + thrust_length * np.array([np.cos(thrust_angle), np.sin(thrust_angle)])
            # draw particles
            particle_pos = (nozzle_pos[0] + 1, end_pos[1])
            self.particles.emit(screen, particle_pos)
            # draw the flames
            pygame.draw.line(screen, (255, 0, 0), nozzle_pos, end_pos, 4)   # outer flame
            pygame.draw.line(screen, (255, 255, 0), nozzle_pos, end_pos, 2) # inner flame
            
    def draw_vectors(self, screen, length):
        if not self.enabled:
            return
        center = np.array(self.body.position)
        pygame.draw.circle(screen, (255, 255, 0), center.astype(int), 1)
        # Thrust vector
        thrust_direction = np.array([np.cos(self.thruster_angle), np.sin(self.thruster_angle)])
        thrust_vector = thrust_direction * self.thruster_strength * length
        thrust_end = center + thrust_vector
        pygame.draw.line(screen, (0, 255, 0), center.astype(int), thrust_end.astype(int), 1)
        # Torque visualization (rotation vector)
        angular_velocity = self.body.angular_velocity
        torque_vector = np.array([-np.sin(self.body.angle), np.cos(self.body.angle)]) * angular_velocity * length
        torque_end = center + torque_vector
        pygame.draw.line(screen, (255, 0, 0), center.astype(int), torque_end.astype(int), 1)
        
class Game:
    def __init__(self, config):
        self.config = config
        pygame.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("Vehicle Simulation")
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.space.gravity = (0, config.gravity)
        self.create_confinement()
        self.vehicles = self.create_vehicles(25)

    def create_vehicles(self, amount):
        return [Vehicle(self.space, random.randint(50, self.config.width - 50),
                        random.randint(50, self.config.height - 50), 10, self.config)
                for _ in range(amount)]

    def create_confinement(self):
        walls = [
            [(0, 0), (0, self.config.height)],
            [(self.config.width, 0), (self.config.width, self.config.height)],
            [(0, self.config.height), (self.config.width, self.config.height)],
            [(0, 0), (self.config.width, 0)]
        ]

        for wall in walls:
            shape = pymunk.Segment(self.space.static_body, wall[0], wall[1], 1)
            shape.friction = 0.1
            shape.elasticity = 0.8
            self.space.add(shape)

    def handle_input(self):
        keys = pygame.key.get_pressed()

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.handle_input()

            for _ in range(10):
                self.space.step(self.config.dt)

            self.screen.fill((0, 0, 0))
            for vehicle in self.vehicles:
                if vehicle.enabled:
                    vehicle.update()
                    vehicle.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        pygame.quit()

if __name__ == "__main__":
    config = Config()
    game = Game(config)
    game.run()
