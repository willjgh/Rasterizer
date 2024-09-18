import pygame
import os
import math
import numpy as np
import pygame.gfxdraw
from profilehooks import profile
import pandas as pd


class Triangle:
    '''Primitive faces of model objects'''
    def __init__(self, i, j, k, colour=(255, 255, 255), normal=None):

        # vertex indices (clockwise with outward normal)
        self.i = i
        self.j = j
        self.k = k

        # screen space coordinates: stored after all computation, before drawing
        self.screen_coordinates = None

        # average depth: stored after all computation
        self.average_depth = None

        # colour
        if colour == "random":
            self.colour = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        else:
            self.colour = colour
        
        # outward facing normal
        self.normal = normal


class Model:
    '''Descriptions of objects in the scene'''
    def __init__(self, scale, rotation, translation, model_points, bounding_centre, bounding_radius, model_triangles):

        # points in model, world and camera space
        self.model_points = model_points
        self.world_points = model_points
        self.camera_points = None

        # triangles of the model
        self.model_triangles = model_triangles

        # bounding sphere
        self.bounding_centre = bounding_centre
        self.bounding_radius = bounding_radius

        # transform
        self.scale_transform(scale)
        self.rotation_transform(rotation)
        self.translation_transform(translation)

    def scale_transform(self, scale):
        '''scale model about centre'''
        # centre at origin, scale, recentre
        points = self.world_points - self.bounding_centre
        points = points * scale
        points = points + self.bounding_centre
        self.world_points = points # ((self.world_points - self.bounding_centre) * scale) + self.bounding_centre
        # update bounding sphere radius
        self.bounding_radius = self.bounding_radius * scale

    def rotation_transform(self, rotation):
        '''rotate model about centre'''
        theta, phi, psi = rotation
        rotation_matrix = np.array(
            [
                [
                    np.cos(phi)*np.cos(psi),
                    -np.sin(theta)*np.sin(phi)*np.cos(psi) + np.cos(theta)*np.sin(psi),
                    np.cos(theta)*np.sin(phi)*np.cos(psi) + np.sin(theta)*np.sin(psi)
                ],
                [
                    -np.cos(phi)*np.sin(psi),
                    np.sin(theta)*np.sin(phi)*np.sin(psi) + np.cos(theta)*np.cos(psi),
                    -np.cos(theta)*np.sin(phi)*np.sin(psi) + np.sin(theta)*np.cos(psi)
                ],
                [
                    -np.sin(phi),
                    -np.sin(theta)*np.cos(phi),
                    np.cos(theta)*np.cos(phi)
                ]
            ]
        )
        points = self.world_points - self.bounding_centre
        points = points @ rotation_matrix.T
        points = points + self.bounding_centre
        self.world_points = points # ((self.world_points - self.bounding_centre) @ rotation_matrix.T) + self.bounding_centre
        # update triangle normals
        for triangle in self.model_triangles:
            triangle.normal = rotation_matrix @ triangle.normal

    def translation_transform(self, translation):
        '''translate model'''
        self.world_points = self.world_points + translation
        # update bounding sphere centre
        self.bounding_centre = self.bounding_centre + translation


class Cube(Model):
    '''Cube object'''
    def __init__(self, scale=1, rotation=(0, 0, 0), translation=np.array([0.0, 0.0, 0.0]), colour=(255, 255, 255)):

        # define points in model space
        model_points = np.array([
            [1, 1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, -1, -1],
            [1, -1, -1]
        ])

        # define bounding sphere
        bounding_centre = np.array([0.0, 0.0, 0.0])
        bounding_radius = np.sqrt(3)

        # define triangles
        model_triangles = [
            Triangle(0, 1, 5, normal=np.array([0.0, 0.0, 1.0]), colour=colour),
            Triangle(0, 5, 4, normal=np.array([0.0, 0.0, 1.0]), colour=colour),
            Triangle(2, 3, 7, normal=np.array([0.0, 0.0, -1.0]), colour=colour),
            Triangle(2, 7, 6, normal=np.array([0.0, 0.0, -1.0]), colour=colour),

            Triangle(0, 7, 3, normal=np.array([1.0, 0.0, 0.0]), colour=colour),
            Triangle(0, 4, 7, normal=np.array([1.0, 0.0, 0.0]), colour=colour),
            Triangle(1, 2, 5, normal=np.array([-1.0, 0.0, 0.0]), colour=colour),
            Triangle(2, 6, 5, normal=np.array([-1.0, 0.0, 0.0]), colour=colour),

            Triangle(0, 2, 1, normal=np.array([0.0, 1.0, 0.0]), colour=colour),
            Triangle(0, 3, 2, normal=np.array([0.0, 1.0, 0.0]), colour=colour),
            Triangle(4, 5, 7, normal=np.array([0.0, -1.0, 0.0]), colour=colour),
            Triangle(5, 6, 7, normal=np.array([0.0, -1.0, 0.0]), colour=colour)
        ]

        # call model constructor
        super().__init__(scale, rotation, translation, model_points, bounding_centre, bounding_radius, model_triangles)


class Plane(Model):
    '''plane object'''
    def __init__(self, scale=1, rotation=(0, 0, 0), translation=np.array([0.0, 0.0, 0.0]), n=3, m=3, height_map_type=None, height_map=None):

        # if not given create height map
        if height_map_type == "random":
            height_map = np.random.uniform(size=(m + 1, n + 1))
        elif height_map_type == "flat":
            height_map = np.zeros((m + 1, n + 1))

        # define points in model space
        try:
            model_points = np.array([[i - m / 2, 2 ** height_map[i, j], j - n / 2] for i in range(m + 1) for j in range(n + 1)])
        except IndexError:
            print(height_map)

        # define bounding sphere
        bounding_centre = np.array([0.0, 0.0, 0.0])
        bounding_radius = np.sqrt(m**2 + n**2) / 2


        # define triangles: 2 lists for half of each square
        model_triangles = [
            Triangle(i + (n + 1)*j, i + 1 + (n + 1)*j, i + 1 + n + (n + 1)*j, normal=np.array([0.0, 1.0, 0.0])) for j in range(m) for i in range(m)
        ] + [
            Triangle(i + 1 + (n + 1)*j, i + n + 2 + (n + 1)*j, i + n + 1 + (n + 1)*j, normal=np.array([0.0, 1.0, 0.0])) for j in range(m) for i in range(n)
        ]

        # calculate triangle normals if height_map
        if height_map_type == "random" or "input":
            for triangle in model_triangles:
                triangle.normal = np.cross(model_points[triangle.j, :] - model_points[triangle.i, :], model_points[triangle.k, :] - model_points[triangle.i, :])
                triangle.normal = triangle.normal / np.linalg.norm(triangle.normal)
        
        # call model constructor
        super().__init__(scale, rotation, translation, model_points, bounding_centre, bounding_radius, model_triangles)


class Obj(Model):
    '''object from .obj file'''
    def __init__(self, filename, scale=1, rotation=(0, 0, 0), translation=np.array([0.0, 0.0, 0.0])):

        # model data
        model_points = []
        model_triangles = []

        # read file
        file = open(filename, "r")
        for line in file:
            # vertices
            if line[0] == "v":
                data = line.split()
                x = float(data[1])
                y = float(data[2])
                z = float(data[3])
                model_points.append([x, y, z])

            # triangles
            if line[0] == "f":
                data = line.split()
                i = int(data[1]) - 1
                j = int(data[2]) - 1
                k = int(data[3]) - 1
                model_triangles.append(Triangle(i, j, k))

        # convert to array
        model_points = np.array(model_points)

        # centre at origin
        centre = np.mean(model_points, axis=0)
        model_points = model_points - centre

        # calculate triangle normals
        for triangle in model_triangles:
            triangle.normal = np.cross(model_points[triangle.j, :] - model_points[triangle.i, :], model_points[triangle.k, :] - model_points[triangle.i, :])
            triangle.normal = triangle.normal / np.linalg.norm(triangle.normal)

        # define bounding sphere
        bounding_centre = np.array([0.0, 0.0, 0.0])
        bounding_radius = np.max(np.linalg.norm(model_points, axis=0))

        # call model constructor
        super().__init__(scale, rotation, translation, model_points, bounding_centre, bounding_radius, model_triangles)
                

class Game:
    def __init__(self, window_width=700, window_height=700, canvas_width=700, canvas_height=700, view_width=4, view_height=4):
        
        # initialize pygame
        pygame.init()

        # initialise window: high resolution, display
        self.window_width = window_width
        self.window_height = window_height
        self.window = pygame.display.set_mode((window_width, window_height))

        # canvas: low resolution, draw to
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas = pygame.Surface((canvas_width, canvas_height))

        # view plane: dimensions in view / world space
        self.view_width = view_width
        self.view_height = view_height

        # configs
        pygame.display.set_caption("Rasterizer")
        self.clock = pygame.time.Clock()
        self.dt = 0.0
        self.font = pygame.font.SysFont("Arial" , 18 , bold = True)

        # game running
        self.running = True

        # models
        self.model_list = []

        # triangle draw list: per frame
        self.drawing_triangles = []

        # camera
        self.camera_position = np.array([0.0, 0.0, -4.0])
        self.camera_direction = np.array([0.0, 0.0, 1.0])
        self.theta = 0.0
        self.phi = 0.0
        self.psi = 0.0
        self.camera_matrix = None
        self.d = 1.0
        self.D = 100.0

        # normals of view frustrum
        angle_1 = np.arctan(2 * self.d / self.view_height)
        angle_2 = np.arctan(2 * self.d / self.view_width)
        self.view_normals = {
            'r1': np.array([0.0, np.sin(angle_1), -np.cos(angle_1)]),
            'r2': np.array([np.sin(angle_2), 0.0, -np.cos(angle_2)]),
            'r3': np.array([0.0, -np.sin(angle_1), -np.cos(angle_1)]),
            'r4': np.array([-np.sin(angle_2), 0.0, -np.cos(angle_2)]),
            'r5': np.array([0.0, 0.0, -1.0]),
            'r6': np.array([0.0, 0.0, 1.0]),
            'k5': -self.d,
            'k6': self.D
        }

        # lighting
        self.light = np.array([0.0, 0.0, 1.0])

        # lock mouse
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def framerate_counter(self):
        """Calculate and display frames per second."""
        # get fps
        fps = str(int(self.clock.get_fps()))
        # create text
        fps_t = self.font.render(fps , 1, (0, 255, 0))
        # display on canvas
        self.window.blit(fps_t,(0,0))


    def camera_matrix_update(self):
        '''compute 3x3 rotation matrix about x then y then z axes by theta, phi, psi'''
        self.camera_matrix = np.array(
            [
                [
                    np.cos(self.phi)*np.cos(self.psi),
                    -np.sin(self.theta)*np.sin(self.phi)*np.cos(self.psi) + np.cos(self.theta)*np.sin(self.psi),
                    np.cos(self.theta)*np.sin(self.phi)*np.cos(self.psi) + np.sin(self.theta)*np.sin(self.psi)
                ],
                [
                    -np.cos(self.phi)*np.sin(self.psi),
                    np.sin(self.theta)*np.sin(self.phi)*np.sin(self.psi) + np.cos(self.theta)*np.cos(self.psi),
                    -np.cos(self.theta)*np.sin(self.phi)*np.sin(self.psi) + np.sin(self.theta)*np.cos(self.psi)
                ],
                [
                    -np.sin(self.phi),
                    -np.sin(self.theta)*np.cos(self.phi),
                    np.cos(self.theta)*np.cos(self.phi)
                ]
            ]
        )
    

    def input(self):
        '''Take inputs'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

        # mouse movement
        dx, dy = pygame.mouse.get_rel()
        self.phi += 0.0001 * dx * self.dt
        self.theta -= 0.0001 * dy * self.dt

        # movement
        movement = np.array([0.0, 0.0, 0.0])
        step = 0.005 * self.dt

        # get keys held
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            movement[2] += step
        if keys[pygame.K_s]:
            movement[2] -= step
        if keys[pygame.K_a]:
            movement[0] -= step
        if keys[pygame.K_d]:
            movement[0] += step
        if keys[pygame.K_SPACE]:
            movement[1] += step
        if keys[pygame.K_LSHIFT]:
            movement[1] -= step

        # orientation change
        angle = 0.0025 * self.dt

         # rotate about x-axis: up, down
        if keys[pygame.K_UP]:
            self.theta += angle
            # prevent looking up past vertical
            #if self.theta < math.pi/2:
            #    self.theta += angle
        if keys[pygame.K_DOWN]:
            self.theta -= angle
            # prevent looking down past vertical
            #if self.theta > -math.pi/2:
            #    self.theta -= angle
        # rotate about y-axis: left, right
        if keys[pygame.K_LEFT]:
            self.phi -= angle
        if keys[pygame.K_RIGHT]:
            self.phi += angle
        # rotate about z-axis: (,),(.)
        if keys[pygame.K_COMMA]:
            self.psi += angle
        if keys[pygame.K_PERIOD]:
            self.psi -= angle

        # compute camera matrix
        self.camera_matrix_update()

        # rotate movement to be in relation to new camera orientation
        movement = self.camera_matrix @ movement

        # update camera position
        self.camera_position = self.camera_position + movement

        '''can change to just slice the 3rd column of the matrix'''
        # update camera direction 
        self.camera_direction = self.camera_matrix @ np.array([0.0, 0.0, 1.0])

        # 'flashlight' lighting
        # self.light = self.camera_direction


    def render(self):

        # clear triangles to draw list
        self.drawing_triangles = []

        # draw background
        self.canvas.fill((0, 0, 0))

        # loop over models
        for model in self.model_list:

            '''bounding spheres'''
            # transform centre point to view space: inverse of camera matrix transform
            centre = self.camera_matrix.T @ (model.bounding_centre - self.camera_position)
            # clip to planes: discard if failure
            if np.dot(centre, self.view_normals['r1']) > model.bounding_radius:
                continue
            if np.dot(centre, self.view_normals['r2']) > model.bounding_radius:
                continue
            if np.dot(centre, self.view_normals['r3']) > model.bounding_radius:
                continue
            if np.dot(centre, self.view_normals['r4']) > model.bounding_radius:
                continue
            if np.dot(centre, self.view_normals['r5']) > model.bounding_radius + self.view_normals['k5']:
                continue
            if np.dot(centre, self.view_normals['r6']) > model.bounding_radius + self.view_normals['k6']:
                continue
                
            '''camera transform'''
            # first translate by camera position
            model.camera_points = model.world_points - self.camera_position
            # then rotate by camera orientation
            model.camera_points = model.camera_points @ self.camera_matrix

            # loop over triangles in model
            for triangle in model.model_triangles:

                '''backface culling'''
                # if normal pointing away from direction of any triangle corner to camera: ignore
                if np.dot(triangle.normal, model.world_points[triangle.i, :] - self.camera_position) > 0:
                    continue

                '''clipping'''
                # extract vertex information
                A = model.camera_points[triangle.i, :]
                B = model.camera_points[triangle.j, :]
                C = model.camera_points[triangle.k, :]

                if A[2] < self.d or B[2] < self.d or C[2] < self.d:
                    continue
                else:

                    '''projection'''
                    # screen space coordinates
                    ax = self.canvas_width * (1/2 + ((self.d * A[0]) / (A[2] * self.view_width)))
                    ay = self.canvas_height * (1/2 - ((self.d * A[1]) / (A[2] * self.view_height)))
                    bx = self.canvas_width * (1/2 + ((self.d * B[0]) / (B[2] * self.view_width)))
                    by = self.canvas_height * (1/2 - ((self.d * B[1]) / (B[2] * self.view_height)))
                    cx = self.canvas_width * (1/2 + ((self.d * C[0]) / (C[2] * self.view_width)))
                    cy = self.canvas_height * (1/2 - ((self.d * C[1]) / (C[2] * self.view_height)))
                    
                    # store
                    triangle.screen_coordinates = {'ax': int(ax), 'ay': int(ay), 'bx': int(bx), 'by': int(by), 'cx': int(cx), 'cy': int(cy)}

                    # compute average depth: used for drawing order
                    triangle.average_depth = (A[2] + B[2] + C[2]) / 3

                    # add to triangles to be drawn
                    self.drawing_triangles.append(triangle)

        '''drawing order'''
        # sort triangles to be draw by decreasing average depth
        self.drawing_triangles.sort(key=lambda tri: tri.average_depth, reverse=True)

        '''drawing'''
        for triangle in self.drawing_triangles:

            '''lighting'''
            # dot product surface normal with light direction
            # [-1, 1]
            shade = -np.dot(self.light, triangle.normal)
            # [0, 2]
            shade += 1
            # [0, 255]
            colour = int(shade * 127.5)
            if colour < 0:
                colour = 0
            elif colour > 255:
                colour = 255
            # greyscale
            triangle.colour = (colour, colour, colour)


            pygame.gfxdraw.filled_trigon(
                self.canvas,
                triangle.screen_coordinates['ax'],
                triangle.screen_coordinates['ay'],
                triangle.screen_coordinates['bx'],
                triangle.screen_coordinates['by'],
                triangle.screen_coordinates['cx'],
                triangle.screen_coordinates['cy'],
                triangle.colour)


        # blit surface to window
        self.window.blit(pygame.transform.scale(self.canvas, self.window.get_rect().size), (0, 0))
        self.framerate_counter()

        # update canvas
        pygame.display.flip()


    def load(self):
        '''Populate the scene with objects'''

        
        # Load a spaced 3D grid of cubes

        self.model_list = [
            Cube(
                scale = 1,
                rotation = (np.random.uniform(0, np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)),
                translation=np.array([float(5*x), float(5*y), float(5*z)]),
                colour = "random"
            )
        for x in range(-2, 2) for y in range(-2, 2) for z in range(-2, 2)]


        '''
        # Load model from an obj file

        obj = Obj("Rasterizer/Models/teapot.obj", scale=1)
        self.model_list = [obj]
        '''
      
        '''
        # Load a grid of triangles that follow a specified heightmap

        # squares per side of grid
        M = 10
        N = 10
        # squares per model
        K = 2
        # units per square
        S = 1

        self.model_list = []

        # heightmap of plane
        full_height_map = np.random.uniform(size=(M + 1, N + 1))

        # construct plane
        for i in range(0, M - K + 1, K):
            for j in range(0, N - K + 1, K):
                self.model_list.append(
                    Plane(height_map=full_height_map[i:i+1+K, j:j+1+K], height_map_type="input", m=K, n=K, scale=S, translation=np.array([i*S, 0.0, j*S]))
                )
        '''

    
    def update(self):
        pass
        '''
        # Rotate all models per frame
        for model in self.model_list:
            model.rotation_transform((0.001 * self.dt, 0.001 * self.dt, 0.001 * self.dt))
        '''

    def run(self):
        
        # load
        self.load()

        # loop
        while self.running:

            # clock
            self.dt = self.clock.tick()

            # take input
            self.input()

            # update
            self.update()

            # draw
            self.render()

        # quit
        pygame.quit()

@profile
def main():
    game = Game(window_width=700, window_height=700, canvas_width=700, canvas_height=700, view_width=3, view_height=3)
    game.run()

if __name__ == "__main__":
    main()