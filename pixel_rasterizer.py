import pygame
import os
import math
import numpy as np
import pygame.gfxdraw
from profilehooks import profile
import pandas as pd
import copy


class Triangle:
    '''Primitive faces of model objects'''
    def __init__(self, i, j, k, colour=(255, 255, 255), normal=None):

        # vertex indices (clockwise with outward normal)
        self.i = i
        self.j = j
        self.k = k

        # vertices: stored after camera computation
        self.vertices = None

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


class Test(Model):
    '''Single triangle model for testing'''
    def __init__(self):
        # define points in model space
        model_points = np.array([
            [0, 4, 4],
            [-2, 2, 4],
            [0, 0, 4]
        ])

        # define bounding sphere
        bounding_centre = np.array([0.0, 0.0, 0.0])
        bounding_radius = 10

        # define triangles
        model_triangles = [
            Triangle(0, 1, 2, normal=np.array([0.0, 0.0, -1.0]), colour=(255, 0, 0))
        ]

        # call model constructor
        super().__init__(scale=1,
                         rotation=(0, 0, 0),
                         translation=np.array([0.0, 0.0, 0.0]),
                         model_points=model_points,
                         bounding_centre=bounding_centre,
                         bounding_radius=bounding_radius,
                         model_triangles=model_triangles)


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
    def __init__(self, filename, scale=1, rotation=(0, 0, 0), translation=np.array([0.0, 0.0, 0.0]), colour=(255, 255, 255)):

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
                model_triangles.append(Triangle(i, j, k, colour=colour))

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
            'k1': 0.0,
            'k2': 0.0,
            'k3': 0.0,
            'k4': 0.0,
            'k5': -self.d,
            'k6': self.D
        }

        # depth buffer
        self.depth_buffer = np.zeros((canvas_height + 1, canvas_width + 1))

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

    
    def plane_intersection(self, A, B):
        '''
        Compute intersection point of the line AB and plane

        A: tuple (point, distance to plane)
        B: tuple (point, distance to plane)
        '''
        # t = dA / (dA - dB)
        t = A[1] / (A[1] - B[1])
        # B' = A + (B - A)t
        return A[0] + (B[0] - A[0])*t

    
    def interpolate(self, i0, d0, i1, d1):
        '''
        Interpolate between (i0, d0) and (i1, d1), producing a list of values
        of the dependent variable d for each value of the independent variable
        i from i0 to i1

        i0, d0, i1, d1: integers (start and end points)
        '''
        # dependent variable values
        values = []

        # 'gradient' and 'intercept'
        m = (d1 - d0) / (i1 - i0)
        d = d0

        # iterate over independent variable
        for i in range(i0, i1 + 1):
            # store as integer
            values.append(int(d))
            # increment
            d += m

        return values


    def render(self):

        # clear triangles to draw list
        self.drawing_triangles = []

        pygame.gfxdraw.pixel(self.canvas, 0, 0, (255, 255, 255))

        # clear depth buffer
        self.depth_buffer = np.zeros((self.canvas_height + 1, self.canvas_width + 1))

        # draw background
        self.canvas.fill((0, 0, 0))

        # loop over models
        for model in self.model_list:

            '''bounding spheres'''
            # transform centre point to view space: inverse of camera matrix transform
            centre = self.camera_matrix.T @ (model.bounding_centre - self.camera_position)
            # comapre to each view plane in turn
            for plane in range(1, 7):
                # bounding sphere outside plane: discard model
                if np.dot(centre, self.view_normals[f'r{plane}']) > model.bounding_radius + self.view_normals[f'k{plane}']:
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

                # if visible: extract and store vertex information on triangle
                else:
                    # vertices
                    A = model.camera_points[triangle.i, :]
                    B = model.camera_points[triangle.j, :]
                    C = model.camera_points[triangle.k, :]

                    # store
                    triangle.vertices = [A, B, C]

                    # add to drawing list
                    self.drawing_triangles.append(triangle)


        '''clipping'''
        # copy of triangle to draw list
        self.drawing_triangles_new = []

        # planes to clip to
        clipping_planes = [1, 2, 3, 4, 5, 6]
        # clipping_planes = [5]

        # clip to each view plane in turn
        for plane in clipping_planes:

            # loop over triangles to be drawn
            for triangle in self.drawing_triangles:

                # points inside / outside plane
                inside_points = []
                outside_points = []
                inside_point_number = 0

                # compute distance of vertices to plane
                for v in triangle.vertices:
                    dv = np.dot(v, self.view_normals[f'r{plane}']) - self.view_normals[f'k{plane}']

                    # store status of vertices: inside / outside plane
                    if dv > 0:
                        outside_points.append((v, dv))
                    else:
                        inside_points.append((v, dv))
                        inside_point_number += 1


                # all vertices outside plane
                if inside_point_number == 0:

                    # do not add to new list
                    continue

                # all vertices inside plane
                elif inside_point_number == 3:
                    
                    # add to new list
                    self.drawing_triangles_new.append(triangle)

                # one vertex inside
                elif inside_point_number == 1:

                    # extract vertices
                    A = inside_points[0]
                    B = outside_points[0]
                    C = outside_points[1]

                    # compute new vertices
                    B_prime = self.plane_intersection(A, B)
                    C_prime = self.plane_intersection(A, C)

                    # update triangle vertices (clip)
                    triangle.vertices = [A[0], B_prime, C_prime]

                    ''' testing: red for 1 clipped triangle '''
                    # triangle.colour = (255, 0, 0)

                    # add to new list
                    self.drawing_triangles_new.append(triangle)

                # two vertices inside: clip
                elif inside_point_number == 2:
                    
                    # extract vertices
                    A = outside_points[0]
                    B = inside_points[0]
                    C = inside_points[1]

                    # compute new vertices
                    B_prime = self.plane_intersection(A, B)
                    C_prime = self.plane_intersection(A, C)

                    # create a copy of triangle
                    triangle_copy = copy.deepcopy(triangle)

                    # update vertices of both (clipping)
                    triangle.vertices = [B_prime, B[0], C[0]]
                    triangle_copy.vertices = [B_prime, C[0], C_prime]

                    ''' testing: green, blue for 2 clipped triangles'''
                    # triangle.colour = (0, 255, 0)
                    # triangle_copy.colour = (0, 0, 255)

                    # add both triangles to new list
                    self.drawing_triangles_new.append(triangle)
                    self.drawing_triangles_new.append(triangle_copy)
                
            # update drawing lists
            self.drawing_triangles = self.drawing_triangles_new
            self.drawing_triangles_new = []


        '''projection'''
        for triangle in self.drawing_triangles:

            # extract vertices
            A = triangle.vertices[0]
            B = triangle.vertices[1]
            C = triangle.vertices[2]

            # screen space coordinates
            ax = self.canvas_width * (1/2 + ((self.d * A[0]) / (A[2] * self.view_width)))
            ay = self.canvas_height * (1/2 - ((self.d * A[1]) / (A[2] * self.view_height)))
            bx = self.canvas_width * (1/2 + ((self.d * B[0]) / (B[2] * self.view_width)))
            by = self.canvas_height * (1/2 - ((self.d * B[1]) / (B[2] * self.view_height)))
            cx = self.canvas_width * (1/2 + ((self.d * C[0]) / (C[2] * self.view_width)))
            cy = self.canvas_height * (1/2 - ((self.d * C[1]) / (C[2] * self.view_height)))

            # reciprocal depth values
            az = 1 / A[2]
            bz = 1 / B[2]
            cz = 1 / C[2]
            
            # store
            triangle.screen_coordinates = [
                {'x': int(ax), 'y': int(ay), 'z': az},
                {'x': int(bx), 'y': int(by), 'z': bz},
                {'x': int(cx), 'y': int(cy), 'z': cz}
            ]
            # {'ax': int(ax), 'ay': int(ay), 'bx': int(bx), 'by': int(by), 'cx': int(cx), 'cy': int(cy)}


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

            '''pixel drawing'''
            # screen space vertices of triangle
            A = triangle.screen_coordinates[0]
            B = triangle.screen_coordinates[1]
            C = triangle.screen_coordinates[2]

            # sort by increasing y values
            if B['y'] < A['y']: A, B = B, A
            if C['y'] < A['y']: A, C = C, A
            if C['y'] < B['y']: B, C = C, B

            # draw top triangle if not flat edge
            if B['y'] - A['y'] > 0:

                # AB line
                m_ab = (B['x'] - A['x']) / (B['y'] - A['y'])
                x_ab = A['x']
                mz_ab = (B['z'] - A['z']) / (B['y'] - A['y'])   
                z_ab = A['z']

                # AC line (to level of B)
                m_ac = (C['x'] - A['x']) / (C['y'] - A['y'])
                x_ac = A['x']
                mz_ac = (C['z'] - A['z']) / (C['y'] - A['y'])   
                z_ac = A['z']

                # left and right edge starting x values and gradients
                m_left = 0
                x_left = 0
                mz_left = 0
                z_left = 0

                m_right = 0
                x_right = 0
                mz_right = 0
                z_right = 0
                
                # set left and right sides
                if m_ab < m_ac:
                    m_left = m_ab
                    x_left = x_ab
                    mz_left = mz_ab
                    z_left = z_ab

                    m_right = m_ac
                    x_right = x_ac
                    mz_right = mz_ac
                    z_right = z_ac

                else:
                    m_left = m_ac
                    x_left = x_ac
                    mz_left = mz_ac
                    z_left = z_ac

                    m_right = m_ab
                    x_right = x_ab
                    mz_right = mz_ab
                    z_right = z_ab

                # loop over y rows
                for y in range(A['y'], B['y'] + 1):

                    # z (reciprocal) start value
                    z = z_left

                    # z gradient across row (if not a single pixel)
                    if x_left == x_right:
                        mz = 0
                    else:
                        mz = (z_right - z_left) / (x_right - x_left)

                    # loop over x across row
                    for x in range(int(x_left), int(x_right) + 1):

                        # check depth buffer
                        if self.depth_buffer[x, y] < z:

                            # draw pixel
                            pygame.gfxdraw.pixel(self.canvas, x, y, triangle.colour)

                            # update buffer
                            self.depth_buffer[x, y] = z

                        # increment z
                        z += mz

                    # increment x_left and x_right values
                    x_left += m_left
                    x_right += m_right

                    # increment z_left and z_right values
                    z_left += mz_left
                    z_right += mz_right

            # draw bottom triangle if not flat edge
            if C['y'] - B['y'] > 0:

                # BC line
                m_bc = (B['x'] - C['x']) / (B['y'] - C['y'])
                x_bc = B['x']
                mz_bc = (B['z'] - C['z']) / (B['y'] - C['y'])
                z_bc = B['z']

                # AC line (from level of B)
                m_ac = (C['x'] - A['x']) / (C['y'] - A['y'])
                x_ac = A['x'] + (B['y'] - A['y'])*m_ac
                mz_ac = (C['z'] - A['z']) / (C['y'] - A['y'])
                z_ac = A['z'] + (B['y'] - A['y'])*mz_ac

                # left and right edge starting x values and gradients
                m_left = 0
                x_left = 0
                mz_left = 0
                z_left = 0

                m_right = 0
                x_right = 0
                mz_right = 0
                z_right = 0
                
                # set left and right sides
                if x_bc < x_ac:
                    m_left = m_bc
                    x_left = x_bc
                    mz_left = mz_bc
                    z_left = z_bc

                    m_right = m_ac
                    x_right = x_ac
                    mz_right = mz_ac
                    z_right = z_ac
                else:
                    m_left = m_ac
                    x_left = x_ac
                    mz_left = mz_ac
                    z_left = z_ac

                    m_right = m_bc
                    x_right = x_bc
                    mz_right = mz_bc
                    z_right = z_bc

                # loop over y rows
                for y in range(B['y'], C['y'] + 1):

                    # z (reciprocal) start value
                    z = z_left

                    # z gradient across row (if not a single pixel)
                    if x_left == x_right:
                        mz = 0
                    else:
                        mz = (z_right - z_left) / (x_right - x_left)

                    # loop over x across row
                    for x in range(int(x_left), int(x_right) + 1):

                        # check depth buffer
                        if self.depth_buffer[x, y] < z:

                            # draw pixel
                            pygame.gfxdraw.pixel(self.canvas, x, y, triangle.colour)

                            # update buffer
                            self.depth_buffer[x, y] = z

                        # increment z
                        z += mz

                    # increment x_left and x_right values
                    x_left += m_left
                    x_right += m_right

                    # increment z_left and z_right values
                    z_left += mz_left
                    z_right += mz_right


            '''reset colour for testing '''
            # triangle.colour = (255, 255, 255)
            

        # blit surface to window
        self.window.blit(pygame.transform.scale(self.canvas, self.window.get_rect().size), (0, 0))
        self.framerate_counter()

        # update canvas
        pygame.display.flip()


    def load(self):
        '''Populate the scene with objects'''

        '''
        # test model
        self.model_list = [Cube(colour="random")]
        '''

        '''
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

        obj = Obj("Rasterizer/Models/cow.obj", scale=1, colour="random")
        self.model_list = [obj]

      
        '''
        # Load a grid of triangles that follow a specified heightmap

        # squares per side of grid
        M = 50
        N = 50
        # squares per model
        K = 5
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
    game = Game(window_width=700, window_height=700, canvas_width=100, canvas_height=100, view_width=3, view_height=3)
    game.run()

if __name__ == "__main__":
    main()