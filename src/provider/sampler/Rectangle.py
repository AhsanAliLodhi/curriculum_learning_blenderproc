import numpy as np
import mathutils

from src.main.Provider import Provider
import random
try:
    random = random.SystemRandom()
except Exception as e:
    import random


class Rectangle(Provider):
    """ Samples a point from a given a plane in 3D bounded by a ractangle.

    **Configuration**:

    .. csv-table::
       :header:, "Parameter", "Description"

       "vertices", "An array of three consecutive points which form the
                    rectangle Type: list[(x, y, z)]"
    """

    def __init__(self, config=None, vertices=None):
        # three  corners of bounded plane.
        if vertices is None and config is not None:
            vertices = self.config.get_list("vertices")            
        Provider.__init__(self, config)
        # Make sure vertices are an instance of Vector
        for idx, v in enumerate(vertices):
            if type(v) != mathutils.Vector:
                vertices[idx] = mathutils.Vector(v)
        # save the middle vertex as origin
        self.origin = vertices[1]
        # calculate the vectors from origin in direction of other two vertices
        self.vec1 = vertices[0] - vertices[1]
        self.vec2 = vertices[2] - vertices[1]

    def sample(self):
        """
        Samples a point from the bounded plane.
        :return: A location within the plane bounded by the rectangle.
                Type: mathutils.Vector
        """
        # sample the scaling factors at random
        r1 = random.random()
        r2 = random.random()
        # calculate the new location by adding the scaled vectors to origin
        location = self.origin + r1*self.vec1 + r2*self.vec2
        return location

    def run(self):
        """
        :param config: A configuration object containing the parameters
                        necessary to sample.
        :return: A location within the plane bounded by the rectangle.
                    plane. Type: Mathutils vector
        """
        location = self.sample()
        return location
