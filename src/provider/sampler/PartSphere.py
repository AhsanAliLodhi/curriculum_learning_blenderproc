import mathutils

from src.main.Provider import Provider
from src.provider.sampler.Sphere import Sphere


class PartSphere(Provider):
    """
    Samples a point from the surface or from the interior of solid sphere which is split in two parts.

    https://math.stackexchange.com/a/87238
    https://math.stackexchange.com/a/1585996

    Example 1: Sample a point from the surface of the sphere that is split by a plane with displacement of 0.5
    above center and a normal of [1, 0, 0].

    .. code-block:: yaml

        {
          "provider":"sampler.PartSphere",
          "center": [0, 0, 0],
          "part_sphere_vector": [1, 0, 0],
          "mode": "SURFACE",
          "distance_above_center": 0.5
        }

    **Configuration**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - center
          - Location of the center of the sphere.
          - mathutils.Vector
        * - radius
          - The radius of the sphere. dius of the sphere. Type: float
          - float
        * - mode
          - Mode of sampling. Determines the geometrical structure used for sampling. Available: SURFACE (sampling
            from the 2-sphere), INTERIOR (sampling from the 3-ball).
          - string
        * - distance_above_center
          - The distance above the center, which should be used. Default: 0.0 (half of the sphere).
          - float
        * - part_sphere_vector
          - The direction in which the sphere should be split, the end point of the vector, will be in the middle of
            the sphere pointing towards the middle of the resulting surface. Default: [0, 0, 1].
          - mathutils.Vector
    """

    def __init__(self, config):
        Provider.__init__(self, config)

    def run(self):
        """
        :param config: A configuration object containing the parameters necessary to sample.
        :return: A random point lying inside or on the surface of a solid sphere. Type: mathutils.Vector
        """
        # Center of the sphere.
        center = mathutils.Vector(self.config.get_list("center"))
        # Radius of the sphere.
        radius = self.config.get_float("radius")
        # Mode of operation.
        mode = self.config.get_string("mode")
        dist_above_center = self.config.get_float("distance_above_center", 0.0)
        part_sphere_dir_vector = self.config.get_vector3d("part_sphere_vector", [0, 0, 1])
        part_sphere_dir_vector.normalize()

        if dist_above_center >= radius:
            raise Exception("The dist_above_center value is bigger or as big as the radius!")

        while True:
            location = Sphere.sample(center, radius, mode)
            # project the location onto the part_sphere_dir_vector and get the length
            loc_in_sphere = location - center
            length = loc_in_sphere.dot(part_sphere_dir_vector)
            if length > dist_above_center:
                return location
