import bpy
import numpy as np

from src.main.Module import Module


class Dist2Depth(Module):
    """ Transforms Distance Image Rendered using Mist/Z pass to a depth image.

    **Configuration**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - depth_output_key
          - The key which should be used for storing the output data in a merged file. Default: 'depth'.
          - string
    """
    def __init__(self, config):
        Module.__init__(self, config)

    def run(self, dist, key, version):
        """
        :param dist: The distance data.
        :param key: The key used to store distance data.
        :param version: Version of the produced distance data.
        :return: The depth data, an appropriate key and version.
        """
        if len(dist.shape) > 2:
            dist = dist[:, :, 0] # All channles have the same value, so just extract any single channel
        else:
            dist = dist.squeeze()
        
        height, width = dist.shape

        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        max_resolution = max(width, height) 

        # Compute Intrinsics from Blender attributes (can change)
        f = width / (2 * np.tan(cam.angle / 2.))
        cx = (width - 1.0) / 2. - cam.shift_x * max_resolution
        cy = (height - 1.0) / 2. + cam.shift_y * max_resolution

        xs, ys = np.meshgrid(np.arange(dist.shape[1]), np.arange(dist.shape[0]))
        
        # coordinate distances to principal point
        x_opt = np.abs(xs-cx)
        y_opt = np.abs(ys-cy)

        # Solve 3 equations in Wolfram Alpha: 
        # Solve[{X == (x-c0)/f0*Z, Y == (y-c1)/f0*Z, X*X + Y*Y + Z*Z = d*d}, {X,Y,Z}]
        depth = dist * f / np.sqrt(x_opt**2 + y_opt**2 + f**2)
        output_key = self.config.get_string("depth_output_key", "depth")
        version = "1.0.0"
        return depth, output_key, version
