import os

import bpy

from src.loader.LoaderInterface import LoaderInterface
from src.utility.Config import Config
from src.utility.Utility import Utility


class RockEssentialsGroundConstructor(LoaderInterface):
    """
    Constructs a ground plane with a material using RE PBR Rock Shader.

    Example 1: Construct a scaled ground plane with 30 subdivision cuts, custom name and subdiv level value for
    rendering using PBR Rock Shader from the specified .blend file.

    .. code-block:: yaml

        {
          "module": "constructor.RockEssentialsGroundConstructor",
          "config": {
            "tiles": [
            {
              "shader_path": "<args:0>/Rock Essentials/Individual Rocks/Volcanic/Rocks_Volcanic_Small.blend",
              "plane_scale": [50, 50, 1],
              "subdivision_cuts": 30,
              "subdivision_render_levels": 2,
              "tile_name": "Gr_Plane_1"
            }
            ]
          }
        }

    **Configuration**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - tiles
          - Ground tiles to create, each cell contains a separate tile configuration.
          - list

    **Ground plane properties**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - shader_path
          - Path to a .blend file that containing PBR Rock Shader in //NodeTree// section.
          - string
        * - plane_scale
          - Scale of a ground plane. Default: [10, 10, 1].
          - mathutils.Vector/list
        * - subdivision_cuts
          - Amount of cuts along each plane axis. Default: 50.
          - int
        * - subdivision_render_levels
          - Render level for a plane's subdivision modifier. Default: 3.
          - int
        * - tile_name
          - Name of the ground tile. Set one if you plan to use this module multiple times in one config. Default:
            'RE_ground_plane'.
          - string
    """

    def __init__(self, config):
        LoaderInterface.__init__(self, config)

    def run(self):
        """ Constructs a ground plane.
            1. Get configuration parameters.
            2. Load shader.
            3. Construct ground plane and it's material node tree.
        """

        tiles = self.config.get_list("tiles")
        for tile in tiles:
            if tile:
                ground_config = Config(tile)
                self._load_shader(ground_config)
                self._construct_ground_plane(ground_config)

    def _load_shader(self, ground_config):
        """ Loads PBR Rock Shader

        :param ground_config: Config object that contains user-defined settings for ground plane. Type: Config.
        """
        shader_path = ground_config.get_string("shader_path")
        bpy.ops.wm.append(filepath=os.path.join(shader_path, "/NodeTree", "", "PBR Rock Shader"),
                          filename="PBR Rock Shader", directory=os.path.join(shader_path+"/NodeTree"))

    def _construct_ground_plane(self, ground_config):
        """ Constructs a ground plane.

        :param ground_config: Config object that contains user-defined settings for ground plane. Type: Config.
        """
        # get scale\'size' of a plane to be created 10x10 if not defined
        plane_scale = ground_config.get_vector3d("plane_scale", [1, 1, 1])
        # get the amount of subdiv cuts, 50 (50x50=250 segments) if not defined
        subdivision_cuts = ground_config.get_int("subdivision_cuts", 50)
        # get the amount of subdiv render levels, 2 if not defined
        subdivision_render_levels = ground_config.get_int("subdivision_render_levels", 3)
        # get name, 'RE_ground_plane' if not defined
        tile_name = ground_config.get_string("tile_name", "RE_ground_plane")

        # create new plane, set its size
        bpy.ops.mesh.primitive_plane_add()
        bpy.context.object.name = tile_name
        plane_obj = bpy.data.objects[tile_name]
        plane_obj.scale = plane_scale

        # create new material, enable use of nodes
        mat_obj = bpy.data.materials.new(name="re_ground_mat")
        mat_obj.use_nodes = True

        # set material
        plane_obj.data.materials.append(mat_obj)
        nodes = mat_obj.node_tree.nodes
        links = mat_obj.node_tree.links

        # delete Principled BSDF node
        nodes.remove(Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled"))

        # create PBR Rock Shader, connect its output 'Shader' to the Material Output nodes input 'Surface'
        group_pbr = nodes.new("ShaderNodeGroup")
        group_pbr.node_tree = bpy.data.node_groups['PBR Rock Shader']
        output_node = Utility.get_the_one_node_with_type(nodes, 'OutputMaterial')
        links.new(group_pbr.outputs['Shader'], output_node.inputs['Surface'])

        # create Image Texture nodes for color, roughness, reflection, and normal maps
        self._create_node(nodes, links, 'color', 'Color')
        self._create_node(nodes, links, 'roughness', 'Roughness')
        self._create_node(nodes, links, 'reflection', 'Reflection')
        self._create_node(nodes, links, 'normal', 'Normal')

        # create subsurface and displacement modifiers
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=subdivision_cuts)
        bpy.ops.object.modifier_add(type="DISPLACE")
        bpy.ops.object.modifier_add(type="SUBSURF")

        # create new texture
        texture_name = tile_name + "_texture"
        bpy.data.textures.new(name=texture_name, type="IMAGE")

        # set new texture as a displacement texture, set UV texture coordinates
        plane_obj.modifiers['Displace'].texture = bpy.data.textures[texture_name]
        plane_obj.modifiers['Displace'].texture_coords = 'UV'

        bpy.ops.object.editmode_toggle()
        # scale, set render levels for subdivision, strength of displacement and set passive rigidbody state
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.context.object.modifiers["Subdivision"].render_levels = subdivision_render_levels
        plane_obj["physics"] = False
        self._set_properties([plane_obj])

    def _create_node(self, nodes, links, map_type, in_point):
        """ Handles the creation a ShaderNodeTexImage node, setting maps and creating links.

        :param nodes: All nodes in the node tree of the material object. Type: Nodes.
        :param links: All links in the node tree of the material object. Type: NodeLinks.
        :param map_type: Type of image/map that will be assigned to this node. Type: string.
        :param in_point: Name of an input point in PBR Rock Shader node to use. Type: string.
        """
        nodes.new('ShaderNodeTexImage')
        # set output point of the node to connect
        a = nodes.get(nodes[-1].name).outputs['Color']
        nodes[-1].label = map_type
        # special case for a normal map since the link between TextureNode and PBR RS is broken with Normal Map node
        if map_type == 'normal':
            # create new node
            group_norm_map = nodes.new('ShaderNodeNormalMap')
            # magic: pre-last node, select Color output
            a_norm = nodes.get(nodes[-2].name).outputs['Color']
            # select input point
            b_norm = group_norm_map.inputs['Color']
            # connect them
            links.new(a_norm, b_norm)
            # redefine main output point to connect
            a = nodes.get(nodes[-1].name).outputs['Normal']
        # select main input point of the PBR Rock Shader
        b = nodes.get("Group").inputs[in_point]
        links.new(a, b)
