import os

import bpy

from src.provider.getter.Material import Material


class MaterialLoaderUtility(object):
    """
    This is the base class for different MaterialLoaders. It is currently used by the
    :class:`src.loader.HavenMaterialLoader` and the :class:`src.loader.CCMaterialLoader`.
    """

    x_texture_node = -1500
    y_texture_node = 300

    @staticmethod
    def find_cc_material_by_name(material_name: str, custom_properties: dict):
        """
        Finds from all loaded materials the cc material, which has the given material_name and the given
        custom_properties.

        :param material_name: Name of the searched material
        :param custom_properties: Custom properties, which have been assigned before
        :return: bpy.types.Material: Return the searched material, if not found returns None
        """
        # find used cc materials with this name
        cond = {"cp_is_cc_texture": True, "cp_asset_name": material_name}
        for key, value in custom_properties.items():
            cond[key] = value
        new_mats = Material.perform_and_condition_check(cond, [])
        if len(new_mats) == 1:
            new_mat = new_mats[0]
            return new_mat
        elif len(new_mats) > 1:
            raise Exception("There was more than one material found!")
        else:
            # the material was not even loaded
            return None

    @staticmethod
    def is_material_used(material: bpy.types.Material):
        """
        Checks if the given material is used on any object.

        :param material: Material, which should be checked
        :return: True if the material is used
        """
        # check amount of usage of this material
        return material.users != 0

    @staticmethod
    def create_new_cc_material(material_name: str, add_custom_properties: dict):
        """
        Creates a new material, which gets the given custom properties and the material name.

        :param material_name: The name of the material
        :param add_custom_properties: The custom properties, which should be added to the material
        :return: bpy.types.Material: Return the newly created material
        """
        # create a new material with the name of the asset
        new_mat = bpy.data.materials.new(material_name)
        new_mat["is_cc_texture"] = True
        new_mat["asset_name"] = material_name
        new_mat.use_nodes = True
        for key, value in add_custom_properties.items():
            if key.startswith("cp_"):
                cp_key = key[len("cp_"):]
            else:
                raise Exception("All cp have to start with cp_")
            new_mat[cp_key] = value
        return new_mat

    @staticmethod
    def create_image_node(nodes: bpy.types.Nodes, image_path: str, non_color_mode=False, x_location=0, y_location=0):
        """
        Creates a texture image node inside of a material.

        :param nodes: Nodes from the current material
        :param image_path: Path to the image which should be loaded
        :param non_color_mode: If this True, the color mode of the image will be "Non-Color"
        :param x_location: X Location in the node tree
        :param y_location: Y Location in the node tree
        :return: bpy.type.Node: Return the newly constructed image node
        """
        image_node = nodes.new('ShaderNodeTexImage')
        image_node.image = bpy.data.images.load(image_path, check_existing=True)
        if non_color_mode:
            image_node.image.colorspace_settings.name = 'Non-Color'
        image_node.location.x = x_location
        image_node.location.y = y_location
        return image_node

    @staticmethod
    def add_base_color(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, base_image_path,
                       principled_bsdf: bpy.types.Node):
        """
        Adds base color to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param base_image_path: Path to the base image
        :param principled_bsdf: Principled BSDF node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(base_image_path):
            base_color = MaterialLoaderUtility.create_image_node(nodes, base_image_path, False,
                                                                 MaterialLoaderUtility.x_texture_node,
                                                                 MaterialLoaderUtility.y_texture_node)
            links.new(base_color.outputs["Color"], principled_bsdf.inputs["Base Color"])
            return base_color
        return None

    @staticmethod
    def add_ambient_occlusion(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, ambient_occlusion_image_path,
                              principled_bsdf: bpy.types.Node, base_color: bpy.types.Node):
        """
        Adds ambient occlusion to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param ambient_occlusion_image_path: Path to the ambient occlusion image
        :param principled_bsdf: Principled BSDF node of the current material
        :param base_color: Base color node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(ambient_occlusion_image_path):
            ao_color = MaterialLoaderUtility.create_image_node(nodes, ambient_occlusion_image_path, True,
                                                               MaterialLoaderUtility.x_texture_node,
                                                               MaterialLoaderUtility.y_texture_node * 2)
            math_node = nodes.new(type='ShaderNodeMixRGB')
            math_node.blend_type = "MULTIPLY"
            math_node.location.x = MaterialLoaderUtility.x_texture_node * 0.5
            math_node.location.y = MaterialLoaderUtility.y_texture_node * 1.5
            math_node.inputs["Fac"].default_value = 0.333

            links.new(base_color.outputs["Color"], math_node.inputs[1])
            links.new(ao_color.outputs["Color"], math_node.inputs[2])
            links.new(math_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

            return ao_color
        return None

    @staticmethod
    def add_metal(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, metalness_image_path: str,
                  principled_bsdf: bpy.types.Node):
        """
        Adds metal to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param metalness_image_path: Path to the metal image
        :param principled_bsdf: Principled BSDF node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(metalness_image_path):
            metallic = MaterialLoaderUtility.create_image_node(nodes, metalness_image_path, True,
                                                               MaterialLoaderUtility.x_texture_node, 0)
            links.new(metallic.outputs["Color"], principled_bsdf.inputs["Metallic"])
            return metallic
        return None

    @staticmethod
    def add_roughness(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, roughness_image_path: str,
                      principled_bsdf: bpy.types.Node):
        """
        Adds roughness to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param roughness_image_path: Path to the metal image
        :param principled_bsdf: Principled BSDF node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(roughness_image_path):
            roughness_texture = MaterialLoaderUtility.create_image_node(nodes, roughness_image_path, True,
                                                                        MaterialLoaderUtility.x_texture_node,
                                                                        MaterialLoaderUtility.y_texture_node * -1)
            links.new(roughness_texture.outputs["Color"], principled_bsdf.inputs["Roughness"])
            return roughness_texture
        return None

    @staticmethod
    def add_specular(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, specular_image_path: str,
                     principled_bsdf: bpy.types.Node):
        """
        Adds specular to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param specular_image_path: Path to the metal image
        :param principled_bsdf: Principled BSDF node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(specular_image_path):
            specular_texture = MaterialLoaderUtility.create_image_node(nodes, specular_image_path, True,
                                                                       MaterialLoaderUtility.x_texture_node, 0)
            links.new(specular_texture.outputs["Color"], principled_bsdf.inputs["Specular"])
            return specular_texture
        return None

    @staticmethod
    def add_alpha(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, alpha_image_path: str,
                  principled_bsdf: bpy.types.Node):
        """
        Adds alpha to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param alpha_image_path: Path to the metal image
        :param principled_bsdf: Principled BSDF node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(alpha_image_path):
            alpha_texture = MaterialLoaderUtility.create_image_node(nodes, alpha_image_path, True,
                                                                    MaterialLoaderUtility.x_texture_node,
                                                                    MaterialLoaderUtility.y_texture_node * -2)
            links.new(alpha_texture.outputs["Color"], principled_bsdf.inputs["Alpha"])
            return alpha_texture
        return None

    @staticmethod
    def add_normal(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, normal_image_path: str,
                   principled_bsdf: bpy.types.Node, invert_y_channel: bool):
        """
        Adds normal to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param normal_image_path: Path to the metal image
        :param principled_bsdf: Principled BSDF node of the current material
        :param invert_y_channel: If this is True the Y Color Channel is inverted.
        :return: bpy.types.Node: The newly constructed texture node
        """
        normal_y_value = MaterialLoaderUtility.y_texture_node * -3
        if os.path.exists(normal_image_path):
            normal_texture = MaterialLoaderUtility.create_image_node(nodes, normal_image_path, True,
                                                                     MaterialLoaderUtility.x_texture_node,
                                                                     normal_y_value)
            if invert_y_channel:

                separate_rgba = nodes.new('ShaderNodeSeparateRGB')
                separate_rgba.location.x = 4.0 / 5.0 * MaterialLoaderUtility.x_texture_node
                separate_rgba.location.y = normal_y_value
                links.new(normal_texture.outputs["Color"], separate_rgba.inputs["Image"])

                invert_node = nodes.new("ShaderNodeInvert")
                invert_node.inputs["Fac"].default_value = 1.0
                invert_node.location.x = 3.0 / 5.0 * MaterialLoaderUtility.x_texture_node
                invert_node.location.y = normal_y_value

                links.new(separate_rgba.outputs["G"], invert_node.inputs["Color"])

                combine_rgba = nodes.new('ShaderNodeCombineRGB')
                combine_rgba.location.x = 2.0 / 5.0 * MaterialLoaderUtility.x_texture_node
                combine_rgba.location.y = normal_y_value
                links.new(separate_rgba.outputs["R"], combine_rgba.inputs["R"])
                links.new(invert_node.outputs["Color"], combine_rgba.inputs["G"])
                links.new(separate_rgba.outputs["B"], combine_rgba.inputs["B"])

                current_output = combine_rgba.outputs["Image"]
            else:
                current_output = normal_texture.outputs["Color"]

            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.inputs["Strength"].default_value = 1.0
            normal_map.location.x = 1.0 / 5.0 * MaterialLoaderUtility.x_texture_node
            normal_map.location.y = normal_y_value
            links.new(current_output, normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], principled_bsdf.inputs["Normal"])
            return normal_texture
        return None

    @staticmethod
    def add_bump(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, bump_image_path: str,
                 principled_bsdf: bpy.types.Node):
        """
        Adds bump to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param bump_image_path: Path to the metal image
        :param principled_bsdf: Principled BSDF node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        bump_y_value = MaterialLoaderUtility.y_texture_node * -3
        if os.path.exists(bump_image_path):
            bump_texture = MaterialLoaderUtility.create_image_node(nodes, bump_image_path, True,
                                                                   MaterialLoaderUtility.x_texture_node,
                                                                   bump_y_value)
            bump_map = nodes.new("ShaderNodeBumpMap")
            bump_map.inputs["Strength"].default_value = 1.0
            bump_map.location.x = 1.0 / 5.0 * MaterialLoaderUtility.x_texture_node
            bump_map.location.y = bump_y_value
            links.new(bump_texture.outputs["Color"], bump_map.inputs["Heights"])
            links.new(bump_map.outputs["Normal"], principled_bsdf.inputs["Normal"])
            return bump_texture
        return None

    @staticmethod
    def add_displacement(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, displacement_image_path: str,
                         output_node: bpy.types.Node):
        """
        Adds bump to the principled bsdf node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param displacement_image_path: Path to the metal image
        :param output_node: Output node of the current material
        :return: bpy.types.Node: The newly constructed texture node
        """
        if os.path.exists(displacement_image_path):
            displacement_texture = MaterialLoaderUtility.create_image_node(nodes, displacement_image_path, True,
                                                                           MaterialLoaderUtility.x_texture_node,
                                                                           MaterialLoaderUtility.y_texture_node * -4)
            displacement_node = nodes.new("ShaderNodeDisplacement")
            displacement_node.inputs["Midlevel"].default_value = 0.5
            displacement_node.inputs["Scale"].default_value = 0.15
            displacement_node.location.x = MaterialLoaderUtility.x_texture_node * 0.5
            displacement_node.location.y = MaterialLoaderUtility.y_texture_node * -4
            links.new(displacement_texture.outputs["Color"], displacement_node.inputs["Height"])
            links.new(displacement_node.outputs["Displacement"], output_node.inputs["Displacement"])
            return displacement_texture
        return None

    @staticmethod
    def connect_uv_maps(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks, collection_of_texture_nodes: list):
        """
        Connect all given texture nodes to a newly constructed UV node.

        :param nodes: Nodes from the current material
        :param links: Links from the current material
        :param collection_of_texture_nodes: List of :class: `:w
        bpy.type.Node of type :class: `ShaderNodeTexImage`
        """
        if len(collection_of_texture_nodes) > 0:
            texture_coords = nodes.new("ShaderNodeTexCoord")
            texture_coords.location.x = MaterialLoaderUtility.x_texture_node * 1.4
            mapping_node = nodes.new("ShaderNodeMapping")
            mapping_node.location.x = MaterialLoaderUtility.x_texture_node * 1.2

            links.new(texture_coords.outputs["UV"], mapping_node.inputs["Vector"])
            for texture_node in collection_of_texture_nodes:
                if texture_node is not None:
                    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])
