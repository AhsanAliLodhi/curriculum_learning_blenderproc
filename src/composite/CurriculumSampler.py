from src.main.Module import Module
from src.utility.Utility import Utility
import os
from src.utility.Curriculum import scale_obj, make_bg, replicate, \
                                    ObjectInstanceProvider, \
                                    set_dimensions_obj,\
                                    make_real_bg,\
                                    get_scale_range, get_fov_dimensions, \
                                    get_2D_bounds, bounds_to_vertices, \
                                    point_to, get_obj_size, scale_normalized_obj

from src.provider.sampler.Rectangle import Rectangle as RectangleSampler
from src.provider.sampler.Icosphere import Icosphere as IcosphereSampler
from src.utility.Polygon import Polygon
from src.renderer.RgbRenderer import RgbRenderer
from src.renderer.SegMapRenderer import SegMapRenderer
from src.writer.CocoAnnotationsWriter import CocoAnnotationsWriter
from src.utility.BlenderUtility import load_image, get_all_mesh_objects

import bpy
import random
import math
try:
    random = random.SystemRandom()
except Exception as e:
    import random


class CurriculumSampler(Module):
    """
    This composite module (CameraSampler & Manupilator) takes uses loaded
    foreground objects and background objects and uses techniques mentioned
    in https://arxiv.org/abs/1902.09967 (Curriculum Learning) to make
    sythetic image training dataset.

    **Configuration**:

    .. csv-table::
       :header: "Parameter", "Description"

        :fg_objs: list of foreground objects
        :bg_objs: list of background objects
    """

    def __init__(self, config):
        Module.__init__(self, config)
        self.config = config
        self.scale_set = config.get_list("scales", [])
        self.is_bg_valid = False
        self.cropping_allowed = config.get_float("cropping_allowed", 0.5)
        self.intersection_allowed = config.get_float("intersection_allowed", 0.3)
        self.occlusion_chance = config.get_float("occlusion_chance", 0.8)
        self.min_occlusion = config.get_float("min_occlusion", 0.2)
        self.max_occlusion = config.get_float("max_occlusion", 0.5)
        self.no_of_layers = config.get_int("no_of_layers", 3)
        self.num_inplane_rotations = config.get_int("num_inplane_rotations", 8)
        self.max_fobjs_in_scene = config.get_int("max_fobjs_in_scene", 5)
        self.dropout = config.get_float("dropout", 0.2)  # TODO: better name
        self.fobj_placement_max_tries = \
            config.get_int("fobj_placement_max_tries", 20)
        self.oobj_placement_max_tries = \
            config.get_int("oobj_placement_max_tries", 100)
        self.icosphereSampler = IcosphereSampler(min_n_points=12, radius=8)
        self.disable_occlusions = \
            config.get_bool("disable_occlusions", True)
        self.real_backgrounds = \
            config.get_string("real_backgrounds", "None")
        if self.real_backgrounds == "None":
            self.real_backgrounds = None

        self.last_frame_to_render = config.get_int("last_frame_to_render", -1)
        self.background_light_energy = config.get_int("background_light_energy", 0)
        self.foreground_light_energy = 0

        self.fg_object_location = [0.0, 0.0, 0.0]
        
        self.images_for_bg = config.get_string("images_for_bg", None)
 

    def init(self):
        config = self.config
        #set origin to center of mass for all
        origin_to_center()
        # Get foreground objects
        self.fg_objs = config.get_list("fg_objs", [])
        # Make a provider which manages copies
        self.fg_objs_provider = ObjectInstanceProvider(self.fg_objs, (0, 0, 0))
        print("-------------config-------------------------")
        print(config)
        print("-------------self.fg_objs-------------------------")
        print(self.fg_objs)
        # Get background objects
        self.bg_objs = config.get_list("bg_objs", [])
        # Make a provider which manages copies, 0 is class for background
        self.bg_objs_provider = ObjectInstanceProvider(self.bg_objs, (100, 100,
                                                       100), default_class=0)

        # Disable all shadows (for cycles renderer only)
        self.all_source_object_names = []
        self.all_source_objects = []
        for obj in self.fg_objs+self.bg_objs:
            obj.cycles_visibility.shadow = False
            self.all_source_object_names.append(obj.name)
            

        # Hide all foreground objects 
        for obj in self.fg_objs:
            obj.location = (50, 0, 0)
            obj.hide_render = True
            self.all_source_objects.append(obj)

        # Set number of classes for segmaprenderer
        # foreground objects + background
        bpy.context.scene["num_labels"] = len(self.fg_objs) + 1
        # Set class of world as background as well
        bpy.context.scene.world["category_id"] = 0
        self.render = RgbRenderer(self.config)
        self.seg_map_renderer = SegMapRenderer(self.config)
        self.coco_writer = CocoAnnotationsWriter(self.config)

    def run(self):
        self.init()
        scene = bpy.context.scene
        scene.cycles.device = 'GPU'
        camera = bpy.context.scene.camera
        cam_location = camera.location.copy()
        cam_rotation_euler = camera.rotation_euler.copy()
        current_frame = 1

        # Get light name
        bpy.ops.object.select_all(action='DESELECT')
        fg_light = bpy.data.objects['light']
        fg_light.select_set(True)
        self.foreground_light_energy = fg_light.data.energy

        print("=================================================")
        print(self.fg_objs)
        print(self.bg_objs)
        print(self.scale_set)
        print("====================RUNNING======================")

        # create a new 1x1x1 cube in order to get scale value
        bpy.ops.mesh.primitive_cube_add(location=(120,120,120),size=1)
        cube = bpy.context.selected_objects[0]
        cube.name = "cube1x1"
        cube["category_id"] = 0
        self.all_source_object_names.append(cube.name)

        self.reset_objs(self.bg_objs, location=(100, 100, 100), hide=False)
        for scale in self.scale_set:
            print("====================NEW SCALE======================")
            print("####### Rendering SCALE: ", scale)

            #Get scale for fg objects
            cube.location = (0, 0, 0)
            cube.hide_render = False
            bpy.context.view_layer.update()
            fg_scale_range, fg_scale = self.get_scale_range_from_cube(scale, cube, scene, camera)
            
            dist_to_fg = (cube.location - camera.location).length * 2
            bpy.context.view_layer.update() 
            print("FG scale:", fg_scale)

            #Move etalon cube to bg position to get scale for bg objects
            cube.location = - camera.location
            bpy.context.view_layer.update()  
            bg_scale_range, bg_scale = self.get_scale_range_from_cube(scale, cube, scene, camera)
            print("BG scale:", bg_scale)
            cube.hide_render = True
            
            #cube.location = (120, 120, 120)
            bpy.context.view_layer.update() 

            for f_idx, f_obj in enumerate(self.fg_objs):
                print("====================NEW OBJECT======================")
                print("####### Rendering FG object: ", f_obj.name)  
                camera.location = cam_location
                camera.rotation_euler = cam_rotation_euler #[1.5708,0,1.5708]
                 
                current_frame = 0
                # clear the taken status for front and background objects
                self.fg_objs_provider.clear()
                self.bg_objs_provider.clear()
                # set the fobj in focus as taken
                # (this makes sure correct management of foreground objects
                # when and if producing more foreground objects)
                self.fg_objs_provider.set_taken(f_obj)
                # reset scale and position of all fobjs
                self.reset_objs(self.fg_objs)
                bpy.context.view_layer.update()
                is_height = False

                # create a background for sampled scale
                if self.images_for_bg is not None:
                        real_image_path = random.choice(
                                            os.listdir(self.images_for_bg))
                        real_image_path = os.path.join(self.images_for_bg,
                                                real_image_path)

                print("Adding real image: ", real_image_path)
                print("MAKING BACKGROUND")

                corrected_dist_to_fg = dist_to_fg/2 * scale + dist_to_fg
                if self.real_backgrounds is None:
                    bg, bg_light = make_bg(self.bg_objs, f_obj, f_idx, camera, scene,
                                 scale_range=bg_scale_range,
                                 bg_scale=bg_scale,
                                 no_of_layers=self.no_of_layers,
                                 dropout=self.dropout,
                                 dist=corrected_dist_to_fg,
                                 real_image_path = real_image_path,
                                 light_energy=self.background_light_energy)
                else:
                    image_path = random.choice(
                                        os.listdir(self.real_backgrounds))
                    image_path = os.path.join(self.real_backgrounds,
                                              image_path)
                    bg = make_real_bg(image_path, f_obj, camera, scene)
                
                # Set class to 0, for background
                bg["category_id"] = 0
                print("BACKGROUND COMPLETE", bg.name)

                bg.hide_render = False
                bpy.context.view_layer.update()
                
                # sample the fg objects to render
                fobjs_sampled = \
                    self.sample_fg_objects(f_obj, f_idx, self.fg_objs,
                                           camera, scene)
                # try to place as many sampled objects as possible
                print("ADDING FOREGROUND OBJECTS")
                placed_objs, current_polys = \
                    self.scale_and_place_fg_objs(fobjs_sampled,
                                                 fg_scale_range,
                                                 is_height, scene, camera,
                                                 verbose=True)

                # for each object placed successfully
                fobjs_torender = []
                for r_idx, fobj_torender in enumerate(fobjs_sampled):
                    if placed_objs[r_idx]:
                        # add it to objects we will render
                        fobjs_torender.append(fobj_torender)
                        # and set renderability to true
                        fobj_torender.hide_render = False
                    else:
                        # else set renderability to false
                        fobj_torender.hide_render = True
                if not self.disable_occlusions:
                    print("MAKING OCCLUSION LAYER")
                    # now make an occlusion layer
                    o_objs = self.generate_occlusions(fobjs_torender,
                                                      fg_scale_range,
                                                      is_height, current_polys,
                                                      camera, scene)
                print("FREEZING FRAME ", current_frame)

                # now for each outplane rotation
                for point_to_look_at in self.icosphereSampler.next():
                    # point_to_look_at is 0 centered, so translate it to
                    # center the point around object center
                    # then rotate object to look at the point
                    main_obj = fobjs_sampled[0]
                    point_to(main_obj, point_to_look_at +
                             main_obj.location)
                    bpy.context.view_layer.update()
                    step_inplane_rotation = ((2*math.pi) /
                                             self.num_inplane_rotations)
                    # Reset cemera Y rotation 
                    camera.rotation_euler[1] = 0
                    distance_to_camera = (cube.location - camera.location).length
                    # for each inplane rotation
                    for _ in range(self.num_inplane_rotations):
                        
                        # freeze current state
                        camera.keyframe_insert(data_path="rotation_euler",
                                               frame=current_frame)
                        if self.real_backgrounds is None:
                            self.set_random_light(fg_light, bg_light, distance_to_camera)
                            bpy.context.view_layer.update()
                            self.freeze_current_state([fg_light, bg_light], current_frame, light=True)
                            
                        self.freeze_current_state([bg], current_frame)
                        self.freeze_current_state(fobjs_torender, current_frame)
                        if not self.disable_occlusions:
                            self.freeze_current_state(o_objs, current_frame)

                        # inplane rotate camera
                        current_frame += 1
                        camera.rotation_euler[1] += step_inplane_rotation

                cube.hide_render = True
                bpy.context.scene.frame_start = 0
                if self.last_frame_to_render is not None and self.last_frame_to_render > 0:
                    bpy.context.scene.frame_end = self.last_frame_to_render
                else:
                    bpy.context.scene.frame_end = current_frame

                #Rendering RGB frames
                print('#### Rendering frames: {}-{} '.format(bpy.context.scene.frame_start, bpy.context.scene.frame_end))
                file_prefix = "obj{}_sc{}_".format(f_idx, int(scale*100)) 
                self.render.run_and_continue(file_prefix)

                #Rendering segmentation frames
                seg_file_prefix = "seg_obj{}_sc{}_".format(f_idx, int(scale*100))
                self.seg_map_renderer.run_curr(self.all_source_object_names)
                nodes = bpy.context.scene.world.node_tree.nodes
                nodes.get("Background").inputs['Color'].default_value = [0.6,0.6,0.6,0]
 
                #Writing coco annotation data
                file_postfix = "_obj{}_sc{}".format(f_idx, int(scale*100)) 
                self.coco_writer.run(file_postfix)
 
                #remove generated BG objects
                bpy.ops.object.select_all(action='DESELECT')
                bg.select_set(True)
                bpy.ops.object.delete()

                if self.real_backgrounds is None:
                    bg_light.select_set(True)
                    bpy.ops.object.delete()

                #delete created objects
                objs_with_mats = get_all_mesh_objects()

                for obj in objs_with_mats:
                    if obj.name not in self.all_source_object_names:
                        print("Deleting FG:",obj.name)
                        bpy.data.objects[obj.name].select_set(True)
                        bpy.ops.object.delete()  

                # Remove all not used data
                for block in bpy.data.meshes:
                    if block.users == 0:
                        bpy.data.meshes.remove(block)

                for block in bpy.data.materials:
                    if block.users == 0:
                        bpy.data.materials.remove(block)

                for block in bpy.data.textures:
                    if block.users == 0:
                        bpy.data.textures.remove(block)

                for block in bpy.data.images:
                    if block.users == 0:
                        bpy.data.images.remove(block) 

                for key in bpy.context.scene.keys():
                    del bpy.context.scene[key]




        bpy.context.scene.frame_end = current_frame


    def _remove_orphan_data(self):
        """ Remove all data blocks which are not used anymore. """
        data_structures = [
            bpy.data.meshes,
            bpy.data.materials,
            bpy.data.textures,
            bpy.data.images,
            bpy.data.brushes,
            bpy.data.actions,
            bpy.data.lights
        ]

        for data_structure in data_structures:
            for block in data_structure:
                # If no one uses this block => remove it
                if block.users == 0:
                    data_structure.remove(block)

    def freeze_current_state(self, objs, frame_key_id, light=False):
        for obj in objs:
            # freeze location
            obj.keyframe_insert(data_path="location", frame=frame_key_id)
            # freeze rotation
            obj.keyframe_insert(data_path="rotation_euler", frame=frame_key_id)
            # freeze scale
            obj.keyframe_insert(data_path="scale", frame=frame_key_id)
            # freeze rendering status
            obj.keyframe_insert(data_path="hide_render", frame=frame_key_id)
            if light:
                #obj.keyframe_insert(data_path="energy", frame=frame_key_id)
                obj.data.keyframe_insert(data_path='energy', frame=frame_key_id)

    def freeze_default_state(self, objs, frame_key_id, location=(0, 0, 0),
                             renderable=False):
        # store previous states
        locations = []
        renderables = []
        # move to base states
        for obj in objs:
            locations.append(obj.location.copy())
            renderables.append(not obj.hide_render)
            obj.location = location
            obj.hide_render = not renderable
        # update view
        bpy.context.view_layer.update()
        # save keyframes
        self.freeze_current_state(objs, frame_key_id)
        # reset values
        for idx, obj in enumerate(objs):
            obj.location = locations[idx]
            obj.hide_render = not renderables[idx]
        # update view
        bpy.context.view_layer.update()

    def generate_occlusions(self, fobjs_torender, scale_range, is_height,
                            current_polys, camera, scene):
        o_polys = {}
        o_objs = []
        for r_idx, obj in enumerate(fobjs_torender):
            # occlude with some probability
            if random.uniform(0, 1) < self.occlusion_chance:
                continue
            # take a random background object to occlude with
            # put it at origin
            o_obj = self.bg_objs_provider.get_random()
            o_objs.append(o_obj)
            bpy.context.view_layer.update()
            # scale accordinlgy
            scale = random.uniform(scale_range[0], scale_range[1])
            scale_obj(o_obj, scale, is_height)
            # rotate at random
            o_obj.rotation_euler = (random.uniform(-1, 1),
                                    random.uniform(-1, 1),
                                    random.uniform(-1, 1))
            max_distance = max(o_obj.dimensions) + max(obj.dimensions)
            fobj_dist = (fobjs_torender[0].location - camera.location).length
            _, bounds = get_fov_dimensions(camera, fobj_dist, scene, 0)
            sampler = RectangleSampler(vertices=bounds)
            tries_left = self.oobj_placement_max_tries
            while(tries_left):
                possible_loc = sampler.sample()  # sample a point on the plane
                point_dist = (possible_loc - obj.location).length
                if point_dist > max_distance:  # if point too far away
                    tries_left -= 1
                    continue  # resample
                o_obj.location = possible_loc
                bpy.context.view_layer.update()
                obj_vertices = Polygon.sort_points_counterclockwise(
                    bounds_to_vertices(
                        get_2D_bounds(scene, camera, o_obj)))
                o_polys[o_obj.name] = Polygon(obj_vertices)
                intersecting_poly = o_polys[o_obj.name]. \
                    get_intersecting_polygon(current_polys[r_idx])
                intersecting_portion = 0
                if intersecting_poly is not None:
                    intersecting_portion = (intersecting_poly.get_area() /
                                            current_polys[r_idx].get_area())
                if (intersecting_portion < self.min_occlusion) or\
                   (intersecting_portion > self.max_occlusion):
                    tries_left -= 1
                    continue
                break
            # if the object survives it all now lets more it above the
            # foreground and scale it down
            if tries_left > 0:
                # we move it half the distance up
                normal = (camera.location - o_obj.location)
                dist_prev = normal.length
                normal.normalize()
                # in the direction of normal of foreground object plane
                o_obj.location += (dist_prev/2)*normal
                o_obj.scale = o_obj.scale/2
                bpy.context.view_layer.update()
                # render to true
                o_obj.hide_render = False
            else:
                # render to false
                o_obj.hide_render = True
        return o_objs

    def reset_objs(self, fg_objs, location=None, hide=True):
        if location is None:
            location = (50, 0, 0)
        # set all objs to origin and hide them
        for obj in fg_objs:
            obj.location = location
            # scale down object to 1m by wrt highest dimension
            set_dimensions_obj(obj, 1)
            obj.hide_render = hide

    def sample_scale_range(self, scale, f_obj, camera, scene):
        # sample a scale range
        max_scale, min_scale, _scale, \
            _, _, _, is_height = \
            get_scale_range(scale, f_obj, camera, scene)
        _range = get_scale_range(scale, f_obj, camera, scene)
        scale_obj(f_obj, _scale, is_height)
        bound1 = random.uniform(min_scale, max_scale)
        bound2 = random.uniform(min_scale, max_scale)
        scaled_scale_range = (bound1*1.5, bound2*1.5)
        base_scale_range = (min_scale, max_scale)
        return scaled_scale_range, base_scale_range, _range[2], _range[6]


    def get_scale_range_from_cube(self, scale, obj, scene, camera, min_scale=0.8,
                             max_scale=1.1):
 
        base_dist = (obj.location - camera.location).length

        width, height, _, _ = get_obj_size(obj, camera, scene, dist=base_dist, do_clamp=False)
        is_height = height > width
        obj_size_in_image = height if is_height else width

        _scale = scale/obj_size_in_image

        med_scale = (min_scale + max_scale) /2
        bound1 = random.uniform(min_scale * _scale,  med_scale *_scale)
        bound2 = random.uniform(med_scale*_scale, max_scale * _scale)
        scale_range = (bound1, bound2)
 
        return scale_range, _scale

    def sample_fg_objects(self, f_obj, f_idx, fg_objs, camera,
                          scene):
        
        new_obj = replicate(f_obj, 1, 0, new_loc=(0, 0, 0))[0]
        fobjs_torender = [new_obj]
        # select number of object at random
        no_objs_in_scene = random.randint(0, self.max_fobjs_in_scene)
        print("======Adding {} fg objects=======", no_objs_in_scene +1)
        # add random many foreground objects
        for _ in range(no_objs_in_scene -1):
            # pick a random object
            o_idx = random.randint(0, len(self.all_source_objects)-1)
            new_obj = replicate(self.all_source_objects[o_idx], 1, 0, new_loc=(0, 0, 0))[0]
            fobjs_torender.append(new_obj)
        bpy.context.view_layer.update()
        return fobjs_torender

    def scale_and_place_fg_objs(self, fobjs_torender, base_scale_range,
                                is_height, scene, camera, verbose=False):
        base_polys = []
        current_polys = []
        placed_objs = [False] * len(fobjs_torender)
        fobj_dist = (fobjs_torender[0].location - camera.location).length
        _, bounds = get_fov_dimensions(camera, fobj_dist, scene, 0)
        bounds = [bounds[0], bounds[1], bounds[3], bounds[2]]
        sampler = RectangleSampler(vertices=bounds)
        for r_idx, obj in enumerate(fobjs_torender):
            # rescale the object to render
            fscale = random.uniform(base_scale_range[0], base_scale_range[1])
            scale_normalized_obj(obj, fscale)
            # assigning base poly, for now as the object is in center and
            # hence fully exposed to camera
            base_obj_vertices = \
                Polygon.sort_points_counterclockwise(
                    bounds_to_vertices(
                        get_2D_bounds(scene, camera, obj)))
            base_polys.append(Polygon(base_obj_vertices))
            # Now for certiain times, try to place object without violating
            # any rules
            tries_left = self.fobj_placement_max_tries
            while(tries_left and placed_objs[r_idx] is False):
                # place at a new position
                pos = sampler.sample()
                obj.location = pos
                bpy.context.view_layer.update()
                # calculate the new poly
                obj_vertices = Polygon.sort_points_counterclockwise(
                    bounds_to_vertices(
                        get_2D_bounds(
                            scene, camera, obj)))
                # if its the first try placing
                if r_idx < len(current_polys):
                    current_polys[r_idx] = Polygon(obj_vertices)
                # else replace instead
                else:
                    current_polys.append(Polygon(obj_vertices))
                # check if all rules are being followed
                # 1: crop rule
                base_polys_area = base_polys[r_idx].get_area()
                base_polys_area = 0.0000001 if base_polys_area == 0 else base_polys_area

                crop_rule = (current_polys[r_idx].get_area() / base_polys_area) > (1 - self.cropping_allowed)
                if not crop_rule and verbose:
                    print("failed crop rule, visible area = ",
                          current_polys[r_idx].get_area() /
                          base_polys_area, pos)
                elif crop_rule and verbose:
                    print("crop rule passed, visible area = ",
                          current_polys[r_idx].get_area() /
                          base_polys_area)
                # 2: overlap rule
                overlap_rule = True
                # Check overlap with all previously placed objects
                for p_idx in range(len(base_polys) - 1):
                    # compute the intersecting poly
                    intersecting_poly = current_polys[p_idx]. \
                        get_intersecting_polygon(current_polys[-1])
                    intersecting_portion = 0
                    # if found an intersecting poly then calculate
                    # intersecting portion
                    if (intersecting_poly is not None and
                            intersecting_poly.get_area() > 0):
                        intersecting_portion = (intersecting_poly.get_area() /
                                                current_polys[-1].get_area())
                        # if intersection is more than allowed, then rule fails
                        overlap_rule = (overlap_rule and
                                        (intersecting_portion <
                                         self.intersection_allowed))
                    if not overlap_rule and verbose:
                        print("failed overlap rule")
                        print(fobjs_torender[p_idx], 'v', obj, " overlap:",
                              round(intersecting_portion, 2))
                    elif overlap_rule and verbose:
                        print("overlap rule not violated with ",
                              fobjs_torender[p_idx], " at only ",
                              round(intersecting_portion, 2), "intersection")
                if crop_rule and overlap_rule and verbose:
                    print("marking ", obj, " placed!!")
                    # mark the object succesfully placed
                    placed_objs[r_idx] = True
                elif verbose:
                    print("going to place again")
                tries_left -= 1
            if tries_left < 1 and verbose:
                print("couldnt place ", obj, " moving on to next object")
        return placed_objs, current_polys

    def set_random_light(self, fg_light, bg_light, distance_to_camera):
 
        # Set random light energy
        light_intensity = random.uniform(0, 1)
        fg_light.data.energy = (light_intensity + 0.1) * self.foreground_light_energy

        # Change light position
        fg_light.location[1] = random.uniform(-distance_to_camera, distance_to_camera)
        fg_light.location[2] = random.uniform(-distance_to_camera, distance_to_camera)
        bg_light.data.energy = light_intensity * self.background_light_energy

def origin_to_center():    
    # deselect all of the objects
    bpy.ops.object.select_all(action='DESELECT')
    # loop all scene objects    
    for obj in bpy.context.scene.objects:
        # get the meshes
        if obj.type=="MESH":
            # select / reset origin / deselect
            obj.select_set(state=True)
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            obj.select_set(state=False)