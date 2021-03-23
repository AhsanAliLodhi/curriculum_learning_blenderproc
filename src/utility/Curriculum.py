import bpy
import random
from mathutils import Vector
import numpy as np
import math
# from src.utility.CameraUtility import move_and_point_to_camera,\
#                                         get_obj_size, get_scale_range,\
#                                         get_fov_dimensions,\
#                                         make_fov_fitting_plane,\
#                                         apply_image_texture
try:
    random = random.SystemRandom()
except Exception as e:
    import random


def join_all(objs, parent):
    """
    Takes in a list of bpy objects and one parent object, then joins
    all objects in the list to the parent object.
    :objs: list of bpy mesh objects
    ;parent: a bpy object
    """
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    parent.select_set(True)
    bpy.context.view_layer.objects.active = parent
    bpy.ops.object.join()


def replicate(obj, times, dist=3, new_loc=None):
    """
    Takes in a bpy mesh bjects then makes it 'time' copies.
    :obj: a bpy mesh object
    :times: number of copies we want
    :dist: Optional, places the new object dist untis away from obj on z axis
    :new_loc: Optional, places the new objects at specified location
    returns a list of bpy objects only containing the copies of obj
    """
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    objs = []
    for i in range(times):
        bpy.ops.object.duplicate({"object": obj})
        new_obj = bpy.context.object
        if new_loc is None:
            new_obj.location[2] = new_obj.location[2] + dist
        else:
            new_obj.location = new_loc
        #print("============Replicated object {}=>{} times {}==============".format(obj.name, new_obj.name, times))
        objs.append(new_obj)
    return objs
    
 

def scale_obj(obj, scale, index=0):
    """
    scales an object proportionally along all axis
    :obj: bpy mesh object to scale
    :scale: scale to be applied
    :index: possible values [0-2] x=0, y=1, z=2, specifies the axis for
            above scale
    """
    new_scale = [s*scale/obj.scale[index] for s in obj.scale]
    obj.scale = new_scale
    bpy.context.view_layer.update()

def scale_normalized_obj(obj, scale):
    """
    scales an object proportionally along all axis
    :obj: bpy mesh object to scale
    :scale: scale to be applied
 
    """
    new_scale = [s*scale for s in obj.scale]
    obj.scale = new_scale
    bpy.context.view_layer.update()

def set_dimensions_obj(obj, dimension_value, index=None):
    """
    sets one dimension of an object and 
    scales the rest of dimensions proportionally along all axis
    :obj: bpy mesh object to scale
    :dimension_value: absolute dimension value to be applied
    :index: possible values [0-2] x=0, y=1, z=2, specifies the axis for
            above dimension to be applied, if None chooses the index with
            highest dimension
    """
    if index is None:
        max_d = max(obj.dimensions)
        # Get max dimension
        index = [i for i, j in enumerate(obj.dimensions) if j==max_d][0]
    new_dimensions = [s*dimension_value/obj.dimensions[index] for s in obj.dimensions]
    obj.dimensions = new_dimensions
    bpy.context.view_layer.update()


def get_num_to_fit(obj, camera, scene, rx=None, ry=None):
    """
    Given an bpy mesh object, returns the number of obj needed to
    align consequtively to fill the camera view along the largest dimension
    of that object
    :obj: bpy mesh object
    :camera: camera object
    :scene: scene object
    :rx: (optional) resolution in x axis
    :ry: (optional) resolution in y axis
    returns an integer, number of objects needed to fill screen
    """
    dist = (obj.location - camera.location).length
    if rx is None:
        rx = scene.render.resolution_x
    if ry is None:
        ry = scene.render.resolution_y
    w, h, _, _ = get_obj_size(obj, camera, scene, dist)
    factor = int( max(rx, ry) / max(w*rx, h*ry))
    return factor


def create_layer(bgobjs, minp=(-5, -5), maxp=(5, 5), divide_by=10, dropout=0.5,
                 z=0, scale_range=None, camera=None, scene=None,
                 is_height=True):
    """
    Creats a layer of objects from a given a list of objects using following
    routine.
    1 - Split the bounded plane made from minp and maxp using divide_by
        or simply (Make a grid)
    2 - For each block place a random object from bgobjs at a random position
        in that block with probability 1 - dropout
    :bgobjs: List of bpy mesh objects
    :minp: min values for x and y axis
    :maxp: max values for x and y axis
    :divide_by: value used to make blocks, final number of blocks will be
                divide_by**2
    :dropout: probability value to ignore a block
    :z: is the value of z axis used to position the objects in layer
    :scale_range: Optional, Given a tuple (r1, r2) scales the object with
                a random scale form this range.
    :camera: The camera object
    :scene: The scene object
    :is_height: if True means scales are for y axis, otherwise x axis
    returns list of objetcs used in the layer, element most closest to the
            center
    """
    dx = (maxp[0] - minp[0])/divide_by
    dy = (maxp[1] - minp[1])/divide_by
    center_point = Vector(((maxp[0] - minp[0])/2, (maxp[1] - minp[1])/2, z))
    objs = []
    obj_idx =  [0] * len(bgobjs)
    almost_at_center = None
    total_obj = 0
    #dropout = dropout * divide_by/10
    for i in range(divide_by):
        x = random.uniform(minp[0] + dx*i, minp[0] + dx*i + dx)
        for j in range(divide_by):
            put_obj = random.uniform(0, 1) < (1 - dropout)
            if not put_obj:
                continue
            total_obj += 1
            y = random.uniform(minp[1] + dy*j, minp[1] + dy*j + dy)
            o_idx = random.randint(0, len(bgobjs)-1)  # pick a random object
            obj_idx[o_idx] += 1
            new_obj = replicate(bgobjs[o_idx], 1, 0, new_loc=(x, y, z))[0]
            new_obj.rotation_euler = (random.uniform(-1, 1),
                                      random.uniform(-1, 1),
                                      random.uniform(-1, 1))
            if almost_at_center is not None:
                if (almost_at_center.location - center_point).length < \
                   (new_obj.location - center_point).length:
                    almost_at_center = almost_at_center
                else:
                    almost_at_center = new_obj
            else:
                almost_at_center = new_obj
            objs.append(new_obj)
    for obj in objs:
        if scale_range is not None:
            scale = random.uniform(scale_range[0], scale_range[1])
            scale_normalized_obj(obj, scale)
    return objs, almost_at_center


def make_bg(bgobjs, fobj, fg_object_idx, camera, scene, scale_range=None, bg_scale=None, dist=None,
            divide_by=None, dropout=0.5, delta_z=2, no_of_layers=3,
            is_height=True, real_image_path=None, light_energy=0):
    """
    Creats a backgound object for a fobj by first making multiple layers using
    bgobjs and then joining the layers.
    :bgobjs: List of bpy mesh objects
    :fobj: a bpy mesh object for which to create a background scene
    :dist: distance at which we want to make the backgroud layer (optional)
    :divide_by: value used to make blocks of background layers,
                final number of blocks will be divide_by**2
    :dropout: probability value to ignore a block in a layer of bg
    :delta_z: is the value change of z axis value for each layer
    :scale_range: Optional, Given a tuple (r1, r2) scales the object with
                a random scale form this range.
    :camera: The camera object
    :scene: The scene object
    :is_height: if True means scales are for y axis, otherwise x axis
    :no_of_layers: Number of layers to make in one background object
    returns the background object
    """
    layers = []
    bpy.ops.mesh.primitive_plane_add()
    center = bpy.context.object
    center.scale = (1, 1, 1)
    center.name = "bg"+str(fg_object_idx)
    center.hide_render = True
    bpy.context.view_layer.update()
    if dist is None:
        dist = (fobj.location - camera.location).length * 2
        
    dims, __ = get_fov_dimensions(camera, dist, scene)
    minp = (-dims[0]/2, -dims[1]/2)
    maxp = (dims[0]/2, dims[1]/2)
    suggested_divideby = int(dims.length/bg_scale * 1.5)

    if divide_by is None:
        divide_by = suggested_divideby
    for i in range(no_of_layers):
        layer, __ = create_layer(bgobjs, minp=minp, maxp=maxp,
                                 divide_by=divide_by, dropout=dropout,
                                 z=-delta_z*i, scale_range=scale_range,
                                 camera=camera, scene=scene,
                                 is_height=is_height)
        layers.append(layer)
    objs = [o for l in layers for o in l]
    join_all(objs, center)
    move_and_point_to_camera(center, camera, dist)
    center.rotation_euler[0] = math.radians(-90) #Euler((0.3, 0.3, 0.4), 'XYZ')

    if real_image_path is not None:
        bpy.ops.mesh.primitive_plane_add()
        bg_image = bpy.context.object
        bg_image.scale = (dims.length/2, dims.length/2, 1)
        bg_image.name = "bg_image"
        apply_image_texture(bg_image, real_image_path)
        move_and_point_to_camera(bg_image, camera, dist*1.3)
        join_all([bg_image], center)


    # Create light data, link it to the new object
    light_obj = []
    if light_energy != 0:
        light_data = bpy.data.lights.new(name="light", type="AREA")
        light_obj = bpy.data.objects.new(name="light", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = [-0.4, 0, 0]
        light_obj.rotation_euler = [0, 1.5708, 0]
        light_obj.scale = (dims.length*2, dims.length*2, 1)
        light_data.energy = light_energy
        light_data.color = [1, 1, 1]
        light_data.distance =  0
        #join_all([light_data], center)
    
    return center, light_obj


def make_real_bg(real_image_path, fobj, camera, scene, dist=None):
    """
    Creats a backgound object for a fobj by first making multiple layers using
    bgobjs and then joining the layers.
    :bgobjs: List of bpy mesh objects
    :fobj: a bpy mesh object for which to create a background scene
    :dist: distance at which we want to make the backgroud layer (optional)
    :divide_by: value used to make blocks of background layers,
                final number of blocks will be divide_by**2
    :dropout: probability value to ignore a block in a layer of bg
    :delta_z: is the value change of z axis value for each layer
    :scale_range: Optional, Given a tuple (r1, r2) scales the object with
                a random scale form this range.
    :camera: The camera object
    :scene: The scene object
    :is_height: if True means scales are for y axis, otherwise x axis
    :no_of_layers: Number of layers to make in one background object
    returns the background object
    """
    if dist is None:
        dist = (fobj.location - camera.location).length * 2
    bg = make_fov_fitting_plane(camera, dist, scene, margin=0)
    apply_image_texture(bg, real_image_path)
    bg.name = "bg"
    return bg




#New classes for v3

def clamp(x, lower_bound, y, do_clamp):
    """
    clamp returns lower_bound if min(x,y) < lower_bound else returns min(x,y)
    :x: a number
    :lower_bound: lower bound
    :y: a number
    :do_clamp: enable or disable clamping
    """
    if do_clamp:
        return max(lower_bound, min(x, y))
    else:
        return min(x, y)


def get_2D_bounds(scene, cam_ob, me_ob, do_clamp=True):
    """
    Get 2D bounding box of an object from a camera's prespective
    :scene: the bpy scene object
    :cam_ob: the camera object
    :me_ob: the object in focus
    :do_clamp: enable or disable clamping
    returns (min_x, min_y, max_x, max_y, )bounds and (dim_x, dim_y) camera
    dimensions where bounds are from 0 to 1 and dimensions are in pixels
    """
    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(preserve_all_data_layers=True)
    me.transform(me_ob.matrix_world)
    me.transform(mat)
    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'
    lx = []
    ly = []
    for v in me.vertices:
        co_local = v.co
        z = -co_local.z
        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)
        lx.append(x)
        ly.append(y)
    min_x = clamp(min(lx), 0.0, 1.0, do_clamp)
    max_x = clamp(max(lx), 0.0, 1.0, do_clamp)
    min_y = clamp(min(ly), 0.0, 1.0, do_clamp)
    max_y = clamp(max(ly), 0.0, 1.0, do_clamp)
    me_ob.to_mesh_clear()
    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    return min_x, min_y, max_x, max_y, dim_x, dim_y


def bounds_to_vertices(bounds):
    """
    Get bounds in format of [minx, miny, maxx, maxy]
    and converts them to consecutrive vertices
    in counter clockwise direction
    :bounds: a list of four floats. Type: [float]
    returns list of 4 tuples where each tuple is (x, y). Type: [(float, float)]
    """
    v1 = (bounds[0], bounds[1])
    v2 = (bounds[0], bounds[3])
    v3 = (bounds[2], bounds[3])
    v4 = (bounds[2], bounds[1])
    return [v1, v2, v3, v4]

def move_camera_z(camera, dist, frm):
    """
    Place camera at a distance of dist from an object
    :camera: the camera object
    :dist: distance in blender units (usually in meters)
    :frm: the object in whose direction we move
    """
    direction = camera.location - frm.location
    direction.normalize()
    camera.location = frm.location + direction * dist
    bpy.context.view_layer.update()


def move_object_z(obj, camera, dist, camera_look_at=Vector((0, 0, 0))):
    """
    Place an object dist untis away from camera in direction of where camera is
    looking at
    :camera: the camera object
    :dist: distance in blender dist (usually in meters)
    :obj: the object in whose direction we move
    :camera_look_at: a point where camera is looking
    """
    direction = camera_look_at - camera.location
    direction.normalize()
    obj.location = camera.location + direction * dist
    bpy.context.view_layer.update()


def point_to(obj, location):
    """
    Rotate an object to look at location in 3D space
    :location: the 3D point
    :obj: the object who moves
    """
    location = Vector(location)
    direction = obj.location - location
    direction.normalize()
    direction = direction.to_track_quat('-Z', 'Y').to_euler()
    obj.rotation_euler = direction
    bpy.context.view_layer.update()


def point_to_camera(obj, camera):
    """
    Rotate an object to face camera
    :camera: the camera object
    :obj: the object in whose direction we move
    """
    point_to(obj, camera.location)


def point_camera_to(obj, camera):
    """
    Rotate an camera to face an object
    :camera: the camera object
    :obj: the object in whose direction we move
    """
    direction = obj.location - camera.location
    direction.normalize()
    direction = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.rotation_euler = direction
    bpy.context.view_layer.update()

def move_and_point_to_camera(obj, camera, units,
                             camera_look_at=Vector((0, 0, 0))):
    """
    combines moving an object and pointing to camera in one call
    (often happen together)
    :camera: the camera object
    :dist: distance in blender dist (usually in meters)
    :obj: the object in whose direction we move
    :camera_look_at: a point where camera is looking
    """
    move_object_z(obj, camera, units)
    point_to_camera(obj, camera)


def get_obj_size(obj, camera, scene, dist=3, do_clamp=True):
    """
    Gets an objects size from a camera's prespective if the object was
    dist units away from camera
    :camera: the camera object
    :dist: distance in blender dist (usually in meters)
    :obj: the object that we want size of
    :do_clamp: enable or disable clamping
    """
    base_state = camera.matrix_world.copy()
    move_camera_z(camera, dist, obj)
    bounds = get_2D_bounds(scene, camera, obj, do_clamp)
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    camera.matrix_world = base_state
    bpy.context.view_layer.update()
    return width, height, bounds[4], bounds[5]





def get_dist_range_for_scale(proportion, obj, camera, scene, min_scale=0.9,
                             max_scale=1.5, dist=3):
    """
    Calculates the required distance of an object from camera if we want it
    to appear said 'proportion' big on the camera image, also returns a
    range of the distances which conform with min_scale and max_scale
    :proportion: the proportion of size of the biggest dimension on camera
    image
    :camera: camera object
    :scene: scene object
    :min_scale: is the minimum factor of proportion, min_scale*proportion
    :max_scale: is the maximum factor of proportion, max_scale*proportion
    :obj: the object that we want size of
    :dist: is the constant dist used to calculate camera constant
    (Note: for the tecnique to work dist should be large enough
    for object to completely appear in the camera)
    """
    def get_camera_const(obj, camera, scene, dist=3, do_clamp=True):
        width, height, _, _ = get_obj_size(obj, camera, scene, dist=dist,
                                           do_clamp=do_clamp)
        is_height = height > width
        obj_size_in_image = height if is_height else width
        const = obj_size_in_image * dist
        return const, is_height

    def get_camera_dist(size, const):
        return const/size
 
    const, is_height = get_camera_const(obj, camera, scene, dist=dist)
    dist = get_camera_dist(proportion, const)
    max_dist = get_camera_dist(proportion * min_scale, const)
    min_dist = get_camera_dist(proportion * max_scale, const)
    return min_dist, max_dist, dist, is_height


def get_scale(d1, d2, s1):
    """
    Given distance and scale and required scale returns the required distance
    (Note: If you replace distances with scales the funciton returns required
    scale)
    :d1: distance
    :d2: required distance
    :s1: required scale
    """
    return d1*s1/d2


def get_scale_range(proportion, obj, camera, scene, min_scale=0.9,
                    max_scale=1.5, dist=3):
    """
    Calculates the required proportion of an object from camera if we want it
    to appear said 'proportion' big on the camera image, also returns a
    range of the scales which conform with min_scale and max_scale,
    This method also returns all the return values of get_dist_range_for_scale
    :proportion: the proportion of size of the biggest dimension on camera
    image
    :camera: camera object
    :scene: scene object
    :min_scale: is the minimum factor of proportion, min_scale*proportion
    :max_scale: is the maximum factor of proportion, max_scale*proportion
    :obj: the object that we want size of
    :dist: is the constant dist used to calculate camera constant
    (Note: for the tecnique to work dist should be large enough
    for object to completely appear in the camera)
    """
    base_dist = (obj.location - camera.location).length
    # assuming that proportion is same for all axis
    min_dist, max_dist, dist, is_height = get_dist_range_for_scale(proportion,
                                                                   obj,
                                                                   camera,
                                                                   scene,
                                                                   min_scale,
                                                                   max_scale,
                                                                   dist)
    if is_height:
        base_scale = obj.scale[1]
    else:
        base_scale = obj.scale[0]
    max_scale = get_scale(base_dist, max_dist, base_scale)
    min_scale = get_scale(base_dist, min_dist, base_scale)
    scale = get_scale(base_dist, dist, base_scale)
    return max_scale, min_scale, scale, min_dist, max_dist, dist, is_height



def make_fov_fitting_plane(camera, dist, scene, margin=0.25):
    """
    this function returns creates a plane which fits field of view
    of a camera at specified distance.
    :camera: camera object
    :dist: distance from camera, in blender its in meters
    :scene: scene object
    :margin: introduces extra margin in terms of proportion of area of the
             bounded plane, 0 means a perfect fit, 0.25 means 25 % extra
             area on borders, like margin in CSS
    """
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object
    move_and_point_to_camera(plane, camera, dist)
    scale_ranges = get_scale_range(1, plane, camera, scene)
    plane.scale = plane.scale * scale_ranges[2] * (1 + margin)
    bpy.context.view_layer.update()
    return plane


def get_fov_dimensions(camera, dist, scene, margin=0.25):
    """
    this function returns dimensions and bounds of field of view
    of a camera at specified distance.
    :camera: camera object
    :dist: distance from camera, in blender its in meters
    :scene: scene object
    :margin: introduces extra margin in terms of proportion of area of the
             bounded plane, 0 means a perfect fit, 0.25 means 25 % extra
             area on borders, like margin in CSS
    """
    plane = make_fov_fitting_plane(camera, dist, scene, margin)
    dimensions = plane.dimensions.copy()
    bounds = []
    for v in plane.data.vertices:
        bounds.append(plane.matrix_world @ v.co)
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    bpy.ops.object.delete()
    return dimensions, bounds

def apply_image_texture(obj, img_path, mat_name="def_mat"):
    """
    This function simply creates a material and applys image
    texture to an object. It also returns the newly created texture
    and material nodes.
    :obj: mesh object
    :img_path: path to the image to be applied in texture
    :mat_name: name of the created material
    """
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(img_path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'],
                            texImage.outputs['Color'])
    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return mat, texImage

def get_fov_dimensions(camera, dist, scene, margin=0.25):
    """
    this function returns dimensions and bounds of field of view
    of a camera at specified distance.
    :camera: camera object
    :dist: distance from camera, in blender its in meters
    :scene: scene object
    :margin: introduces extra margin in terms of proportion of area of the
             bounded plane, 0 means a perfect fit, 0.25 means 25 % extra
             area on borders, like margin in CSS
    """
    plane = make_fov_fitting_plane(camera, dist, scene, margin)
    dimensions = plane.dimensions.copy()
    bounds = []
    for v in plane.data.vertices:
        bounds.append(plane.matrix_world @ v.co)
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    bpy.ops.object.delete()
    return dimensions, bounds

















class ObjectInstanceProvider:
    """
    This is a container class which takes in a list of mesh objects
    and can provide you with any number of copies of any instance
    from that list. This also alows one to soft delete the copies
    and resue the copies instead of creating a new one.
    Idea: for each item in the list, create a buffer, where buffer
    is a list which contians item and all its copies.
    """

    def __init__(self, object_list, origin, default_class=None):
        """
        :object_list: list of bpy mesh objects. Type: [bpy.data.object]
        :origin: a tuple of x, y, z coordinates. Type: (float, float. float)
        """
        self.origin = origin
        self.objs = [[obj] for obj in object_list]
        self.taken = [[False] for _ in object_list]
        for idx, obj in enumerate(object_list):
            if default_class is None:
                obj["category_id"] = idx + 1
            else:
                obj["category_id"] = default_class

    def search(self, obj):
        """
        searches for an object in the list.
        :obj: Bpy mesh object, which is either an item in the list or a copy
              of the item. Type: bpy.data.object
        :return: idx is the index of buffer for object, idx_idx is index of
                    obj in the buffer
        """
        # check if the given object exists in any list
        idx = 0
        for buffer in self.objs:
            if obj in buffer:
                idx_idx = buffer.index(obj)
                return idx, idx_idx
            idx += 1
        # return -1, -1 if item not found
        return -1, -1

    def get(self, idx):
        """
        Returns either the item residing at index idx or a copy of it.
        :idx: index of item to be fetched. Type: int
        :return: a free item in buffer at idx, Type: bpy.data.object
        """
        # if idx is a legal index
        assert idx >= 0 and idx < len(self.objs)
        # if we have an instance available
        if False in self.taken[idx]:
            # then return it
            idx_idx = self.taken[idx].index(False)
            obj = self.objs[idx][idx_idx]
            obj.hide_render = False
            self.taken[idx][idx_idx] = True
            return obj
        else:
            # create a new instance
            copy = replicate(self.objs[idx][0], 1, 1, new_loc=self.origin)[0]
            # ensure the copy has same class as its source
            copy["category_id"] = self.objs[idx][0]["category_id"]
            # add it to the list of objs
            self.objs[idx].append(copy)
            self.taken[idx].append(True)  # True because the copy will be taken
            # set it to renderable before returning
            copy.hide_render = False
            return copy

    def remove(self, obj):
        """
        Soft deletes the object.
        :obj: Bpy mesh object, which is either an item in the list or a copy
              of the item. Type: bpy.data.object
        :return: True if item found and removed, False otherwise Type: bool
        """
        # check if the given object exists in any list
        idx, idx_idx = self.search(obj)
        if idx > -1 and idx_idx > -1:
            # set taken to false for this instance
            self.taken[idx][idx_idx] = False
            # make renderable
            obj.hide_render = True
            return True
        return False

    def set_taken(self, obj, taken=True):
        """
        Custom sets the taken flag for given object.
        :obj: Bpy mesh object, which is either an item in the list or a copy
              of the item. Type: bpy.data.object
        """
        # check if the given object exists in any list
        idx, idx_idx = self.search(obj)
        if idx > -1 and idx_idx > -1:
            # set taken to false for this instance
            self.taken[idx][idx_idx] = taken
            # make renderable if taken and vice versa
            obj.hide_render = not taken
            return True
        return False

    def get_random(self):
        """
        return a random item from the list
        :return: Type: bpy.data.object
        """
        return self.get(random.choice(range(len(self.objs))))

    def flatten(self):
        """
        return a the flatten version of the self.objs
        :return: Type: [bpy.data.object]
        """
        objs = [obj for buffer in self.objs for obj in buffer]
        return objs

    def clear(self):
        """
        set taken for all elements to false
        """
        self.taken = [[False] for _ in self.objs]
