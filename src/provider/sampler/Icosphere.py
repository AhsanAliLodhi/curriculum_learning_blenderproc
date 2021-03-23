import mathutils
import bpy
from src.main.Provider import Provider


import math
import numpy as np

class Icosphere(Provider):
    """ Hinter Sampler samples uniformly distrubuted points on surface of a 
        sphere origined at 0,0,0 and then iterates over them in order.
        Samples 3D points on a sphere surface by refining an icosahedron, as in:
        Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
        Local Patches with a Simple Linear Classifier, BMVC 2008
        https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_lepetit/publications/hinterstoisser_bmvc08.pdf
        for better understanding you might find https://marcciufogreen.com/tag/sampling-a-sphere/ helpful.
    .. csv-table::
       :header:, "Parameter", "Description"
        "min_n_points", "this is the minimum number of points to be generated. Type: int"
        "radius", "this is the radius of the sphere. Type: float"
    """
    def __init__(self, config=None, min_n_points=None, radius=None):
        self.samplers = []
        if config is not None:
            Provider.__init__(self, config)
            self.min_n_points = config.get_int("min_n_points")
            self.radius = config.get_float("radius")
        else:
            self.min_n_points = min_n_points
            self.radius = radius

        if self.radius <= 0:
            raise ValueError("Invalid radius "+str(radius))

        points, _ = hinter_sampling(self.min_n_points, self.radius)
        self.points = points
        self.nextpoint = 0

    def run(self):
        """
        :param config: A configuration object containing the parameters
                       necessary to sample.
        :return: Sampled value. Type: Mathutils Vector
        """
        if self.nextpoint >= len(self.points):
            self.nextpoint = 0
        point = self.points[self.nextpoint]
        self.nextpoint += 1
        return point.tolist()

    def next(self):
        for point in self.points:
            yield(point)

    def __len__(self):
        return len(self.points)


def hinter_sampling(min_n_pts, radius=1.0):
    """
    Samples points according to the Hintler Sampler, see top for more information.
    :return: 3D points on the sphere surface and a list with indices of refinement
    levels on which the points were created.
    For implementation refer to following. 
    https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/view_sampler.py.
    """
    a, b, c = 0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0
    pts = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
            (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b),
            (-c, a, b)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
            (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
            (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
            (8, 6, 7), (9, 8, 1)]
    pts_level = [0 for _ in range(len(pts))]
    ref_level = 0
    while len(pts) < min_n_pts:
        ref_level += 1
        edge_pt_map = {}
        faces_new = [] 
        for face in faces:
            pt_inds = list(face)
            for i in range(3):
                edge = (face[i], face[(i + 1) % 3])
                edge = (min(edge), max(edge))
                if edge not in edge_pt_map.keys():
                    pt_new_id = len(pts)
                    edge_pt_map[edge] = pt_new_id
                    pt_inds.append(pt_new_id)
                    pt_new = 0.5 * (np.array(pts[edge[0]]) + np.array(pts[edge[1]]))
                    pts.append(pt_new.tolist())
                    pts_level.append(ref_level)
                else:
                    pt_inds.append(edge_pt_map[edge])
            faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
                        (pt_inds[3], pt_inds[1], pt_inds[4]),
                        (pt_inds[3], pt_inds[4], pt_inds[5]),
                        (pt_inds[5], pt_inds[4], pt_inds[2])]
        faces = faces_new
    pts = np.array(pts)
    pts *= np.reshape(radius / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))
    pt_conns = {}
    for face in faces:
        for i in range(len(face)):
            pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
            pt_conns[face[i]].add(face[(i + 2) % len(face)])
    top_pt_id = np.argmax(pts[:, 2])
    pts_ordered = []
    pts_todo = [top_pt_id]
    pts_done = [False for _ in range(pts.shape[0])]
    def calc_azimuth(x, y):
        two_pi = 2.0 * math.pi
        return (math.atan2(y, x) + two_pi) % two_pi
    while len(pts_ordered) != pts.shape[0]:
        pts_todo = sorted(
            pts_todo, key=lambda i: calc_azimuth(pts[i][0], pts[i][1]))
        pts_todo_new = []
        for pt_id in pts_todo:
            pts_ordered.append(pt_id)
            pts_done[pt_id] = True
            pts_todo_new += [i for i in pt_conns[pt_id]]
        pts_todo = [i for i in set(pts_todo_new) if not pts_done[i]]
    pts = pts[np.array(pts_ordered), :]
    pts_level = [pts_level[i] for i in pts_ordered]
    pts_order = np.zeros((pts.shape[0],))
    pts_order[np.array(pts_ordered)] = np.arange(pts.shape[0])
    for face_id in range(len(faces)):
        faces[face_id] = [pts_order[i] for i in faces[face_id]]
    return pts, pts_level