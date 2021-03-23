import numpy as np
import math


def is_point_in_closed_segment(a, b, c):
    """ Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    """
    if a[0] < b[0]:
        return a[0] <= c[0] and c[0] <= b[0]
    if b[0] < a[0]:
        return b[0] <= c[0] and c[0] <= a[0]

    if a[1] < b[1]:
        return a[1] <= c[1] and c[1] <= b[1]
    if b[1] < a[1]:
        return b[1] <= c[1] and c[1] <= a[1]

    return a[0] == c[0] and a[1] == c[1]


def closed_segment_intersect(l1, l2):
    """ Verifies if closed segments a, b, c, d do intersect.
    """
    a = l1[0]
    b = l1[1]
    c = l2[0]
    d = l2[1]
    if a == b:
        return a == c or a == d
    if c == d:
        return c == a or c == b

    def side(a, b, c):
        """ Returns a position of the point c relative to the line
            If L is line with points a and b, then returns 1 if c is above l
            -1 of below l and 0 if on the line.
            going through a and b, Points a, b are expected to be different
        """
        d = (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
        return 1 if d > 0 else (-1 if d < 0 else 0)

    s1 = side(a, b, c)
    s2 = side(a, b, d)

    # All points are collinear
    if s1 == 0 and s2 == 0:
        return \
            is_point_in_closed_segment(a, b, c) or \
            is_point_in_closed_segment(a, b, d) or \
            is_point_in_closed_segment(c, d, a) or \
            is_point_in_closed_segment(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    s1 = side(c, d, a)
    s2 = side(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    return True


def get_slope(p1, p2):
    return float(p2[1] - p1[1]) / float(p2[0] - p1[0])


def get_y_intercept(p, slope):
    return p[1] - slope*p[0]


def get_point_of_intersection(l1, l2):
    a = l1[0]
    b = l1[1]
    c = l2[0]
    d = l2[1]
    try:
        m1 = float(get_slope(a, b))
        b1 = get_y_intercept(a, m1)
        m = m1
        b = b1
    except ZeroDivisionError:
        m1 = 'undefined'
    try:
        m2 = float(get_slope(c, d))
        b2 = get_y_intercept(c, m2)
        m = m2
        b = b2
    except ZeroDivisionError:
        m2 = 'undefined'
    x_i = None
    if m1 == 'undefined':  # y axis aligned
        x_i = a[0]
    if m2 == 'undefined':  # y axis aligned
        if x_i is None:
            x_i = c[0]
        elif a[0] == c[0]:  # if both lines lie over each other,
            # return the middle two points
            sorted_points = sorted([l1[0], l1[1], l2[0],
                                    l2[1]], key=lambda x: x[1])
            return [sorted_points[1], sorted_points[2]]
        else:
            raise "Parallel lines can't intersect"
    # Then lines are x-axis aligned, return the middle two points
    if type(m1) == float and type(m2) == float and (m1 - m2) == 0:
        sorted_points = sorted([l1[0], l1[1], l2[0], l2[1]],
                               key=lambda x: x[0])  # Sort by x axis
        return [sorted_points[1], sorted_points[2]]
    if x_i is None:
        x_i = (b2 - b1) / (m1 - m2)
    y_i = m * x_i + b
    return [(x_i, y_i)]


def are_same_floats(f1, f2, diff_threshold=1e-7):
    """
    f1: a flaot
    f2: another float
    """
    return abs(f1 - f2) < diff_threshold


class Polygon:
    def __init__(self, points=[]):
        """
        points : the point in the order to make the convex polygon
        """
        self.points = []
        self.edges = []
        prev_point = None
        for point in points:
            self.points.append(point)
            if prev_point is None:
                prev_point = point
            else:
                self.edges.append((prev_point, point))
                prev_point = point
        self.edges.append((prev_point, self.points[0]))

    @staticmethod
    def __get_center__(points):
        """
        points : ordered set of points which make the polygon
        returns the approximated center of the polygon
        """
        return np.mean(points, axis=0)

    def get_center(self):
        """
        returns the approximated center of the polygon
        """
        return Polygon.__get_center__(self.points)

    @staticmethod
    def ray_intersects_segment(P, A, B, epsilon=0.01):
        """
        P : the point from which the ray starts
        A : the end-point of the segment with the smallest y coordinate
            (A must be "below" B)
        B : the end-point of the segment with the greatest y coordinate
            (B must be "above" A)
        """
        if are_same_floats(P[1], A[1]) or are_same_floats(P[1], B[1]):
            P[1] = P[1] + epsilon
        if P[1] < A[1] or P[1] > B[1]:
            return False
        elif P[0] >= max(A[0], B[0]):
            return False
        else:
            if P[0] < min(A[0], B[0]):
                return True
            else:
                if not are_same_floats(A[0], B[0]):
                    m_red = (B[1] - A[1]) * 1.0 / (B[0] - A[0])
                else:
                    m_red = float('Inf')
                if not are_same_floats(A[0], P[0]):
                    m_blue = (P[1] - A[1]) * 1.0 / (P[0] - A[0])
                else:
                    m_blue = float('Inf')
                if m_blue >= m_red:
                    return True
                else:
                    return False

    def contains_point(self, point):
        """
        point: a tuple of (x,y)
        returns true if point resides inside polygon
        """
        num_of_intersections = 0
        for edge in self.edges:
            lower = [edge[0] if edge[0][1] < edge[1][1] else edge[1]][0]
            heigher = [edge[0] if edge[0][1] > edge[1][1] else edge[1]][0]
            if Polygon.ray_intersects_segment(list(point), list(lower),
                                              list(heigher)):
                num_of_intersections += 1
        # If odd intersections then point was inside
        return num_of_intersections % 2 > 0

    def __hash__(self):
        _hash = 0
        for point in self.points:
            # idea taken from
            # https://stackoverflow.com/questions/2634690/good-hash-function-for-a-2d-index
            # TODO needs refinement
            _hash += (53 + hash(point[0])) * 53 + hash(point[1])
        return _hash

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__hash__() == other.__hash__()

    def get_common_points(self, other_poly):
        """
        other_poly: an instance of this class Polygon
        returns set of all points where the point resided inside any
        other polygon
        """
        common_points = set()
        for point in other_poly.points:
            if self.contains_point(point):
                common_points.add(point)

        for point in self.points:
            if other_poly.contains_point(point):
                common_points.add(point)
        return common_points

    def get_intersections(self, other_poly):
        """
        other_poly: an instance of this class Polygon
        returns set of all intersecting points between this polygon
        and other_poly
        """
        intersections = set()
        for edge1 in self.edges:
            for edge2 in other_poly.edges:
                if closed_segment_intersect(edge1, edge2):
                    intersection_points = get_point_of_intersection(edge1,
                                                                    edge2)
                    for point in intersection_points:
                        intersections.add(point)
        return intersections

    # Technique taken from
    # https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
    @staticmethod
    def __clockwise_angle_and_distance__(point, origin,
                                         refvec=[0, 1]):
        """
        This function makes a line L using P and Origin, and
        using refvec (direction vector) measures the angle between projection
        of L on y axis and L
        point: is a tuple of (x,y)
        origin: is the refrence point for angle
        refvec: this dictates the direction check, [0,1] means clockwise
        return Angle and Distance of Point from Origin
        """
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        # x1*x2 + y1*y2
        dotprod = normalized[0]*refvec[0] + normalized[1]*refvec[1]
        # x1*y2 - y1*x2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to
        # subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance
        # should come first.
        return angle, lenvector

    @staticmethod
    def sort_points_counterclockwise(points):
        """
        points: takes a set of unsorted points
        returns counterclock wise sorted points
        """
        def compare(p1):
            center = Polygon.__get_center__(points)
            return Polygon.__clockwise_angle_and_distance__(p1, center)
        clock_wise_points = sorted(points, key=compare)
        return clock_wise_points

    def get_intersecting_polygon(self, other_poly):
        """
        other_poly: an instance of this class
        returns the convex polygon constructed by all common and intersecting
        points of this polygon and other_poly
        """
        # Get all points of this poly inside other poly and other poly inside
        # this poly
        intersecting_poly_points = self.get_common_points(other_poly)
        # Get all intersecting points
        intersecting_poly_points.update(self.get_intersections(other_poly))
        # Now clockwise sort these points to make a polygon
        intersecting_poly_points = Polygon. \
            sort_points_counterclockwise(list(intersecting_poly_points))
        # If your polygon has more than 2 points it is actually a poygon,
        # otherwise its eiter a point, line or no intersection at all
        if len(intersecting_poly_points) > 2:
            intersecting_poly = Polygon(intersecting_poly_points)
            return intersecting_poly
        else:
            return None

    @staticmethod
    # taken from https://stackoverflow.com/questions/4681737/how-to-calculate-
    # the-area-of-a-polygon-on-the-earths-surface-using-python/4682656#4682656
    def area_of_polygon(x, y):
        """
        Calculates the area of an arbitrary polygon given its verticies
        x: list of x coordinates
        y: list of y coordinates
        returns area
        """
        area = 0.0
        for i in range(-1, len(x)-1):
            a = x[i]
            b = (y[i+1] - y[i-1])
            area += a*b
        return abs(area) / 2.0

    def get_area(self):
        """
        returns area of this polygon
        """
        x = [point[0] for point in self.points]
        y = [point[1] for point in self.points]
        return Polygon.area_of_polygon(x, y)
