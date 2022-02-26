#######################################################################
#             Computational and differential Geometry                 #
#                         Third Homework                              #
#                 Andres Felipe Florian Quitian                       #
#######################################################################

from matplotlib.patches import Rectangle
import matplotlib as mpl
import matplotlib.path as mplPath
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean, cdist
import random as rng


class AutonomousFleet:

    """ Class AutonomousFleet that represents a set of ships
        on the 2D-plane with a KD-Tree as structure

        Attributes: - pt: array withthe coordinates for each ship
                    - tree: KD-Tree structure with pt as data
    """

    def __init__(self, points):

        """ Constructor: Class atributes initialization

            Arguments: - points: array. ships coordinates
        """

        self.pt = points
        self.tree = KDTree(points)

    def _map(self, P, depth, min_x, max_x, min_y, max_y):

        """ Private method that that generates a two-dimensional
            map with the location of each ship in the fleet

            Input: - P: array of points
                   - depth: current depth
                   - min_x, max_x: min and max coordinates in the x axis
                   - min_y, max_y: min and max coordinates in the y axis
        """

        if depth == 0:
            plt.scatter(self.pt[:, 0], self.pt[:, 1], color='black')

        if len(P) == 1:
            return

        else:

            P = P[P[:, depth % 2].argsort()]  # array sort
            n = (len(P) // 2) - 1
            median = P[n]
            P_1 = P[:n + 1]  # left branch
            P_2 = P[n + 1:]  # right branch

            if depth % 2 == 0:
                plt.vlines(x=median[0], ymin=min_y, ymax=max_y, colors='b')

                if len(P_1) > 0:
                    new_max_x = median[0]
                    self._map(P_1, depth + 1, min_x, new_max_x, min_y, max_y)

                if len(P_2) > 0:
                    new_min_x = median[0]
                    self._map(P_2, depth + 1, new_min_x, max_x, min_y, max_y)
            else:
                plt.hlines(y=median[1], xmin=min_x, xmax=max_x, colors='r')

                if len(P_1) > 0:
                    new_max_y = median[1]
                    self._map(P_1, depth + 1, min_x, max_x, min_y, new_max_y)

                if len(P_2) > 0:
                    new_min_y = median[1]
                    self._map(P_2, depth + 1, min_x, max_x, new_min_y, max_y)

    def nearest_ships(self, pt, s, plot=False):

        """ Method that reports the s closest ships for a given ship

            Input: - pt: ship coordinates
                   - s: number of closest neighbors needed
                   - plot: boolean for plotting
            Output: list with the coordinates of the s neighbors
        """

        ball_distance_s = self.tree.query(pt, s + 1)
        nearest = self.pt[ball_distance_s[1][1:]]

        if plot:

            fig = plt.figure()

            plt.scatter(self.pt[:, 0], self.pt[:, 1], s=10,
                        marker='.', c='b')
            plt.scatter(pt[0], pt[1], s=50, marker='*',
                        c='g', label='Given Ship')
            plt.scatter(nearest[:, 0], nearest[:, 1], s=50,
                        marker='*', c='r', label='Neighbors')

            fig.legend(loc='lower right')
            plt.title('Nearest Ships')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        return nearest

    def avoid_collision(self, pt, angle, r, plot=False):

        """ Method that reports if there are any ships on a oriented
            square of length r and a given center (ship)

            Input: - pt: ship coordinates
                   - angle: square orientation
                   - r: length of the square
                   - plot: boolean for plotting
            Output: list with the coordinates of the ships that yields
                    in the square
        """

        minor_r = self.tree.query_ball_point(pt, r/2**(1/2))
        ships = self.pt[minor_r]

        if plot:

            fig, ax = plt.subplots()

            c = pt[0] - r/2, pt[1] - r/2
            t2 = mpl.transforms.Affine2D().rotate_around(pt[0], pt[1], angle)

            square = Rectangle(c, r, r, fill=False)
            square.set_transform(t2 + ax.transData)
            ax.add_patch(square)

            path = square.get_path().vertices[:-1]
            coords = square.get_patch_transform().transform(path)
            coords = t2.transform(coords)

            poly_path = mplPath.Path((coords))
            points = []

            ax.scatter(self.pt[:, 0], self.pt[:, 1],  s=10,
                       marker='.', c='b')

            for point in ships:
                if poly_path.contains_point(point):
                    points.append(point)
            points = np.array(points)

            ax.scatter(points[:, 0], points[:, 1], s=50,
                       marker='*', c='r', label='Ships in square')
            ax.scatter(pt[0], pt[1], s=50,
                       marker='*', c='g', label='Center')

            fig.legend(loc='lower right')
            plt.title('Avoiding Collisions')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            plt.show()

        return ships

    def min_max_ships(self, plot=False):

        """ Method that reports the ships with minimal
            and maximal coordinates in each axis

            Input: - plot: boolean for plotting
        """

        max_x, max_y = self.tree.maxes
        min_x, min_y = self.tree.mins

        max_x_ships = []
        max_y_ships = []
        min_x_ships = []
        min_y_ships = []

        for ship in self.pt:
            if ship[0] == max_x:
                max_x_ships.append(ship)
            if ship[0] == min_x:
                min_x_ships.append(ship)
            if ship[1] == max_y:
                max_y_ships.append(ship)
            if ship[1] == min_y:
                min_y_ships.append(ship)

        if plot:

            fig = plt.figure()

            plt.scatter(self.pt[:, 0], self.pt[:, 1], s=10,
                        marker='.', c='black')

            max_x_ships = np.array(max_x_ships)
            min_x_ships = np.array(min_x_ships)
            max_y_ships = np.array(max_y_ships)
            min_y_ships = np.array(min_y_ships)

            plt.scatter(max_x_ships[:, 0], max_x_ships[:, 1], s=50,
                        marker='*', c='r', label='East')
            plt.scatter(min_x_ships[:, 0], min_x_ships[:, 1], s=50,
                        marker='*', c='y', label='West')
            plt.scatter(max_y_ships[:, 0], max_y_ships[:, 1], s=50,
                        marker='*', c='g', label='North')
            plt.scatter(min_y_ships[:, 0], min_y_ships[:, 1], s=50,
                        marker='*', c='b', label='South')

            fig.legend(loc='lower right')
            plt.title('Leading and lagging vessels')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            plt.show()

        return max_x_ships, max_y_ships, min_x_ships, min_y_ships

    def plot(self):

        """ Method that plots each ships as a point and the
            corresponding splitting lines
        """

        max_x, max_y = self.tree.maxes
        min_x, min_y = self.tree.mins

        self._map(self.pt, 0, min_x, max_x, min_y, max_y)
        plt.title('Two-Dimensional Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
