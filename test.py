from AutonomousFleet import *

np.random.seed(10)
ships = 2*np.random.random_sample((100, 2)) - 1

test_ = AutonomousFleet(ships)
ship = ships[10]

# Punto 1
s = 5
near_ = test_.nearest_ships(ship, s, True)
print("The {0} nearest ships for the ship {1} are: {2}"
      .format(s, ship, near_))

# Punto 2
r = 0.5
angle = -60
collisions = test_.avoid_collision(ship, angle, r, True)
print("The ship located at {0} must avoid collisions"
      " with the ships located at: {1}"
      .format(ship, collisions))

# Punto 3
east, north, west, south = test_.min_max_ships(True)
print("You must explore the next ships {0} (east) "
      ", {1} (north), {2} (west), and {3} (south)"
      .format(east, north, west, south))

# Punto 4
test_.plot()
