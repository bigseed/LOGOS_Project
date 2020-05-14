import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera


def distance(x1, y1, x2, y2):   # Calculate distance between two points in same plane
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)



def animate(interval):  # Animate the movement of ship
    animation = cam.animate(interval=interval)
    animation.save('{}.mp4'.format(seed))


def draw_orbit():   # Plot orbit of planets
    center = (0, 0)
    c1 = plt.Circle(center, planet_radius[0], fc='none', ec='green')
    c2 = plt.Circle(center, planet_radius[1], fc='none', ec='red')
    c3 = plt.Circle(center, planet_radius[2], fc='none', ec='blue')
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)


def draw_planet():  # Plot planets
    colors = ('green', 'red', 'blue')
    for idx in range(len(planet_radius)):
        ax.scatter(planet_radius[idx]*np.cos(planet_theta[idx]),
                   planet_radius[idx]*np.sin(planet_theta[idx]), c=colors[idx])



def fibo(x):    # Return fibonacci number from 1 to x
    fibo_nums = [1, 1]
    while True:
        if fibo_nums[-1] > x:
            break
        fibo_nums.append(fibo_nums[-1] + fibo_nums[-2])

    return fibo_nums[1:-1]

# Return progression of difference from 1 to x
def diff_of_progression(x, c_diff=1):
    progression = [1]

    while True:
        if progression[-1] > x:
            break
        new_num = progression[-1] + c_diff*len(progression)
        progression.append(new_num)
    return progression[:-1]


def move_planet():  # Orbital motion of planet
    global planet_theta

    for i in range(3):
        planet_theta[i] += angular_velocity[i]

def calc_gravity(x, y):  # Calculate gravity of all planets
    g_x, g_y = 0, 0

    for idx in range(3):
        planet_x = planet_radius[idx]*np.cos(planet_theta[idx])
        planet_y = planet_radius[idx]*np.sin(planet_theta[idx])

        d = distance(x, y, planet_x, planet_y)
        if d == 0:
            return ('Crash', 'to Planet')

        gravity = G * planet_mass[idx] / d**3

        g_x += gravity * (planet_x - x)
        g_y += gravity * (planet_y - y)

    return (g_x, g_y)

def calc_value(x, y, fuel):  # Calculate value func.
    start_planet_x = planet_radius[0]*np.cos(planet_theta[0])
    start_planet_y = planet_radius[0]*np.sin(planet_theta[0])
    target_planet_x = planet_radius[2]*np.cos(planet_theta[2])
    target_planet_y = planet_radius[2]*np.sin(planet_theta[2])

    return distance(x, y, start_planet_x, start_planet_y) / distance(x, y, target_planet_x, target_planet_y)**3



def first_make():   # Make first ship
    planet_x = (planet_radius[0]+how_far)*np.cos(planet_theta_init[0])
    planet_y = (planet_radius[0]+how_far)*np.sin(planet_theta_init[0])

    ships = list()
    for _ in range(n):
        info = dict(x=planet_x, y=planet_y, fuel=fuel_amount, weights=np.random.uniform(
            -first_speed_range, first_speed_range, (1, 2)), v_x=how_fast*np.cos(planet_theta_init[0]), v_y=how_fast*np.sin(planet_theta_init[0]), value=0.0, calc=True, isgood=False)
        ships.append(info)

    return ships

def children_make(weights):  # Make children with the best ship of last EPOCH
    planet_x = (planet_radius[0]+how_far)*np.cos(planet_theta_init[0])
    planet_y = (planet_radius[0]+how_far)*np.sin(planet_theta_init[0])

    ships = list()
    for _ in range(n):
        info = dict(x=planet_x, y=planet_y, fuel=fuel_amount,
                    weights=weights, v_x=how_fast*np.cos(planet_theta_init[0]), v_y=how_fast*np.sin(planet_theta_init[0]), value=0.0, calc=True, isgood=False)
        ships.append(info)

    return ships

def move_ships(moment, isokay):  # Moving algorithm of single step
    global ships

    # coords of target planet
    target_x = planet_radius[2]*np.cos(planet_theta[2])
    target_y = planet_radius[2]*np.sin(planet_theta[2])

    for ship in ships:
        if not ship['calc']:
            continue

        x, y = ship['x'], ship['y']
        d_to_target = distance(x, y, target_x, target_y)

        if d_to_target < 1:  # Reach at target planet
            ship['calc'] = False
            ship['isgood'] = True
            continue

        if distance(x, y, 0, 0) > 20*scale:
            ship['calc'] = False
            continue

        if ship['fuel'] <= 0:
            ship['calc'] = False
            continue

        a_x, a_y = calc_gravity(x, y)
        if a_x == 'Crash':
            ship['calc'] = False
            continue

        if isokay:  # Randomly change the orbit with epsilon
            if np.random.choice([True, False], 1, p=(epsilon, 1-epsilon))[0]:
                ship['weights'][moment] += np.random.uniform(-add_range,
                                                             add_range, (2,))

        else:
            temp_w = np.random.uniform(-speed_range, speed_range, (1, 2))
            ship['weights'] = np.concatenate(
                (ship['weights'], temp_w), axis=0)

        a_x += ship['weights'][moment][0] * fuel_to_acceleration
        a_y += ship['weights'][moment][1] * fuel_to_acceleration
        ship['fuel'] -= (np.abs(ship['weights'][moment][0]) +
                         np.abs(ship['weights'][moment][1]))

        ship['v_x'] = a_x + deceleration*ship['v_x']
        ship['v_y'] = a_y + deceleration*ship['v_y']

        ship['x'] += ship['v_x']
        ship['y'] += ship['v_y']

        ship['value'] = calc_value(
            ship['x'], ship['y'], ship['fuel'])*(1-gamma) + gamma*ship['value']
    move_planet()


def test_ship(ship):    # Test the best ship
    draw_orbit()
    draw_planet()
    plt.scatter(ship['x'], ship['y'], c='k')
    cam.snap()
    for moment in range(ship['weights'].shape[0]):
        print(moment)
        x, y = ship['x'], ship['y']
        a_x, a_y = calc_gravity(x, y)
        if a_x == "Crash":
            break
        a_x += ship['weights'][moment][0] * fuel_to_acceleration
        a_y += ship['weights'][moment][1] * fuel_to_acceleration

        ship['v_x'] = a_x + deceleration*ship['v_x']
        ship['v_y'] = a_y + deceleration*ship['v_y']

        ship['x'] += ship['v_x']
        ship['y'] += ship['v_y']

        draw_orbit()
        draw_planet()
        plt.scatter(ship['x'], ship['y'], c='k')
        cam.snap()
        move_planet()
    animate(75)


# make Canvas
scale = 3
fig, axes = plt.subplots()
ax = plt.axes(xlim=(-20*scale, 20*scale), ylim=(-20*scale, 20*scale))
ax.set_aspect('equal')
cam = Camera(fig)

# planet parameters
seed = 912723129
np.random.seed(seed)
planet_mass = (1, 8, 5)
planet_radius = (4, 11, 18)
planet_radius = tuple(map(lambda x: scale*x, planet_radius))
planet_theta_init = np.random.uniform(-np.pi, np.pi, (3,))
planet_theta = planet_theta_init.copy()
angular_velocity = np.sort(np.random.uniform(0, np.pi / 50, (3,)))[::-1]

# Initialize parameters
time = 150          # Training time
n = 500             # Number of ship which made in single EPOCH
epsilon = 0.0       # Epsilon for random change
deceleration = 0.5  # Velocity deceleration for each step
gamma = 0.05        # Value update parameter
add_range = 5       # Range of acceleration
first_speed_range = 50
speed_range = 50
fuel_to_acceleration = 1 / 50   # Fuel to accelerate 1
fuel_amount = 1e8
G = 1.2
EPOCHS = 1
total_best_ship = (0, 0)
total_best_ship_exist = False
how_far = 1         # Start from a certain distance away vertically
how_fast = 1.5      # Start with this velocity


# Main algorithm
for epoch in range(EPOCHS):

    # Make ships for this EPOCH
    if total_best_ship_exist:
        best_ship = total_best_ship
        ships = children_make(best_ship[0])
    else:
        best_ship = (0, -1)
        ships = first_make()

    for duration in diff_of_progression(time):
        best_ship_exist = False
        weight_length = ships[0]['weights'].shape[0]
        for moment in range(duration):
            if moment < weight_length:
                move_ships(moment, True)
            else:
                move_ships(moment, False)

        # Find the best ship with value
        max_val_ship = (0, -1)
        for ship in ships:
            if ship['isgood']:
                if ship['value'] > best_ship[1]:
                    best_ship = (ship['weights'].copy(), ship['value'])
                    best_ship_exist = True
                    total_best_ship_exist = True
            if ship['value'] > max_val_ship[1]:
                max_val_ship = (ship['weights'].copy(), ship['value'])

        # Make children
        if best_ship_exist:
            ships = children_make(best_ship[0].copy())
            total_best_ship = best_ship
            print('There we Go!')
        else:
            ships = children_make(max_val_ship[0].copy())
            print('None of them...')

        planet_theta = planet_theta_init.copy()

planet_theta = planet_theta_init.copy()
planet_x = (planet_radius[0]+how_far)*np.cos(planet_theta_init[0])
planet_y = (planet_radius[0]+how_far)*np.sin(planet_theta_init[0])

if total_best_ship_exist:
    perfect_ship = dict(x=planet_x, y=planet_y, fuel=fuel_amount,
                        weights=total_best_ship[0].copy(), v_x=how_fast*np.cos(planet_theta_init[0]), v_y=how_fast*np.sin(planet_theta_init[0]))

    test_ship(perfect_ship)
