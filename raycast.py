"""
Raycasting
"""

import arcade
import random
import numpy as np

from noise import pnoise1
from contextlib import contextmanager
from math import cos, sin, radians, inf
from typing import Optional, List


WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Raycasting"

# simple affine stack: (a,b,c,d,e,f) for 2D matrix [[a,b,e],[c,d,f],[0,0,1]]
_stack = [(1, 0, 0, 1, 0, 0)]


def _apply(x, y):
    """
    Takes the local point x,y
    Multiplies it by the current transformation matrix
    Add the translation
    Returns the transformed world coordinates
    """
    a, b, c, d, e, f = _stack[-1]
    return (a*x + b*y + e, c*x + d*y + f)


@contextmanager
def translate(pos: arcade.Vec2):
    """
    arcade, doesn't have a transform which allows you
    to move the origin from 0,0 to wherever you like
    """
    a, b, c, d, e, f = _stack[-1]
    _stack.append((a, b, c, d, e+pos.x, f+pos.y))
    try:
        yield
    finally:
        _stack.pop()


@contextmanager
def rotate(deg: float):
    """
    Use with transform
    """
    a, b, c, d, e, f = _stack[-1]
    r = radians(deg)
    ca, sa = cos(r), sin(r)
    _stack.append((a*ca + b*sa, -a*sa + b*ca, c*ca + d*sa, -c*sa + d*ca, e, f))
    try:
        yield
    finally:
        _stack.pop()


@contextmanager
def scale(sx: float, sy: float = None):
    """
    Use with transform
    """
    if sy is None:
        sy = sx
    a, b, c, d, e, f = _stack[-1]
    _stack.append((a*sx, b*sy, c*sx, d*sy, e, f))
    try:
        yield
    finally:
        _stack.pop()


def draw_line(x1, y1, x2, y2, color=arcade.color.WHITE, width=1):
    """
    We implement our own draw_line to allow for the transform.
    """
    X1, Y1 = _apply(x1, y1)
    X2, Y2 = _apply(x2, y2)
    arcade.draw_line(X1, Y1, X2, Y2, color, width)


def noise(x, repeat=1024, base=0):
    """
    p5.js-style noise() wrapper.
    - x: input value (float)
    - repeat: optional repeat period
    - base: seed offset
    Returns: float in [0, 1]
    """
    val = pnoise1(x, repeat=repeat, base=base)  # [-1, 1]
    return (val + 1) / 2


class Boundry():
    """
    Bascially a line (wall)
    """
    def __init__(self, x1, y1, x2, y2):
        self.a = arcade.Vec2(x1, y1)
        self.b = arcade.Vec2(x2, y2)

    def show(self):
        arcade.draw_line(
            self.a.x, self.a.y,
            self.b.x, self.b.y,
            color=arcade.color.WHITE,
            line_width=1
        )


class Ray():
    """
    A single raycast from an objects position
    Heading would be 0 - 360 (in radians)
    """
    def __init__(self, pos: arcade.Vec2, angle_r: float):
        self.pos = pos
        self.dir = arcade.Vec2.from_polar(angle_r)

    def show(self):
        """
        Displays the ray on the screen
        """
        with translate(self.pos):
            draw_line(
                0, 0,
                self.dir.x * 10, self.dir.y * 10)

    def cast(self, obj: Boundry) -> Optional[arcade.Vec2]:
        """
        Returns the intersection point if the ray
        casts on the obj or None.

        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        """
        # L1
        x1 = obj.a.x
        y1 = obj.a.y
        x2 = obj.b.x
        y2 = obj.b.y

        # L2
        x3 = self.pos.x
        y3 = self.pos.y
        x4 = self.pos.x + self.dir.x
        y4 = self.pos.y + self.dir.y

        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if d == 0:
            # Ray and object are paralle to each other
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        u = -(((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d)

        if t > 0 and t <= 1 and u > 0:
            return arcade.Vec2(
                (x1 + t * (x2 - x1)),
                (y1 + t * (y2 - y1))
            )

        # Ray doesn't intersect
        return None

    def look_at(self, d: arcade.Vec2):
        """
        """
        self.dir = arcade.Vec2(
            d.x - self.pos.x,
            d.y - self.pos.y).normalize()

    def move_to(self, p: arcade.Vec2):
        """
        """
        self.pos = p


class Particle():
    """
    """
    def __init__(self, deg_between_rays: float):
        """
        """
        self.pos = arcade.Vec2(
            WINDOW_WIDTH / 2,
            WINDOW_HEIGHT / 2
        )
        self.rays = []
        self.ray_lines = []

        for a in np.arange(0, 360, deg_between_rays):
            self.rays.append(
                Ray(self.pos, radians(a)))

    def show(self):
        """
        """
        # Draw the particle.
        arcade.draw_ellipse_filled(
            self.pos.x, self.pos.y, 16, 16, color=arcade.color.WHITE)

        for line in self.ray_lines:
            arcade.draw_line(
                self.pos.x, self.pos.y,
                line.x, line.y,
                color=(200, 200, 200, 200))

    def look(self, bs: List[Boundry]):
        '''
        '''
        self.ray_lines.clear()

        for ri, r in enumerate(self.rays):
            record = inf
            closest = None
            for bi, b in enumerate(bs):
                pt = r.cast(b)
                if pt:
                    dist = self.pos.distance(pt)
                    if dist < record:
                        record = dist
                        closest = pt
            if closest:
                self.ray_lines.append(closest)

    def move_to(self, pos: arcade.Vec2):
        '''
        '''
        self.pos = pos
        for ray in self.rays:
            ray.move_to(self.pos)


class GameView(arcade.View):
    """
    """
    def __init__(self):
        super().__init__()

        self.background_color = arcade.color.BLACK

        self.reset()

    def reset(self):
        """
        Reset the game to the inital state.
        """
        self.num_walls = 5
        self.ray_degs = 1

        self.t = 0.0

        self.walls = []

        for w in range(0, self.num_walls):
            x1 = random.randint(0, WINDOW_WIDTH)
            y1 = random.randint(0, WINDOW_HEIGHT)
            x2 = random.randint(0, WINDOW_WIDTH)
            y2 = random.randint(0, WINDOW_HEIGHT)
            self.walls.append(Boundry(x1, y1, x2, y2))

        # Add in walls on the edges
        self.walls.append(Boundry(0, 0,
                                  0, WINDOW_HEIGHT))
        self.walls.append(Boundry(0, WINDOW_HEIGHT,
                                  WINDOW_WIDTH, WINDOW_HEIGHT))
        self.walls.append(Boundry(WINDOW_WIDTH, WINDOW_HEIGHT,
                                  WINDOW_WIDTH, 0))
        self.walls.append(Boundry(WINDOW_WIDTH, 0,
                                  0, 0))

        self.particle = Particle(self.ray_degs)

    def on_draw(self):
        """
        Render the screen
        """
        self.clear()

        for w in self.walls:
            w.show()

        self.particle.show()

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you will call update() on the sprite lists that
        need it.
        """
        self.t += 0.01
        self.particle.move_to(
            arcade.Vec2(
                noise(self.t) * WINDOW_WIDTH,
                noise(self.t + 100) * WINDOW_HEIGHT))
        self.particle.look(self.walls)

    def on_key_press(self, key, key_modifiers):
        """
        Called whenever a key on the keyboard is pressed.
        For full list, see:
        https://api.arcade.academy/en/latest/arcade.key.html
        """
        pass

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        """
        Called whenever the mouse moves
        """
        #  self.particle.move_to(arcade.Vec2(x, y))
        pass

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button
        """
        pass

    def on_mouse_release(self, x, y, button, key_modifiers):
        """
        Called when a user releases a mouse button
        """
        pass


def main():
    # Create the window
    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)

    # Create and setup the GameView
    game = GameView()

    # Show it
    window.show_view(game)

    # Run the arcade loop
    arcade.run()


if __name__ == '__main__':
    main()
