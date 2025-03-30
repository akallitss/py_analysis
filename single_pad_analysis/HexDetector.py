#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on March 29 7:38 PM 2025
Created in PyCharm
Created as pico_py_analysis/HexDetector.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt


class HexPad:
    def __init__(self, inner_radius, x=0, y=0):
        self.inner_radius = inner_radius
        self.x = x
        self.y = y

    def __repr__(self):
        return f"HexPad(inner_radius={self.inner_radius}, x={self.x:.2f}, y={self.y:.2f})"


class HexDetector:
    def __init__(self, pad_inner_radii, x=0, y=0, rotation=0):
        self.hex_pads = []
        self.pad_inner_radii = pad_inner_radii
        self.x = x
        self.y = y
        self.rotation = rotation

        self.add_pad()

    def set_rotation(self, rotation):
        self.rotation = rotation

    def add_pad(self, reference_pad_index=None, dx_index=1, dy_index=0):
        """Tile a new pad from an existing reference pad. If reference_pad_index is None,
        add the first pad at the origin. dx_index and dy_index define the position of the new pad.
        There are 6 possible dx, dy combinations that will tile the new pad around the reference pad."""
        sqrt3 = np.sqrt(3)
        offsets = {
            (1, 0): (2, 0),
            (-1, 0): (-2, 0),
            (1, 1): (1, sqrt3),
            (1, -1): (1, -sqrt3),
            (-1, 1): (-1, sqrt3),
            (-1, -1): (-1, -sqrt3),
        }

        if reference_pad_index is None:
            if not self.hex_pads:
                self.hex_pads.append(HexPad(self.pad_inner_radii, 0, 0))
            else:
                raise ValueError("No reference pad index given, but first pad already exist.")
            return

        if (dx_index, dy_index) not in offsets:
            raise ValueError("Invalid dx_index, dy_index combination. Must be one of the six valid hex neighbors.")

        ref_pad = self.hex_pads[reference_pad_index]
        dx, dy = offsets[(dx_index, dy_index)]
        new_x = ref_pad.x + dx * self.pad_inner_radii
        new_y = ref_pad.y + dy * self.pad_inner_radii

        if any(np.isclose(pad.x, new_x) and np.isclose(pad.y, new_y) for pad in self.hex_pads):
            raise ValueError("Pad to be added already exists.")

        self.hex_pads.append(HexPad(self.pad_inner_radii, new_x, new_y))

    def get_pad_center(self, pad_index):
        """Get the center position of a pad, applying the global rotation and translation."""
        pad = self.hex_pads[pad_index]
        cos_theta = np.cos(self.rotation)
        sin_theta = np.sin(self.rotation)
        rotated_x = cos_theta * pad.x - sin_theta * pad.y
        rotated_y = sin_theta * pad.x + cos_theta * pad.y
        return rotated_x + self.x, rotated_y + self.y

    def plot_detector(self, global_coords=False):
        """Plot the hexagonal detector layout using matplotlib."""
        fig, ax = plt.subplots()
        for i, pad in enumerate(self.hex_pads):
            if global_coords:
                x, y = self.get_pad_center(i)
                rotation = self.rotation
            else:
                x, y = pad.x, pad.y
                rotation = 0
            hexagon = plt.Polygon(hexagon_vertices(x, y, self.pad_inner_radii, rotation), edgecolor='black',
                                  facecolor='lightblue', alpha=0.5)
            ax.add_patch(hexagon)
            ax.scatter(x, y, s=1, color='black')

        ax.set_aspect('equal')
        extra_scale = 1.2
        ax.set_xlim(min(p.x for p in self.hex_pads) - hex_radius_inner_to_outer(self.pad_inner_radii) * extra_scale + (self.x if global_coords else 0),
                    max(p.x for p in self.hex_pads) + hex_radius_inner_to_outer(self.pad_inner_radii) * extra_scale + (self.x if global_coords else 0))
        ax.set_ylim(min(p.y for p in self.hex_pads) - hex_radius_inner_to_outer(self.pad_inner_radii) * extra_scale + (self.y if global_coords else 0),
                    max(p.y for p in self.hex_pads) + hex_radius_inner_to_outer(self.pad_inner_radii) * extra_scale + (self.y if global_coords else 0))
        plt.show()

    def __repr__(self):
        return f"HexDetector with {len(self.hex_pads)} pads:\n" + "\n".join(map(str, self.hex_pads))


# def hexagon_vertices(x, y, r_inner):
#     """Compute the vertices of a hexagon centered at (x, y) with given inner radius r."""
#     r_outer = hex_radius_inner_to_outer(r_inner)
#     return [(x + r_outer * np.cos(theta + np.pi / 6), y + r_outer * np.sin(theta + np.pi / 6)) for theta in
#             np.linspace(0, 2 * np.pi, 7)]


def hexagon_vertices(x, y, r_inner, rotation=0):
    """Compute the vertices of a hexagon centered at (x, y) with given inner radius r_inner, rotated accordingly."""
    r_outer = (2 / np.sqrt(3)) * r_inner  # Convert inner radius to outer radius
    return [(x + r_outer * np.cos(theta + np.pi / 6 + rotation), y + r_outer * np.sin(theta + np.pi / 6 + rotation))
            for theta in np.linspace(0, 2 * np.pi, 7)]


def hex_radius_inner_to_outer(r_inner):
    """Convert inner radius to outer radius of a hexagon."""
    return (2 / np.sqrt(3)) * r_inner


def hex_radius_outer_to_inner(r_outer):
    """Convert outer radius to inner radius of a hexagon."""
    return (np.sqrt(3) / 2) * r_outer
