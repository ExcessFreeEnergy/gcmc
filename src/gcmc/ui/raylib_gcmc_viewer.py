"""
Interactive 3D Grand Canonical Monte Carlo (GCMC) simulation viewer using Raylib.
"""

import sys
import os
import math
import argparse
import yaml
import numpy as np
import pyray as pr

from gcmc.v2 import GCMCSimulationV2, bindings
from gcmc.v1 import load_config
from .widgets import (
    draw_panel, draw_slider, draw_button, draw_toggle, draw_realtime_curve,
    COLOR_BG, COLOR_TEXT, COLOR_TEXT_DIM, COLOR_ACCENT
)


class GCMCInteractiveViewer:
    """
    3D Interactive Viewer for molecular GCMC simulations.
    """
    def __init__(self, config, input_folder=".", width=1280, height=720):
        self.config = config
        self.input_folder = input_folder
        self.width = width
        self.height = height

        self.sim = GCMCSimulationV2(self.config, input_folder=self.input_folder)

        # Simulation parameters
        self.is_running = True
        self.steps_per_frame = 50
        self.total_steps = 0
        self.T = float(self.config.get('T', 300.0))
        self.kB = float(self.config.get('kB', 1.380649e-23))

        particle_types = self.config.get('particle_types', {})
        mol_flag = self.config.get('molecule', 'None')
        if mol_flag == 'ABC':
            self.mu = float(particle_types.get('ABC', {}).get('mu', -8.0))
        elif mol_flag == 'H2O':
            self.mu = float(particle_types.get('H2O', {}).get('mu', -8.0))
        elif len(particle_types) > 0:
            k = list(particle_types.keys())[0]
            self.mu = float(particle_types[k].get('mu', -8.0))
        else:
            self.mu = -8.0

        self.box_x = float(self.config.get('box_length_x', 20.0))
        self.box_y = float(self.config.get('box_length_y', 20.0))
        self.box_z = float(self.config.get('box_length_z', 20.0))
        self.mol_type = mol_flag

        # History buffers for real-time graphs
        self.history_len = 120
        self.history_n = [float(self.sim.number)]
        self.history_energy = [float(self.sim.total_energy())]

        # 3D Orbit Camera setup
        self.camera = pr.Camera3D(
            pr.Vector3(self.box_x * 1.8, self.box_y * 1.8, self.box_z * 1.8), # position
            pr.Vector3(self.box_x * 0.5, self.box_y * 0.5, self.box_z * 0.5), # target
            pr.Vector3(0.0, 0.0, 1.0),                                        # up vector (z is up)
            45.0,                                                             # fov
            pr.CAMERA_PERSPECTIVE                                             # projection
        )
        self.cam_azimuth = 0.8
        self.cam_elevation = 0.5
        self.cam_dist = max(self.box_x, self.box_y, self.box_z) * 2.2
        self.last_mouse_pos = pr.Vector2(0, 0)
        self.is_dragging = False

    def handle_camera(self):
        mouse_pos = pr.get_mouse_position()
        viewport_w = self.width - 340 # Exclude right sidebar

        if mouse_pos.x < viewport_w:
            # Orbit rotation on left click drag
            if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
                self.is_dragging = True
                self.last_mouse_pos = mouse_pos

            if pr.is_mouse_button_released(pr.MOUSE_BUTTON_LEFT):
                self.is_dragging = False

            if self.is_dragging and pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
                dx = mouse_pos.x - self.last_mouse_pos.x
                dy = mouse_pos.y - self.last_mouse_pos.y
                self.cam_azimuth -= dx * 0.006
                self.cam_elevation += dy * 0.006
                self.cam_elevation = max(-math.pi / 2.2, min(math.pi / 2.2, self.cam_elevation))
                self.last_mouse_pos = mouse_pos

            # Zoom on mouse wheel
            wheel = pr.get_mouse_wheel_move()
            if wheel != 0:
                self.cam_dist -= wheel * (self.cam_dist * 0.1)
                self.cam_dist = max(5.0, min(200.0, self.cam_dist))

        # Update camera position in spherical coordinates around box center
        cx = self.box_x * 0.5
        cy = self.box_y * 0.5
        cz = self.box_z * 0.5

        px = cx + self.cam_dist * math.cos(self.cam_elevation) * math.cos(self.cam_azimuth)
        py = cy + self.cam_dist * math.cos(self.cam_elevation) * math.sin(self.cam_azimuth)
        pz = cz + self.cam_dist * math.sin(self.cam_elevation)

        self.camera.position = pr.Vector3(px, py, pz)
        self.camera.target = pr.Vector3(cx, cy, cz)

    def run_steps(self, num_steps):
        for _ in range(num_steps):
            bindings._lib.gcmc_v2_step(self.sim.handle)
            self.total_steps += 1

        cur_n = float(self.sim.number if hasattr(self.sim, 'number') else (self.sim.number1 + self.sim.number2))
        cur_e = float(self.sim.total_energy())

        self.history_n.append(cur_n)
        self.history_energy.append(cur_e)
        if len(self.history_n) > self.history_len:
            self.history_n.pop(0)
            self.history_energy.pop(0)

    def draw_3d_simulation(self):
        pr.begin_mode_3d(self.camera)

        # Draw 3D periodic bounding box
        box_center = pr.Vector3(self.box_x * 0.5, self.box_y * 0.5, self.box_z * 0.5)
        pr.draw_cube_wires(box_center, self.box_x, self.box_y, self.box_z, pr.Color(90, 110, 140, 180))

        # Draw grid floor
        pr.draw_grid(int(max(self.box_x, self.box_y)), 2.0)

        # Draw molecules
        c_sim = self.sim.handle
        num_mols = self.sim.number

        # Fetch molecular coordinates
        for idx in range(num_mols):
            if self.mol_type in ('ABC', 'H2O'):
                # 3-site molecule
                s0 = self.sim.get_site_pos(idx, 0)
                s1 = self.sim.get_site_pos(idx, 1)
                s2 = self.sim.get_site_pos(idx, 2)

                p0 = pr.Vector3(s0[0], s0[1], s0[2])
                p1 = pr.Vector3(s1[0], s1[1], s1[2])
                p2 = pr.Vector3(s2[0], s2[1], s2[2])

                if self.mol_type == 'H2O':
                    # SPC/E Water: Oxygen (Red), Hydrogen (White)
                    pr.draw_sphere(p0, 0.45, pr.Color(235, 60, 60, 255))
                    pr.draw_sphere(p1, 0.28, pr.Color(240, 240, 245, 255))
                    pr.draw_sphere(p2, 0.28, pr.Color(240, 240, 245, 255))
                    pr.draw_cylinder_ex(p0, p1, 0.1, 0.1, 8, pr.Color(180, 190, 205, 220))
                    pr.draw_cylinder_ex(p0, p2, 0.1, 0.1, 8, pr.Color(180, 190, 205, 220))
                else:
                    # ABC Dipole: A (Cyan), B (+q, Blue), C (-q, Magenta)
                    pr.draw_sphere(p0, 0.40, pr.Color(40, 210, 240, 255))
                    pr.draw_sphere(p1, 0.32, pr.Color(50, 120, 255, 255))
                    pr.draw_sphere(p2, 0.32, pr.Color(240, 50, 160, 255))
                    pr.draw_cylinder_ex(p1, p2, 0.1, 0.1, 8, pr.Color(160, 180, 200, 220))
            else:
                # 1-site Ion / Atom
                s0 = self.sim.get_site_pos(idx, 0)
                p0 = pr.Vector3(s0[0], s0[1], s0[2])
                sp_id = self.sim.get_molecule_species(idx)
                ion_color = pr.Color(50, 130, 255, 255) if sp_id == 0 else pr.Color(40, 210, 100, 255)
                pr.draw_sphere(p0, 0.50, ion_color)

        pr.end_mode_3d()

    def draw_gui_sidebar(self):
        panel_x = self.width - 330
        panel_y = 10
        panel_w = 320
        panel_h = self.height - 20

        draw_panel(panel_x, panel_y, panel_w, panel_h, "  GCMC v2 Controls & Monitoring")

        py = panel_y + 38

        # 1. Playback Controls
        btn_w = 90
        btn_h = 28
        if draw_button(panel_x + 12, py, btn_w, btn_h, "PAUSE" if self.is_running else "RUN", self.is_running):
            self.is_running = not self.is_running

        if draw_button(panel_x + 110, py, btn_w, btn_h, "STEP +1"):
            self.run_steps(1)

        if draw_button(panel_x + 208, py, btn_w, btn_h, "RESET"):
            self.sim = GCMCSimulationV2(self.config, input_folder=self.input_folder)
            self.total_steps = 0
            self.history_n = [float(self.sim.number)]
            self.history_energy = [float(self.sim.total_energy())]

        py += 38

        # 2. Speed Slider
        self.steps_per_frame = int(draw_slider(
            panel_x + 14, py, panel_w - 28, 30,
            "MC Steps / Frame", float(self.steps_per_frame), 1.0, 500.0, "%d"
        ))
        py += 44

        # 3. Chemical Potential Slider
        new_mu = draw_slider(
            panel_x + 14, py, panel_w - 28, 30,
            "Chemical Potential (mu/kBT)", self.mu, -15.0, -2.0, "%.2f"
        )
        if abs(new_mu - self.mu) > 1e-4:
            self.mu = new_mu
            bindings._lib.gcmc_v2_set_thermo(self.sim.handle, self.T, self.kB, self.mu)
        py += 44

        # 4. Temperature Slider
        new_T = draw_slider(
            panel_x + 14, py, panel_w - 28, 30,
            "Temperature (K)", self.T, 200.0, 650.0, "%.1f K"
        )
        if abs(new_T - self.T) > 1e-3:
            self.T = new_T
            bindings._lib.gcmc_v2_set_thermo(self.sim.handle, self.T, self.kB, self.mu)
        py += 50

        # 5. Live Metric Displays
        cur_n = self.sim.number
        cur_e = self.sim.total_energy()
        density = cur_n / (self.box_x * self.box_y * self.box_z)

        stats = [
            ("Total Steps:", f"{self.total_steps:,}"),
            ("Particles (N):", f"{cur_n}"),
            ("Number Density:", f"{density:.4f} / A^3"),
            ("Total Energy:", f"{cur_e:.4e} J"),
        ]

        pr.draw_text("SYSTEM THERMODYNAMICS", panel_x + 14, py, 11, COLOR_TEXT_DIM)
        py += 16
        for lbl, val in stats:
            pr.draw_text(lbl, panel_x + 14, py, 12, COLOR_TEXT_DIM)
            vw = pr.measure_text(val, 12)
            pr.draw_text(val, panel_x + panel_w - 14 - vw, py, 12, COLOR_TEXT)
            py += 18

        py += 10

        # 6. Real-time N(t) curve
        draw_realtime_curve(
            panel_x + 12, py, panel_w - 24, 110,
            self.history_n, title="Particle Number N(t)", color=pr.Color(50, 170, 255, 255)
        )
        py += 120

        # 7. Real-time Energy curve
        draw_realtime_curve(
            panel_x + 12, py, panel_w - 24, 110,
            self.history_energy, title="Total Energy U(t) [J]", color=pr.Color(255, 130, 50, 255)
        )

    def main_loop(self):
        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_RESIZABLE)
        pr.init_window(self.width, self.height, "GCMC v2 - 3D Interactive Molecular Simulation")
        pr.set_target_fps(60)

        while not pr.window_should_close():
            self.handle_camera()

            if self.is_running:
                self.run_steps(self.steps_per_frame)

            pr.begin_drawing()
            pr.clear_background(COLOR_BG)

            self.draw_3d_simulation()

            # 2D HUD text
            fps = pr.get_fps()
            pr.draw_text(f"FPS: {fps} | Orbit: Left Mouse Drag | Zoom: Scroll Wheel", 15, 12, 13, COLOR_TEXT_DIM)

            self.draw_gui_sidebar()

            pr.end_drawing()

        pr.close_window()


def launch_interactive_gcmc(config, folder="."):
    """Launch interactive 3D GCMC viewer."""
    viewer = GCMCInteractiveViewer(config, input_folder=folder)
    viewer.main_loop()


def cli():
    parser = argparse.ArgumentParser(description="Launch interactive 3D GCMC simulation viewer.")
    parser.add_argument("-in", "--input_folder", type=str, default=".", help="Path to input directory.")
    args = parser.parse_args()

    config_path = os.path.join(args.input_folder, "input.yaml")
    if not os.path.exists(config_path):
        # Fallback to dipole fast config if available
        test_cfg = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "v1", "test_configs", "dipole_fast", "input.yaml")
        if os.path.exists(test_cfg):
            config_path = test_cfg
            args.input_folder = os.path.dirname(test_cfg)
        else:
            print(f"Error: Could not find input.yaml in '{args.input_folder}'", file=sys.stderr)
            sys.exit(1)

    config = load_config(config_path)
    launch_interactive_gcmc(config, args.input_folder)


if __name__ == "__main__":
    cli()
