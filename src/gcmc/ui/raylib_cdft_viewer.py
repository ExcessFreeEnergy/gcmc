"""
Interactive 2D/3D cDFT Fluid Manipulation & Reinforcement Learning Viewer using Raylib.
Allows real-time dielectrocapillarity parameter tweaking and active policy control.
"""

import argparse
import os
import sys

import numpy as np
import pyray as pr

from gcmc.envs.cdft_puffer import CdftFluidEnv

from .widgets import (
    COLOR_BG,
    COLOR_BORDER,
    draw_button,
    draw_panel,
    draw_realtime_curve,
    draw_slider,
)


class CDFTInteractiveViewer:
    """
    Interactive GUI for neural classical density functional theory (cDFT)
    dielectrocapillarity control and RL policy evaluation.
    """

    def __init__(self, policy_path=None, width=1280, height=720):
        self.width = width
        self.height = height
        self.policy_path = policy_path

        # Initialize Environment
        self.env = CdftFluidEnv()
        self.obs, self.info = self.env.reset()

        # Control States
        self.is_auto_running = True
        self.is_agent_controlled = False
        self.policy = None

        if self.policy_path and os.path.exists(self.policy_path):
            self.load_policy(self.policy_path)

        # Manual control parameters
        self.phi_0 = 15.0  # Volts
        self.mode_m = 1.0  # Spatial harmonic
        self.v_bias = 0.0  # Bias offset
        self.target_filling = self.env.target_filling

        # History tracking
        self.history_len = 120
        self.history_rewards = [0.0]
        self.history_fillings = [float(self.env.current_filling)]

        # Slit Grid Dimensions
        self.nz = 50
        self.z_grid = np.linspace(-37.5, 37.5, self.nz)

    def load_policy(self, path):
        try:
            import torch

            from gcmc.envs.train_pufferl import CdftPolicy

            self.policy = CdftPolicy(obs_dim=106, act_dim=3)
            checkpoint = torch.load(path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.policy.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.policy.load_state_dict(checkpoint)
            self.policy.eval()
            self.is_agent_controlled = True
            print(f"Successfully loaded trained policy from {path}")
        except Exception as e:
            print(f"Warning: Could not load policy from '{path}': {e}", file=sys.stderr)
            self.policy = None
            self.is_agent_controlled = False

    def step_simulation(self):
        # Keep user target in sync with environment
        self.env.target_filling = self.target_filling

        if self.is_agent_controlled and self.policy is not None:
            import torch

            with torch.no_grad():
                obs_t = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
                action = self.policy(obs_t).squeeze(0).numpy()
                action = np.clip(action, -1.0, 1.0)
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            # Update UI sliders from environment state
            self.phi_0 = float(self.env.phi_0)
            self.mode_m = float(round(self.env.mode_m))
            self.v_bias = float(self.env.v_bias)
        else:
            # Action computed from manual slider adjustments
            d_phi = float(np.clip((self.phi_0 - self.env.phi_0) / 4.0, -1.0, 1.0))
            d_m = float(np.clip((self.mode_m - self.env.mode_m) / 0.5, -1.0, 1.0))
            d_bias = float(np.clip((self.v_bias - self.env.v_bias) / 2.0, -1.0, 1.0))
            action = np.array([d_phi, d_m, d_bias], dtype=np.float32)
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)

        if terminated or truncated:
            cur_target = self.target_filling
            self.obs, self.info = self.env.reset()
            self.env.target_filling = cur_target

        self.history_rewards.append(float(reward))
        self.history_fillings.append(float(self.env.current_filling))
        if len(self.history_rewards) > self.history_len:
            self.history_rewards.pop(0)
            self.history_fillings.pop(0)

    def draw_fluid_slit_channel(self, x, y, w, h):
        """
        Draws 2D slit pore showing liquid density profile, meniscus, and electrodes.
        """
        draw_panel(x, y, w, h, "  Slit Pore Fluid Density & Meniscus")

        inner_x = x + 16
        inner_y = y + 36
        inner_w = w - 32
        inner_h = h - 52

        # Draw electrode boundary walls
        pr.draw_rectangle(inner_x, inner_y, inner_w, 14, pr.Color(160, 175, 195, 255))
        pr.draw_rectangle_lines(inner_x, inner_y, inner_w, 14, COLOR_BORDER)
        pr.draw_text("Top Electrode Wall (z = +37.5 A)", inner_x + 10, inner_y + 2, 10, pr.BLACK)

        bot_wall_y = inner_y + inner_h - 14
        pr.draw_rectangle(inner_x, bot_wall_y, inner_w, 14, pr.Color(160, 175, 195, 255))
        pr.draw_rectangle_lines(inner_x, bot_wall_y, inner_w, 14, COLOR_BORDER)
        pr.draw_text("Bottom Electrode Wall (z = -37.5 A)", inner_x + 10, bot_wall_y + 2, 10, pr.BLACK)

        slit_y = inner_y + 14
        slit_h = inner_h - 28

        # Draw density column gradient along z
        rho = self.env.density_profile
        bin_h = slit_h / float(self.nz)

        for i in range(self.nz):
            val = float(rho[i])
            norm_val = max(0.0, min(1.0, val))

            # Interpolate from vapor light blue to deep liquid blue
            r_col = int(25 + (1.0 - norm_val) * 40)
            g_col = int(80 + (1.0 - norm_val) * 100)
            b_col = int(180 + (1.0 - norm_val) * 75)
            a_col = int(90 + norm_val * 165)
            col = pr.Color(r_col, g_col, b_col, a_col)

            # Invert so index 0 (z=-37.5) is bottom, index nz-1 is top
            by = slit_y + slit_h - (i + 1) * bin_h
            pr.draw_rectangle(inner_x, int(by), inner_w, int(bin_h) + 1, col)

        # Draw electric field lines overlay
        m_val = int(self.env.mode_m)
        for i in range(m_val * 2 + 1):
            line_y = slit_y + int(i * (slit_h / (m_val * 2.0)))
            pr.draw_line(inner_x, line_y, inner_x + inner_w, line_y, pr.Color(255, 220, 60, 120))

        # Annotations
        filling = self.env.current_filling
        target = self.env.target_filling
        pr.draw_text(
            f"Pore Filling: {filling * 100:.1f}%  |  Target: {target * 100:.1f}%",
            inner_x + 12,
            slit_y + 10,
            13,
            pr.WHITE,
        )

    def draw_scientific_plots(self, x, y, w, h):
        """
        Draws live 2D Density profile rho(z) and Potential phi(z).
        """
        draw_panel(x, y, w, h, "  Classical Density Functional Profiles")

        py = y + 36
        plot_h = (h - 50) // 2

        # 1. Density Profile rho(z) in [0.0, 1.0]
        rho = self.env.density_profile
        draw_realtime_curve(
            x + 10,
            py,
            w - 20,
            plot_h,
            rho,
            title="Fluid Density Profile rho(z) [normalized liquid fraction]",
            color=pr.Color(50, 180, 255, 255),
            fixed_min=0.0,
            fixed_max=1.0,
        )

        py += plot_h + 8

        # 2. Applied Electrostatic Potential phi(z)
        z = np.linspace(-37.5, 37.5, self.nz)
        Lz = 75.0
        m = self.env.mode_m
        phi_0 = self.env.phi_0
        bias = self.env.v_bias
        phi_z = (phi_0 / m) * np.cos(2.0 * np.pi * m * z / Lz) + bias

        draw_realtime_curve(
            x + 10,
            py,
            w - 20,
            plot_h,
            phi_z,
            title="Electrostatic Potential phi(z) [V]",
            color=pr.Color(255, 180, 40, 255),
        )

    def draw_control_panel(self, x, y, w, h):
        """
        Draws interactive sliders and control buttons.
        """
        draw_panel(x, y, w, h, "  Dielectrocapillarity Controls")

        py = y + 36

        # 1. Mode Selector (Human vs RL Agent)
        if draw_button(x + 12, py, (w - 30) // 2, 28, "MANUAL CONTROL", not self.is_agent_controlled):
            self.is_agent_controlled = False

        if draw_button(x + 16 + (w - 30) // 2, py, (w - 30) // 2, 28, "RL AGENT ACTIVE", self.is_agent_controlled):
            if self.policy is None and self.policy_path:
                self.load_policy(self.policy_path)
            self.is_agent_controlled = True

        py += 36

        # 2. Play / Step buttons
        btn_w = (w - 36) // 3
        if draw_button(x + 12, py, btn_w, 28, "PAUSE" if self.is_auto_running else "RUN", self.is_auto_running):
            self.is_auto_running = not self.is_auto_running

        if draw_button(x + 16 + btn_w, py, btn_w, 28, "STEP +1"):
            self.step_simulation()

        if draw_button(x + 20 + 2 * btn_w, py, btn_w, 28, "RESET"):
            self.obs, self.info = self.env.reset()
            self.history_rewards = [0.0]
            self.history_fillings = [float(self.env.current_filling)]

        py += 38

        # 3. Voltage Amplitude Slider
        self.phi_0 = draw_slider(
            x + 14, py, w - 28, 30, "Voltage Amplitude phi_0 (V)", self.phi_0, -38.2, 38.2, "%.1f V"
        )
        py += 44

        # 4. Spatial Mode m Slider
        self.mode_m = draw_slider(x + 14, py, w - 28, 30, "Spatial Harmonic Mode m", self.mode_m, 1.0, 4.0, "m = %d")
        self.mode_m = round(self.mode_m)
        py += 44

        # 5. DC Bias Slider
        self.v_bias = draw_slider(
            x + 14, py, w - 28, 30, "DC Bias Offset V_bias (V)", self.v_bias, -10.0, 10.0, "%.1f V"
        )
        py += 44

        # 6. Target Filling Fraction
        new_target = draw_slider(
            x + 14, py, w - 28, 30, "Target Pore Filling Fraction", self.target_filling, 0.10, 0.90, "%.2f"
        )
        if abs(new_target - self.target_filling) > 0.01:
            self.target_filling = new_target
            self.env.target_filling = new_target
        py += 50

        # 7. Real-time Filling and Reward Charts
        draw_realtime_curve(
            x + 12,
            py,
            w - 24,
            100,
            self.history_fillings,
            title="Pore Filling theta(t)",
            color=pr.Color(50, 220, 120, 255),
            ref_val=self.env.target_filling,
            fixed_min=0.0,
            fixed_max=1.0,
        )
        py += 108

        draw_realtime_curve(
            x + 12, py, w - 24, 100, self.history_rewards, title="RL Step Reward R(t)", color=pr.Color(240, 70, 70, 255)
        )

    def main_loop(self):
        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_RESIZABLE)
        pr.init_window(self.width, self.height, "cDFT Fluid Manipulation - Interactive RL Control")
        pr.set_target_fps(60)

        while not pr.window_should_close():
            if self.is_auto_running:
                self.step_simulation()

            pr.begin_drawing()
            pr.clear_background(COLOR_BG)

            # Left column: 2D Slit Fluid Channel (w = 380)
            self.draw_fluid_slit_channel(12, 12, 380, self.height - 24)

            # Middle column: Real-time cDFT Scientific plots (w = 460)
            self.draw_scientific_plots(402, 12, 460, self.height - 24)

            # Right column: Interactive Controls & Policy (w = 390)
            self.draw_control_panel(872, 12, self.width - 884, self.height - 24)

            pr.end_drawing()

        pr.close_window()


def launch_interactive_cdft_rl(policy_path=None):
    """Launch interactive cDFT fluid manipulation viewer."""
    viewer = CDFTInteractiveViewer(policy_path=policy_path)
    viewer.main_loop()


def cli():
    parser = argparse.ArgumentParser(description="Launch interactive cDFT fluid manipulation viewer.")
    parser.add_argument("--policy", type=str, default=None, help="Path to trained PyTorch policy checkpoint.")
    args = parser.parse_args()

    launch_interactive_cdft_rl(policy_path=args.policy)


if __name__ == "__main__":
    cli()
