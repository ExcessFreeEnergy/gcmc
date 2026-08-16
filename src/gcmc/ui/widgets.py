"""
Immediate-mode scientific UI widgets for Raylib (pyray).
"""

import math
import pyray as pr
import numpy as np

# Scientific Dark Theme Palette
COLOR_BG = pr.Color(18, 22, 28, 255)
COLOR_PANEL = pr.Color(26, 32, 42, 240)
COLOR_PANEL_HEADER = pr.Color(35, 43, 56, 255)
COLOR_BORDER = pr.Color(50, 62, 80, 255)
COLOR_TEXT = pr.Color(220, 230, 242, 255)
COLOR_TEXT_DIM = pr.Color(130, 145, 165, 255)
COLOR_ACCENT = pr.Color(45, 125, 240, 255)
COLOR_ACCENT_HOVER = pr.Color(65, 145, 255, 255)
COLOR_BUTTON = pr.Color(40, 50, 65, 255)
COLOR_BUTTON_HOVER = pr.Color(55, 68, 88, 255)
COLOR_BUTTON_ACTIVE = pr.Color(45, 125, 240, 255)
COLOR_SLIDER_TRACK = pr.Color(35, 42, 54, 255)
COLOR_SLIDER_FILL = pr.Color(45, 125, 240, 255)
COLOR_SLIDER_KNOB = pr.Color(230, 240, 255, 255)
COLOR_GRID = pr.Color(38, 46, 60, 255)


def draw_panel(x, y, w, h, title=""):
    """Draw a dark container panel with an optional header."""
    pr.draw_rectangle(x, y, w, h, COLOR_PANEL)
    pr.draw_rectangle_lines(x, y, w, h, COLOR_BORDER)
    if title:
        pr.draw_rectangle(x, y, w, 28, COLOR_PANEL_HEADER)
        pr.draw_rectangle_lines(x, y, w, 28, COLOR_BORDER)
        pr.draw_text(title, x + 10, y + 6, 14, COLOR_TEXT)


def draw_slider(x, y, w, h, label, val, min_v, max_v, format_str="%.2f"):
    """
    Draws an immediate-mode slider and returns the updated value.
    """
    mouse_pos = pr.get_mouse_position()
    mouse_down = pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT)

    track_y = y + 18
    track_h = 8
    val_norm = (val - min_v) / (max_v - min_v + 1e-12)
    val_norm = max(0.0, min(1.0, val_norm))

    knob_x = x + int(val_norm * w)
    knob_y = track_y + track_h // 2
    knob_r = 7

    # Check hover / dragging
    slider_rect = pr.Rectangle(x - 5, track_y - 6, w + 10, track_h + 12)
    is_hover = pr.check_collision_point_rec(mouse_pos, slider_rect)

    if is_hover and mouse_down:
        rel_x = (mouse_pos.x - x) / float(w)
        val_norm = max(0.0, min(1.0, rel_x))
        val = min_v + val_norm * (max_v - min_v)

    # Render label and current value
    pr.draw_text(label, x, y, 12, COLOR_TEXT_DIM)
    val_text = format_str % val
    text_w = pr.measure_text(val_text, 12)
    pr.draw_text(val_text, x + w - text_w, y, 12, COLOR_TEXT)

    # Render track
    pr.draw_rectangle_rounded(pr.Rectangle(x, track_y, w, track_h), 0.5, 4, COLOR_SLIDER_TRACK)
    fill_w = max(4, int(val_norm * w))
    pr.draw_rectangle_rounded(pr.Rectangle(x, track_y, fill_w, track_h), 0.5, 4, COLOR_SLIDER_FILL)

    # Render knob
    knob_color = COLOR_SLIDER_KNOB if not is_hover else pr.WHITE
    pr.draw_circle(knob_x, knob_y, knob_r, knob_color)
    pr.draw_circle_lines(knob_x, knob_y, knob_r, COLOR_BORDER)

    return val


def draw_button(x, y, w, h, text, is_active=False):
    """
    Draws a clickable button and returns True if clicked in this frame.
    """
    mouse_pos = pr.get_mouse_position()
    is_hover = pr.check_collision_point_rec(mouse_pos, pr.Rectangle(x, y, w, h))
    clicked = is_hover and pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT)

    bg_color = COLOR_BUTTON_ACTIVE if is_active else (COLOR_BUTTON_HOVER if is_hover else COLOR_BUTTON)
    pr.draw_rectangle_rounded(pr.Rectangle(x, y, w, h), 0.25, 4, bg_color)
    pr.draw_rectangle_lines_ex(pr.Rectangle(x, y, w, h), 1, COLOR_BORDER)

    text_w = pr.measure_text(text, 14)
    pr.draw_text(text, x + (w - text_w) // 2, y + (h - 14) // 2, 14, pr.WHITE if (is_active or is_hover) else COLOR_TEXT)
    return clicked


def draw_toggle(x, y, w, h, label, state):
    """
    Draws a switch toggle and returns the new state.
    """
    mouse_pos = pr.get_mouse_position()
    switch_w = 34
    switch_h = 18
    switch_x = x + w - switch_w
    switch_y = y + (h - switch_h) // 2

    is_hover = pr.check_collision_point_rec(mouse_pos, pr.Rectangle(x, y, w, h))
    if is_hover and pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
        state = not state

    pr.draw_text(label, x, y + 3, 13, COLOR_TEXT)

    # Switch track
    track_col = COLOR_ACCENT if state else COLOR_SLIDER_TRACK
    pr.draw_rectangle_rounded(pr.Rectangle(switch_x, switch_y, switch_w, switch_h), 0.5, 4, track_col)

    # Knob
    knob_x = switch_x + (switch_w - 9) if state else (switch_x + 9)
    pr.draw_circle(knob_x, switch_y + switch_h // 2, 7, pr.WHITE)

    return state


def draw_realtime_curve(x, y, w, h, y_data, title="", x_label="", y_label="", color=None, ref_val=None):
    """
    Draws a 2D line plot with axes, auto-scaling, and grid lines.
    """
    if color is None:
        color = COLOR_ACCENT

    pr.draw_rectangle(x, y, w, h, pr.Color(16, 20, 26, 255))
    pr.draw_rectangle_lines(x, y, w, h, COLOR_BORDER)

    # Title
    if title:
        pr.draw_text(title, x + 8, y + 6, 12, COLOR_TEXT)

    if len(y_data) < 2:
        return

    min_v = float(np.min(y_data))
    max_v = float(np.max(y_data))
    if abs(max_v - min_v) < 1e-6:
        max_v += 1.0
        min_v -= 1.0

    margin_top = 22
    margin_bottom = 18
    margin_left = 38
    margin_right = 10

    plot_x = x + margin_left
    plot_y = y + margin_top
    plot_w = w - margin_left - margin_right
    plot_h = h - margin_top - margin_bottom

    # Grid lines (3 horizontal)
    for g in range(3):
        gy = plot_y + int(g * (plot_h / 2.0))
        pr.draw_line(plot_x, gy, plot_x + plot_w, gy, COLOR_GRID)
        g_val = max_v - g * 0.5 * (max_v - min_v)
        val_str = f"{g_val:.2f}"
        pr.draw_text(val_str, x + 2, gy - 5, 10, COLOR_TEXT_DIM)

    # Reference target line
    if ref_val is not None and min_v <= ref_val <= max_v:
        ref_norm = (ref_val - min_v) / (max_v - min_v)
        ref_py = plot_y + plot_h - int(ref_norm * plot_h)
        pr.draw_line(plot_x, ref_py, plot_x + plot_w, ref_py, pr.Color(240, 180, 40, 200))

    # Curve points
    num_pts = len(y_data)
    for i in range(num_pts - 1):
        x1 = plot_x + int(i * (plot_w / (num_pts - 1)))
        y1_norm = (y_data[i] - min_v) / (max_v - min_v)
        py1 = plot_y + plot_h - int(y1_norm * plot_h)

        x2 = plot_x + int((i + 1) * (plot_w / (num_pts - 1)))
        y2_norm = (y_data[i + 1] - min_v) / (max_v - min_v)
        py2 = plot_y + plot_h - int(y2_norm * plot_h)

        pr.draw_line(x1, py1, x2, py2, color)
        pr.draw_line(x1, py1 + 1, x2, py2 + 1, color)
