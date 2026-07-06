import tkinter as tk
import tkinter.font as tkfont


class SimulationModelDiagram:
    """Reusable canvas diagram that reflects current simulation setting toggles."""

    def __init__(self, parent, width=520, height=260, bg="white"):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self._vars = {}
        self._last_size = (width, height)
        self.canvas.bind("<Configure>", self._on_resize)

    def bind_to_vars(self, **setting_vars):
        """Bind Tk variables by setting-name and redraw when any changes."""
        self._vars = dict(setting_vars)
        for var in self._vars.values():
            try:
                var.trace_add("write", self._on_var_change)
            except Exception:
                pass
        self.redraw()

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w = max(260, int(c.winfo_width() or self._last_size[0]))
        h = max(160, int(c.winfo_height() or self._last_size[1]))
        self._last_size = (w, h)

        state = self._state()
        m1_facil = bool(state["enable_m1_facilitated_diffusion"])
        m1_porin = bool(state["enable_m1_porin_diffusion"])
        m1_simple = bool(state["enable_m1_diffusion"])
        if m1_facil:
            m1_mode = "facilitated"
        elif m1_porin:
            m1_mode = "porin"
        elif m1_simple:
            m1_mode = "simple"
        else:
            m1_mode = "off"
        m1_diff = (m1_mode != "off")
        m1_import_only = m1_mode in {"facilitated", "porin"}
        m2_diff = state["enable_m2_diffusion"]

        margin_x = int(w * 0.07)
        top = int(h * 0.20)
        bottom = int(h * 0.58)
        r = max(10, int(min(w, h) * 0.045))

        x_m1 = margin_x + int((w - 2 * margin_x) * 0.10)
        x_m2 = margin_x + int((w - 2 * margin_x) * 0.48)
        x_energy = margin_x + int((w - 2 * margin_x) * 0.84)

        label_font = ("Arial", max(8, int(r * 0.70)))
        note_font = ("Arial", max(8, int(r * 0.62)), "italic")
        task_font = ("Arial", max(8, int(r * 0.70)), "bold")

        def draw_node(x, y, label, fill, outline, text_dx=0, text_dy=0):
            c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)
            c.create_text(x + text_dx, y + text_dy, text=label, font=label_font, fill="#222222")

        def draw_env_inflow(x_target, y_target, label, color="#333333"):
            # Keep top env nodes with top-down inflow; bottom env nodes use right-side inflow.
            y_mid = (top + bottom) / 2.0
            if y_target <= y_mid:
                c.create_line(x_target, y_target - r - 18, x_target, y_target - r, width=2, arrow=tk.LAST, fill=color)
                c.create_text(x_target, y_target - r - 24, text=label, font=label_font, fill=color)
            else:
                x_start = min(w - margin_x // 2, x_target + int(2.7 * r))
                x_end = x_target + r + 1
                c.create_line(x_start, y_target, x_end, y_target, width=2, arrow=tk.LAST, fill=color)
                c.create_text(x_start - int(0.5 * r), y_target - int(0.9 * r), text=label, font=label_font, fill=color)

        m1_env_y = bottom if m1_diff else top
        draw_node(x_m1, m1_env_y, "M1\n(env)", "#CFE9FF", "#4B90C9", text_dy=r + 10)
        draw_env_inflow(x_m1, m1_env_y, "M1 Inflow")

        if m1_diff:
            draw_node(x_m1, top, "M1\n(internal)", "#A8D7FF", "#2E6FA6", text_dy=-(r + 10))
            if m1_import_only:
                # Import-only: environment (bottom) -> internal (top)
                c.create_line(x_m1 - 2, bottom - r + 3, x_m1 - 2, top + r - 3, width=3, arrow=tk.LAST)
            else:
                c.create_line(x_m1, top + r - 3, x_m1, bottom - r + 3, width=2, arrow=tk.BOTH)
            c.create_text(
                x_m1 + int(2.4 * r),
                (top + bottom) // 2,
                text=(
                    "facilitated (import)"
                    if m1_mode == "facilitated"
                    else ("porin (import)" if m1_mode == "porin" else "diffusion (in/out)")
                ),
                font=note_font,
                fill="#444444",
            )

        if m2_diff:
            draw_node(x_m2, top, "M2\n(internal)", "#BFE8BF", "#3A8D3A", text_dy=-(r + 10))
            draw_node(x_m2, bottom, "M2\n(env)", "#D7F2D7", "#4FA84F", text_dy=r + 10)
            c.create_line(x_m2, top + r - 3, x_m2, bottom - r + 3, width=2, arrow=tk.BOTH)
            c.create_text(
                x_m2 + int(2.8 * r),
                (top + bottom) // 2,
                text="diffusion (in/out)",
                font=note_font,
                fill="#444444",
            )
            task2_source_y = top
            m2_env_y = bottom
        else:
            draw_node(x_m2, top, "M2\n(env)", "#D7F2D7", "#4FA84F", text_dx=r + 24)
            c.create_text(x_m2 + int(2.8 * r), top + r + 14, text="no internal M2", font=note_font, fill="#444444")
            task2_source_y = top
            m2_env_y = top

        if state["enable_acetate_addition"]:
            draw_env_inflow(x_m2, m2_env_y, "Acetate Inflow", color="#2f6f2f")

        draw_node(x_energy, top, "Waste", "#FFD9A8", "#D58A2F", text_dy=-(r + 10))

        c.create_line(x_m1 + r, top, x_m2 - r, top, width=2, arrow=tk.LAST)
        c.create_text((x_m1 + x_m2) // 2, top - int(1.5 * r), text="A (Task 1)", font=task_font, fill="#1E5A8A")

        c.create_line(x_m2 + r, task2_source_y, x_energy - r, top, width=2, arrow=tk.LAST)
        c.create_text((x_m2 + x_energy) // 2, top - int(1.5 * r), text="B (Task 2)", font=task_font, fill="#4F6A15")

        badge_y_top = int(h * 0.80)
        allow_diffusion_mutation = bool(m1_diff or m2_diff)
        allow_intermediate_costs = bool(m2_diff)
        badges = [
            ("Homog. Init Genotype", state["homogeneous_population"], True),
            ("Independent A/B", state["independent_traits"], True),
            ("Chemostat Flow", state["enable_chemostat_flow"], True),
            ("Initial Energy", state["enable_initial_energy"], True),
            ("Intermed. Costs", state["enable_intermediate_costs"], allow_intermediate_costs),
            ("Diffusion Mutation", state["enable_diffusion_mutation"], allow_diffusion_mutation),
            (
                "Homog. Initial D",
                state["homogeneous_initial_diffusion_const"],
                allow_diffusion_mutation and state["enable_diffusion_mutation"],
            ),
        ]
        mid = (len(badges) + 1) // 2
        row1 = badges[:mid]
        row2 = badges[mid:]
        usable_w = w - 2 * margin_x
        badge_gap = max(8, int(r * 0.40))
        badge_font_size = max(7, int(r * 0.50))
        min_badge_font = 5 if w < 400 else 6

        def measure_badge_row(items, font_size):
            badge_font = ("Arial", font_size)
            font_obj = tkfont.Font(font=badge_font)
            pad_x = max(4 if w < 400 else 6, int(r * 0.35))
            pad_y = max(3 if w < 400 else 4, int(r * 0.25))
            sizes = []
            for name, *_ in items:
                text_w = int(font_obj.measure(name))
                text_h = int(font_obj.metrics("linespace"))
                sizes.append((text_w // 2 + pad_x, text_h // 2 + pad_y))
            total_w = sum(2 * half_w for half_w, _ in sizes) + badge_gap * max(0, len(items) - 1)
            return badge_font, sizes, total_w

        while badge_font_size >= min_badge_font:
            row_widths = [measure_badge_row(row, badge_font_size)[2] for row in (row1, row2) if row]
            if not row_widths or max(row_widths) <= usable_w:
                break
            badge_font_size -= 1

        badge_row_gap = max(18, int(h * 0.08))

        def draw_badge_row(items, y):
            if not items:
                return
            badge_font, sizes, total_w = measure_badge_row(items, badge_font_size)
            x_cursor = max(margin_x, (w - total_w) // 2)
            for (name, enabled, applicable), (half_w, half_h) in zip(items, sizes):
                x = x_cursor + half_w
                if enabled:
                    fill, outline, text_color = "#DFF4DF", "#79B879", "#2D6B2D"
                elif applicable:
                    fill, outline, text_color = "#F7DDDD", "#C76767", "#8A2B2B"
                else:
                    fill, outline, text_color = "#E8E8E8", "#A0A0A0", "#666666"
                c.create_rectangle(
                    x - half_w,
                    y - half_h,
                    x + half_w,
                    y + half_h,
                    fill=fill,
                    outline=outline,
                    width=1,
                )
                c.create_text(x, y, text=name, font=badge_font, fill=text_color)
                x_cursor += 2 * half_w + badge_gap

        draw_badge_row(row1, badge_y_top)
        draw_badge_row(row2, badge_y_top + badge_row_gap)

    def _state(self):
        def _get_bool(name, default=False):
            var = self._vars.get(name)
            if var is None:
                return bool(default)
            try:
                return bool(var.get())
            except Exception:
                return bool(default)

        return {
            "enable_m1_diffusion": _get_bool("enable_m1_diffusion", False),
            "enable_m1_facilitated_diffusion": _get_bool("enable_m1_facilitated_diffusion", False),
            "enable_m1_porin_diffusion": _get_bool("enable_m1_porin_diffusion", False),
            "enable_m2_diffusion": _get_bool("enable_m2_diffusion", True),
            "enable_diffusion_mutation": _get_bool("enable_diffusion_mutation", False),
            "homogeneous_initial_diffusion_const": _get_bool("homogeneous_initial_diffusion_const", False),
            "homogeneous_population": _get_bool("homogeneous_population", False),
            "independent_traits": _get_bool("independent_traits", False),
            "enable_chemostat_flow": _get_bool("enable_chemostat_flow", False),
            "enable_initial_energy": _get_bool("enable_initial_energy", False),
            "enable_intermediate_costs": _get_bool("enable_intermediate_costs", False),
            "enable_acetate_addition": _get_bool("enable_acetate_addition", False),
        }

    def _on_var_change(self, *_args):
        self.redraw()

    def _on_resize(self, *_args):
        self.redraw()
