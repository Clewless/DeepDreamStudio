"""Dream-layer selection panel: single-layer menu and multi-layer chips.

Builds and drives the Dream Layer card in the GUI sidebar. Mixed into
DeepDreamGUI; the owning window drives state setup via _init_layer_state and
builds the panel via _build_sidebar_layers.
"""

import tkinter as tk
from collections.abc import Callable
from typing import Any

from .gui_widgets import (
    ACCENT,
    ACCENT_HOVER,
    BG_CARD,
    BG_DEEP,
    BG_PANEL,
    BORDER,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MAIN,
    ModernButton,
    ModernEntry,
    ModernFont,
    ModernFrame,
    ModernLabel,
    ModernOptionMenu,
    ModernSegmentedButton,
    Tooltip,
)
from .model_data import (
    DEFAULT_MODEL_KEY,
    MODEL_BY_LABEL,
    MODEL_META,
    layer_display_list,
    layer_value_map,
)

# Categorized layer depth tiers for visual layer selection
LAYER_TIERS = {
    "googlenet": [
        (
            "Shallow Textures",
            "Fine textures, lines & ripples",
            ["inception3a", "inception3b"],
        ),
        (
            "Shapes & Objects",
            "Eyes, faces, dogs & organic shapes",
            ["inception4a", "inception4b", "inception4c", "inception4d", "inception4e"],
        ),
        (
            "Deep Abstraction",
            "Large structures & heavy hallucination",
            ["inception5a", "inception5b"],
        ),
    ],
    "inceptionv3": [
        (
            "Shallow Textures",
            "Fine patterns & soft textures",
            ["Mixed_5b", "Mixed_5c", "Mixed_5d"],
        ),
        (
            "Shapes & Objects",
            "Objects and shapes forming",
            ["Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e"],
        ),
        (
            "Deep Abstraction",
            "Bold, complex objects & structures",
            ["Mixed_7a", "Mixed_7b", "Mixed_7c"],
        ),
    ],
    "vgg16": [
        (
            "Shallow Textures",
            "Fine textures & brush patterns",
            ["features.3", "features.8"],
        ),
        (
            "Shapes & Objects",
            "Bold shapes & emerging objects",
            ["features.15", "features.22"],
        ),
        ("Deep Abstraction", "Abstract forms & heavy hallucination", ["features.29"]),
    ],
    "vgg19": [
        (
            "Shallow Textures",
            "Micro-textures & ripples",
            ["features.3", "features.8"],
        ),
        (
            "Shapes & Objects",
            "Geometric patterns & objects forming",
            ["features.17", "features.26"],
        ),
        ("Deep Abstraction", "Deep structures & abstraction", ["features.35"]),
    ],
}

RECOMMENDED_MULTI_LAYERS = {
    "googlenet": ["inception4c", "inception5b"],
    "inceptionv3": ["Mixed_6d", "Mixed_7b"],
    "vgg16": ["features.15", "features.22"],
    "vgg19": ["features.26", "features.35"],
}


class LayerPanelMixin:
    """Single- and multi-layer selection for the Dream Layer card."""

    _section: Callable[..., Any]
    _proc: Any
    _model_var: tk.StringVar
    _layer_var: tk.StringVar
    _layer_mode_var: tk.StringVar
    _custom_layers_var: tk.StringVar
    _layer_menu: Any
    _custom_layers_entry: Any
    _layer_mode_seg: Any
    _layer_badge_count: Any
    _layer_chips_container: Any
    _single_layer_frame: Any
    _multi_layer_frame: Any

    def _init_layer_state(self) -> None:
        """Create the layer-selection variables, traces, and widget handles."""
        self._layer_menu = None
        self._custom_layers_entry = None
        self._layer_mode_seg = None
        self._layer_badge_count = None
        self._layer_chips_container = None
        self._single_layer_frame = None
        self._multi_layer_frame = None
        self._layer_mode_var = tk.StringVar(value="Single Layer")
        self._custom_layers_var = tk.StringVar(value="")
        self._custom_layers_var.trace_add("write", self._on_custom_layers_var_changed)

    def _build_sidebar_layers(self, sb):
        # Layer selector
        self._section(sb, row=12, label="Dream Layer")
        layer_card = ModernFrame(sb, fg_color=BG_CARD)
        layer_card.grid(row=13, column=0, sticky="ew", padx=16, pady=(0, 8))
        layer_card.grid_columnconfigure(0, weight=1)

        # Mode toggle: Single Layer vs Multi-Layer
        self._layer_mode_seg = ModernSegmentedButton(
            layer_card,
            values=["Single Layer", "Multi-Layer"],
            variable=self._layer_mode_var,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=11),
            command=self._on_layer_mode_change,
        )
        self._layer_mode_seg.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))
        Tooltip(
            self._layer_mode_seg,
            "Layer mode:\n"
            "• Single Layer: Focuses all dream iterations on one specific network depth.\n"
            "• Multi-Layer: Combines and balances features across multiple layers at once.",
        )

        # Single Layer container
        self._single_layer_frame = ModernFrame(layer_card, fg_color="transparent")
        self._single_layer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        self._single_layer_frame.grid_columnconfigure(0, weight=1)

        self._layer_menu = ModernOptionMenu(
            self._single_layer_frame,
            variable=self._layer_var,
            values=layer_display_list(DEFAULT_MODEL_KEY),
            command=self._on_single_layer_select,
            fg_color=BG_PANEL,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            font=ModernFont(family="Segoe UI", size=12),
            text_color=TEXT_MAIN,
            height=34,
        )
        self._layer_menu.grid(row=0, column=0, sticky="ew")
        Tooltip(
            self._layer_menu,
            "Which layer to optimize:\n"
            "• Early layers: Fine lines, ripples, textures.\n"
            "• Middle layers: Geometric forms, eyes, shapes.\n"
            "• Deep layers: Large high-level structures.",
        )

        # Multi-Layer container
        self._multi_layer_frame = ModernFrame(layer_card, fg_color="transparent")
        self._multi_layer_frame.grid_columnconfigure(0, weight=1)

        ml_header = ModernFrame(self._multi_layer_frame, fg_color="transparent")
        ml_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ml_header.grid_columnconfigure(0, weight=1)

        self._layer_badge_count = ModernLabel(
            ml_header,
            text="Active (0)",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_LABEL,
        )
        self._layer_badge_count.grid(row=0, column=0, sticky="w")

        ml_rec_btn = ModernButton(
            ml_header,
            text="★ Rec",
            width=50,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10, weight="bold"),
            command=self._select_recommended_layers,
        )
        ml_rec_btn.grid(row=0, column=1, padx=(0, 4))
        Tooltip(ml_rec_btn, "Activate recommended layer combination for this model.")

        ml_all_btn = ModernButton(
            ml_header,
            text="All",
            width=36,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._select_all_layers,
        )
        ml_all_btn.grid(row=0, column=2, padx=(0, 4))
        Tooltip(ml_all_btn, "Activate all available layers.")

        ml_clear_btn = ModernButton(
            ml_header,
            text="Clear",
            width=44,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._clear_multi_layers,
        )
        ml_clear_btn.grid(row=0, column=3)
        Tooltip(ml_clear_btn, "Deactivate all layers.")

        # Interactive chips container
        self._layer_chips_container = ModernFrame(
            self._multi_layer_frame,
            fg_color=BG_PANEL,
        )
        self._layer_chips_container.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        # Direct text entry for custom layer names.
        self._custom_layers_entry = ModernEntry(
            self._multi_layer_frame,
            textvariable=self._custom_layers_var,
            placeholder_text="e.g. inception4c,inception5b",
            fg_color=BG_PANEL,
            border_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10),
            height=26,
        )
        self._custom_layers_entry.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        Tooltip(
            self._custom_layers_entry,
            "Direct layer input: comma-separated list of layer names. Auto-syncs with badges above.",
        )

    # Multi-layer management (integrated in-window)

    def _on_single_layer_select(self, val):
        self._custom_layers_var.set("")
        self._layer_mode_var.set("Single Layer")
        self._update_layer_ui()

    def _on_layer_mode_change(self, mode=None):
        if mode is not None:
            self._layer_mode_var.set(mode)
        else:
            mode = self._layer_mode_var.get()
        if mode == "Multi-Layer":
            if not self._custom_layers_var.get().strip():
                # Seed multi-layer with currently selected single layer
                model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
                label_to_key = layer_value_map(model_key)
                cur_key = label_to_key.get(self._layer_var.get(), self._layer_var.get().split()[0])
                self._custom_layers_var.set(cur_key)
        else:
            self._custom_layers_var.set("")
        self._update_layer_ui()

    def _on_custom_layers_var_changed(self, *args):
        if self._layer_chips_container is not None:
            self._refresh_layer_chips()

    def _toggle_layer(self, key):
        if self._proc and self._proc.poll() is None:
            return
        raw = self._custom_layers_var.get().strip()
        layers = [layer_name.strip() for layer_name in raw.split(",") if layer_name.strip()]
        if key in layers:
            layers.remove(key)
        else:
            layers.append(key)
        self._custom_layers_var.set(",".join(layers))
        self._refresh_layer_chips()

    def _select_recommended_layers(self):
        if self._proc and self._proc.poll() is None:
            return
        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        rec = RECOMMENDED_MULTI_LAYERS.get(model_key, ["inception4c", "inception5b"])
        self._custom_layers_var.set(",".join(rec))
        self._refresh_layer_chips()

    def _select_all_layers(self):
        if self._proc and self._proc.poll() is None:
            return
        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        meta = MODEL_META.get(model_key, MODEL_META[DEFAULT_MODEL_KEY])
        all_keys = [k for k, _ in meta["layers"]]
        self._custom_layers_var.set(",".join(all_keys))
        self._refresh_layer_chips()

    def _remove_layer_chip(self, key):
        if self._proc and self._proc.poll() is None:
            return
        raw = self._custom_layers_var.get().strip()
        layers = [layer_name.strip() for layer_name in raw.split(",") if layer_name.strip()]
        if key in layers:
            layers.remove(key)
        self._custom_layers_var.set(",".join(layers))
        self._refresh_layer_chips()

    def _clear_multi_layers(self):
        if self._proc and self._proc.poll() is None:
            return
        self._custom_layers_var.set("")
        self._refresh_layer_chips()

    def _update_layer_ui(self):
        mode = self._layer_mode_var.get()
        if self._layer_mode_seg is not None:
            self._layer_mode_seg.set(mode)
        if mode == "Multi-Layer":
            if self._single_layer_frame is not None:
                self._single_layer_frame.grid_remove()
            if self._multi_layer_frame is not None:
                self._multi_layer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
            self._refresh_layer_chips()
        else:
            if self._multi_layer_frame is not None:
                self._multi_layer_frame.grid_remove()
            if self._single_layer_frame is not None:
                self._single_layer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

    def _refresh_layer_chips(self):
        if self._layer_chips_container is None:
            return
        for child in self._layer_chips_container.winfo_children():
            child.destroy()

        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        meta = MODEL_META.get(model_key, MODEL_META[DEFAULT_MODEL_KEY])
        layer_items = meta["layers"]
        layer_desc_map = dict(layer_items)
        tiers = LAYER_TIERS.get(model_key, [])

        raw = self._custom_layers_var.get().strip()
        active_layers = [layer_name.strip() for layer_name in raw.split(",") if layer_name.strip()]

        if self._layer_badge_count is not None:
            self._layer_badge_count.configure(text=f"Active ({len(active_layers)})")

        self._layer_chips_container.grid_columnconfigure(0, weight=1)
        self._layer_chips_container.grid_columnconfigure(1, weight=1)

        row_idx = 0
        used_keys = set()

        for tier_title, _tier_sub, tier_keys in tiers:
            matching = [k for k in tier_keys if k in layer_desc_map]
            if not matching:
                continue
            row_idx = self._add_layer_tier_label(tier_title, row_idx)
            used_keys.update(matching)
            row_idx = self._add_layer_chips(matching, active_layers, layer_desc_map, row_idx)

        # Any leftover keys not in tiers
        leftovers = [k for k, _ in layer_items if k not in used_keys]
        if leftovers:
            row_idx = self._add_layer_tier_label("Other Layers", row_idx)
            self._add_layer_chips(leftovers, active_layers, layer_desc_map, row_idx)

    def _add_layer_tier_label(self, title, row_idx):
        """Render a tier heading in the chips grid; returns the next free row."""
        lbl = ModernLabel(
            self._layer_chips_container,
            text=title,
            font=ModernFont(family="Segoe UI", size=9, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        )
        lbl.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))
        return row_idx + 1

    def _add_layer_chips(self, keys, active_layers, layer_desc_map, row_idx):
        """Render one group of layer chips in a two-column grid; returns the next row."""
        for j, key in enumerate(keys):
            is_active = key in active_layers
            raw_desc = layer_desc_map[key]
            clean_desc = raw_desc.split(" - ", 1)[1].strip() if " - " in raw_desc else raw_desc

            r = row_idx + (j // 2)
            c = j % 2

            btn = ModernButton(
                self._layer_chips_container,
                text=f"✓ {key}" if is_active else f"+ {key}",
                height=26,
                fg_color=ACCENT if is_active else BG_CARD,
                hover_color=ACCENT_HOVER if is_active else BORDER,
                text_color=BG_DEEP if is_active else TEXT_DIM,
                border_width=0 if is_active else 1,
                border_color=BORDER,
                font=ModernFont(
                    family="Segoe UI",
                    size=10,
                    weight="bold" if is_active else "normal",
                ),
                command=(
                    (lambda k=key: self._remove_layer_chip(k))
                    if is_active
                    else (lambda k=key: self._toggle_layer(k))
                ),
            )
            if len(keys) % 2 == 1 and j == len(keys) - 1 and c == 0:
                btn.grid(row=r, column=0, columnspan=2, padx=3, pady=2, sticky="ew")
            else:
                btn.grid(row=r, column=c, padx=3, pady=2, sticky="ew")

            Tooltip(
                btn,
                f"{key}: {clean_desc}\n"
                f"Status: {'Active (click to deactivate)' if is_active else 'Inactive (click to activate)'}",
            )
        return row_idx + (len(keys) + 1) // 2
