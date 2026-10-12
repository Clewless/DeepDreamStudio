"""Target-class guidance panel: search, filter, dropdown, and listbox UI.

Builds and drives the ImageNet target-class picker in the GUI's Guidance
section. Mixed into DeepDreamGUI; the owning window drives state setup via
_init_target_class_state and builds the panel via _build_target_class_panel.
"""

import tkinter as tk
from typing import Any

from .gui_widgets import (
    ACCENT,
    ACCENT_HOVER,
    BG_CARD,
    BG_DEEP,
    BG_PANEL,
    BORDER,
    ERROR_COL,
    SUCCESS,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MAIN,
    ModernButton,
    ModernEntry,
    ModernFont,
    ModernFrame,
    ModernLabel,
    ModernOptionMenu,
    ModernScrollbar,
    ModernSegmentedButton,
    Tooltip,
    resolve_font_family,
)
from .imagenet_categories import lookup_category, search_categories


class TargetClassPanelMixin:
    """Searchable target-class picker (badge, search, filters, listbox)."""

    _target_class_var: tk.StringVar
    _class_search_var: tk.StringVar
    _class_filter_name: str
    _class_selected_label: Any
    _class_dropdown: Any
    _class_listbox: Any
    _class_listbox_height: int
    _class_listbox_min_height: int
    _class_listbox_max_height: int
    _class_list_resize_state: tuple[int, int] | None

    def _init_target_class_state(self) -> None:
        """Create the target-class variables, traces, and resize state."""
        self._class_selected_label = None
        self._class_dropdown = None
        self._class_listbox = None
        self._target_class_var = tk.StringVar(value="")
        self._target_class_var.trace_add("write", self._on_target_class_var_changed)
        self._class_search_var = tk.StringVar(value="")
        self._class_search_var.trace_add("write", self._on_class_search_changed)
        self._class_filter_name = "Popular"
        self._class_listbox_height = 8
        self._class_listbox_min_height = 3
        self._class_listbox_max_height = 20
        self._class_list_resize_state = None

    def _build_target_class_panel(self, sb) -> None:
        # Target Class container
        self._class_frame = ModernFrame(sb, fg_color=BG_CARD)
        self._class_frame.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._class_frame.grid_columnconfigure(0, weight=1)

        # Active Target Class Badge / Status
        self._class_status_card = ModernFrame(self._class_frame, fg_color=BG_PANEL)
        self._class_status_card.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self._class_status_card.grid_columnconfigure(0, weight=1)

        self._class_selected_label = ModernLabel(
            self._class_status_card,
            text="No target selected (search or click below)",
            font=ModernFont(family="Segoe UI", size=10, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        )
        self._class_selected_label.grid(row=0, column=0, sticky="w", padx=8, pady=4)

        self._class_clear_btn = ModernButton(
            self._class_status_card,
            text="✕ Clear",
            width=50,
            height=22,
            fg_color=BG_CARD,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=9),
            command=self._clear_target_class,
        )
        self._class_clear_btn.grid(row=0, column=1, padx=6, pady=4)
        Tooltip(self._class_clear_btn, "Clear current target class.")

        # Search Box with Clear Button
        search_box_frame = ModernFrame(self._class_frame, fg_color="transparent")
        search_box_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        search_box_frame.grid_columnconfigure(0, weight=1)

        self._class_search_entry = ModernEntry(
            search_box_frame,
            textvariable=self._class_search_var,
            placeholder_text="Search category or index (e.g. dog, 130)...",
            fg_color=BG_PANEL,
            border_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10),
            height=28,
        )
        self._class_search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._class_search_entry.bind("<Return>", self._on_class_search_enter)
        Tooltip(
            self._class_search_entry,
            "Type any category name or numeric index (0-999) to filter the dropdown in real time.",
        )

        clear_search_btn = ModernButton(
            search_box_frame,
            text="✕",
            width=24,
            height=28,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=9, weight="bold"),
            command=self._clear_class_search,
        )
        clear_search_btn.grid(row=0, column=1)
        Tooltip(clear_search_btn, "Clear search text.")

        # Mouse-only Quick Category Filters
        self._class_filter_seg = ModernSegmentedButton(
            self._class_frame,
            values=["Popular", "Animals", "Food", "Vehicles", "All"],
            command=self._on_class_category_filter_change,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=9),
            height=24,
        )
        self._class_filter_seg.set("Popular")
        self._class_filter_seg.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
        Tooltip(
            self._class_filter_seg, "Click to filter categories by theme with only mouse input."
        )

        # Initial popular choices for dropdown
        initial_matches = search_categories("", category_filter="★ Popular")
        init_values = [m[2] for m in initial_matches] or ["#130: flamingo"]

        # Dropdown Menu
        self._class_dropdown = ModernOptionMenu(
            self._class_frame,
            values=init_values,
            command=self._on_class_dropdown_select,
            fg_color=BG_PANEL,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_MAIN,
            height=30,
        )
        self._class_dropdown.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 4))
        Tooltip(
            self._class_dropdown,
            "Dropdown menu: click with mouse to browse and select ImageNet categories.",
        )

        # Integrated Dropdown Listbox (scrollable with mouse wheel, 1-click select)
        # Height is 8 rows by default; the grip below resizes it (3-20 rows).
        list_container = ModernFrame(self._class_frame, fg_color=BG_PANEL)
        list_container.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 2))
        list_container.grid_columnconfigure(0, weight=1)

        self._class_listbox_height = 8
        self._class_listbox_min_height = 3
        self._class_listbox_max_height = 20
        self._class_list_resize_state = None
        self._class_listbox = tk.Listbox(
            list_container,
            bg=BG_PANEL,
            fg=TEXT_MAIN,
            selectbackground=ACCENT,
            selectforeground=BG_DEEP,
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=(resolve_font_family("Segoe UI"), 9),
            height=self._class_listbox_height,
        )
        self._class_listbox.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=4)

        lb_scroll = ModernScrollbar(list_container, command=self._class_listbox.yview, width=12)
        lb_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 4), pady=4)
        self._class_listbox.configure(yscrollcommand=lb_scroll.set)

        self._class_listbox.bind("<<ListboxSelect>>", self._on_class_listbox_select)
        self._class_listbox.bind("<ButtonRelease-1>", self._on_class_listbox_select)
        Tooltip(
            list_container,
            "Scrollable category box: click any item with mouse to select as target.",
        )

        # Drag-to-resize grip for the listbox above
        self._class_list_resize_grip = ModernLabel(
            self._class_frame,
            text="⋯  ⇕ drag to resize list  ⋯",
            font=ModernFont(family="Segoe UI", size=8),
            text_color=TEXT_DIM,
            fg_color=BG_PANEL,
            cursor="sb_v_double_arrow",
            anchor="center",
        )
        self._class_list_resize_grip.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 8))
        Tooltip(
            self._class_list_resize_grip,
            "Click and drag up/down to resize the category list to fit your needs.",
        )
        self._class_list_resize_grip.bind("<Button-1>", self._on_class_list_resize_start)
        self._class_list_resize_grip.bind("<B1-Motion>", self._on_class_list_resize_drag)
        self._class_list_resize_grip.bind("<ButtonRelease-1>", self._on_class_list_resize_end)

        # Populate initial listbox items
        for v in init_values:
            self._class_listbox.insert(tk.END, f"  {v}")

        self._class_frame.grid_remove()  # Hidden initially

    # Target Class management (searchable & mouse-selectable)

    def _on_target_class_var_changed(self, *args):
        if self._class_selected_label is None:
            return
        val = self._target_class_var.get().strip()
        if val:
            res = lookup_category(val)
            if res:
                idx, name, label = res
                self._class_selected_label.configure(
                    text=f"Target: {label}",
                    text_color=SUCCESS,
                )
                if self._class_dropdown is not None:
                    try:
                        cur_vals = self._class_dropdown.cget("values") or []
                        if label not in cur_vals:
                            cur_vals = [label] + [v for v in cur_vals if v != label]
                            self._class_dropdown.configure(values=cur_vals)
                        self._class_dropdown.set(label)
                    except (tk.TclError, AttributeError):
                        pass
            else:
                # An unresolvable value must not read as a valid target: the
                # CLI would reject it and the job would fail after starting.
                self._class_selected_label.configure(
                    text=f"Target: {val} (unrecognized \u2014 pick from the list)",
                    text_color=ERROR_COL,
                )
        else:
            self._class_selected_label.configure(
                text="No target selected (search or click below)",
                text_color=TEXT_LABEL,
            )

    def _on_class_search_changed(self, *args):
        self._update_class_list_display()

    def _update_class_list_display(self):
        if self._class_dropdown is None or self._class_listbox is None:
            return
        query = self._class_search_var.get().strip()
        filter_name = self._class_filter_name

        cat_filter = (
            None if (filter_name in ("Popular", "★ Popular", "All") and query) else filter_name
        )
        matches = search_categories(query, category_filter=cat_filter)
        if not matches and cat_filter is not None:
            matches = search_categories(query, category_filter=None)

        if matches:
            vals = [m[2] for m in matches]
        else:
            vals = ["No matching categories"]

        self._class_dropdown.configure(values=vals)
        if matches:
            self._class_dropdown.set(vals[0])
        else:
            self._class_dropdown.set("No matching categories")

        self._class_listbox.delete(0, tk.END)
        for v in vals:
            self._class_listbox.insert(tk.END, f"  {v}")

    def _on_class_category_filter_change(self, val):
        self._class_filter_name = val
        if self._class_search_var.get():
            self._class_search_var.set("")
        else:
            self._update_class_list_display()

    def _on_class_dropdown_select(self, choice):
        if not choice or choice == "No matching categories":
            return
        self._target_class_var.set(choice)
        if self._class_listbox is not None:
            self._class_listbox.selection_clear(0, tk.END)
            for i in range(self._class_listbox.size()):
                item_text = self._class_listbox.get(i).strip()
                if item_text == choice.strip():
                    self._class_listbox.selection_set(i)
                    self._class_listbox.see(i)
                    break

    def _on_class_listbox_select(self, event=None):
        if self._class_listbox is None:
            return
        sel = self._class_listbox.curselection()
        if not sel:
            return
        chosen = self._class_listbox.get(sel[0]).strip()
        if not chosen or chosen == "No matching categories":
            return
        self._target_class_var.set(chosen)
        if self._class_dropdown is not None:
            self._class_dropdown.set(chosen)

    def _class_list_row_pixels(self):
        """Approximate pixel height of one listbox row for drag-resizing."""
        try:
            import tkinter.font as _tkfont

            f = _tkfont.Font(font=self._class_listbox.cget("font"))
            return max(12, int(f.metrics("linespace")) + 2)
        except (tk.TclError, RuntimeError, TypeError):
            return 16

    def _on_class_list_resize_start(self, event=None):
        try:
            h = int(self._class_listbox.cget("height"))
        except (ValueError, tk.TclError):
            h = self._class_listbox_height
        y = getattr(event, "y_root", 0) or 0
        self._class_list_resize_state = (y, h)
        return "break"

    def _on_class_list_resize_drag(self, event=None):
        state = self._class_list_resize_state
        if state is None or event is None:
            return "break"
        start_y, start_h = state
        dy = getattr(event, "y_root", start_y) - start_y
        row_px = max(1, self._class_list_row_pixels())
        min_h = int(self._class_listbox_min_height)
        max_h = int(self._class_listbox_max_height)
        new_h = min(max_h, max(min_h, start_h + int(round(dy / float(row_px)))))
        try:
            if int(self._class_listbox.cget("height")) != new_h:
                self._class_listbox.configure(height=new_h)
                self._class_listbox_height = new_h
        except (ValueError, tk.TclError):
            pass
        return "break"

    def _on_class_list_resize_end(self, event=None):
        self._class_list_resize_state = None
        return "break"

    def _clear_target_class(self):
        self._target_class_var.set("")
        if self._class_listbox is not None:
            self._class_listbox.selection_clear(0, tk.END)
        if self._class_dropdown is not None:
            self._class_dropdown.set("")

    def _clear_class_search(self):
        self._class_search_var.set("")

    def _on_class_search_enter(self, event=None):
        if self._class_listbox is None or self._class_listbox.size() == 0:
            return
        first = self._class_listbox.get(0).strip()
        if first and first != "No matching categories":
            self._target_class_var.set(first)
            if self._class_dropdown is not None:
                self._class_dropdown.set(first)
