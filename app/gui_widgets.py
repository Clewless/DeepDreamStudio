"""Custom dark-themed Tkinter / ttk widgets for DeepDream Studio."""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Any

from PIL import Image, ImageTk

from .gui_helpers import _fmt_slider_value, slider_t_to_val, slider_val_to_t

ACCENT = "#D0D0D0"  # light grey accent
ACCENT_HOVER = "#FFFFFF"
BG_DEEP = "#0A0A0A"  # near-black window background
BG_PANEL = "#111111"  # slightly lighter panel
BG_CARD = "#181818"  # card / section background
BORDER = "#2C2C2C"
TEXT_MAIN = "#EFEFEF"
TEXT_DIM = "#666666"
TEXT_LABEL = "#999999"
SUCCESS = "#6FBF73"  # green: done
WARNING = "#D9A441"  # amber: in progress / stopped
ERROR_COL = "#D96262"  # red: failed

# Cap on console-log lines kept in the GUI Text widget: long video runs emit
# one line per frame/iteration, and an unbounded widget grows process memory.
MAX_LOG_LINES = 4000

# Modern Tk theme kit (stdlib only): flat Tk widgets plus styled ttk controls.

_ttk_style_ready = False
_ttk_style_root = None


def _resolve_bg(parent, color, default=None):
    if color is None or (isinstance(color, str) and color.lower() == "transparent"):
        try:
            if parent is not None:
                return parent.cget("bg")
        except tk.TclError:
            pass
        return default if default is not None else BG_DEEP
    return color


def _px_to_chars(px, default=None):
    try:
        v = int(px)
    except (TypeError, ValueError):
        return default
    return max(1, int(round(v / 7.0)))


def _px_to_lines(px, default=1):
    try:
        v = int(px)
    except (TypeError, ValueError):
        return default
    if v >= 60:
        return max(2, int(round(v / 22.0)))
    return 1


def _filter_kw(kw: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Filter kwargs down to *allowed*, silently dropping unsupported options."""
    return {k: v for k, v in kw.items() if k in allowed}


def _tk_dim(value, converter, default):
    """Convert a pixel dimension to Tk text units; 0 stays 0 (auto-size, needed
    when an image is shown, since a text-unit width becomes pixels and would
    crush it). An inconvertible value falls back to *default* (None = drop)."""
    try:
        if int(value) == 0:
            return 0
    except (TypeError, ValueError):
        return default
    return converter(value, default)


def _detach_var_trace(var, trace_id):
    """Remove a Tk variable write trace; always returns None so callers clear state."""
    if var is not None and trace_id:
        try:
            var.trace_remove("write", trace_id)
        except tk.TclError:
            pass
    return None


# Preferred font families, best first per role. "Segoe UI" and "Consolas" ship
# with Windows only; on macOS and Linux Tk silently substitutes a default face,
# which wrecks the layout because widget widths are computed from font metrics.
# So resolve the request against the families the running Tk actually provides.
_UI_FAMILIES = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "DejaVu Sans", "Arial")
_MONO_FAMILIES = (
    "Consolas",
    "SF Mono",
    "Menlo",
    "Ubuntu Mono",
    "DejaVu Sans Mono",
    "Courier New",
)
_FAMILY_CANDIDATES = {"Segoe UI": _UI_FAMILIES, "Consolas": _MONO_FAMILIES}
_resolved_families: dict[str, str] = {}


def resolve_font_family(family):
    """Return *family* if this machine has it, else the best available substitute.

    Returns the request unchanged when Tk cannot be queried (no root yet), so
    this is always safe to call.
    """
    candidates = _FAMILY_CANDIDATES.get(family)
    if not candidates:
        return family
    cached = _resolved_families.get(family)
    if cached is not None:
        return cached
    try:
        available = set(tkfont.families())
    except (tk.TclError, AttributeError, RuntimeError):
        # No usable Tk interpreter yet: do not cache, the answer can change.
        return family
    if not available:
        return family
    resolved = next((name for name in candidates if name in available), family)
    _resolved_families[family] = resolved
    return resolved


def tk_option_font(family="Segoe UI", size=11):
    """Font description string for the Tk option database (option_add).

    The value is parsed by Tcl as a LIST, so a multi-word family such as
    "DejaVu Sans" must be braced: "{DejaVu Sans} 11". An unbraced
    "DejaVu Sans 11" splits into [DejaVu, Sans, 11], and Tcl reads "Sans"
    as the integer size ("expected integer but got ..."), which breaks
    every widget created on that interpreter.
    """
    return f"{{{resolve_font_family(family)}}} {size}"


class ModernFont(tkfont.Font):
    """tkinter.font.Font subclass with a simplified constructor."""

    def __init__(
        self,
        family="Segoe UI",
        size=11,
        weight="normal",
        slant="roman",
        underline=False,
        overstrike=False,
        **kw,
    ):
        w = "bold" if str(weight).lower() in ("bold", "700") else "normal"
        try:
            super().__init__(
                family=resolve_font_family(family),
                size=size,
                weight=w,  # type: ignore[arg-type]
                slant=slant,
                underline=underline,
                overstrike=overstrike,
            )
        except tk.TclError:
            super().__init__()


def _ensure_ttk_style(root=None):
    # ttk styles live on the Tcl interpreter, so track which root they were
    # configured for: a fresh Tk root (e.g. a second window in one process)
    # needs them applied again, while repeated calls on the same root are
    # no-ops.
    global _ttk_style_ready, _ttk_style_root
    if _ttk_style_ready and _ttk_style_root is root:
        return
    try:
        style = ttk.Style(root)
    except tk.TclError:
        return
    try:
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except tk.TclError:
        pass
    # Combobox (dropdown)
    try:
        style.configure(
            "Modern.TCombobox",
            fieldbackground=BG_PANEL,
            background=BG_PANEL,
            foreground=TEXT_MAIN,
            arrowcolor=ACCENT,
            bordercolor=BORDER,
            darkcolor=BORDER,
            lightcolor=BORDER,
            relief="flat",
            padding=6,
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", BG_PANEL), ("disabled", BG_DEEP)],
            foreground=[("disabled", TEXT_DIM)],
            background=[("readonly", BG_PANEL)],
            arrowcolor=[("disabled", TEXT_DIM)],
        )
        style.configure(
            "Modern.TCombobox.Listbox",
            background=BG_CARD,
            foreground=TEXT_MAIN,
            selectbackground=BORDER,
            selectforeground=TEXT_MAIN,
        )
        try:
            root.option_add("*TCombobox*Listbox.background", BG_CARD)
            root.option_add("*TCombobox*Listbox.foreground", TEXT_MAIN)
            root.option_add("*TCombobox*Listbox.selectBackground", BORDER)
            root.option_add("*TCombobox*Listbox.selectForeground", TEXT_MAIN)
            root.option_add("*TCombobox*Listbox.font", tk_option_font())
        except tk.TclError:
            pass
    except tk.TclError:
        pass
    # Scale (slider): chunky enough for precise dragging
    try:
        style.configure(
            "Modern.Horizontal.TScale",
            background=BG_CARD,
            troughcolor=BG_PANEL,
            bordercolor=BG_CARD,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            sliderthickness=20,
            troughrelief="flat",
            borderwidth=0,
        )
    except tk.TclError:
        pass
    # Progressbar
    try:
        style.configure(
            "Modern.Horizontal.TProgressbar",
            background=ACCENT,
            troughcolor=BG_CARD,
            bordercolor=BG_CARD,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=6,
        )
    except tk.TclError:
        pass
    # Scrollbars
    try:
        style.configure(
            "Modern.Vertical.TScrollbar",
            background=BG_CARD,
            troughcolor=BG_PANEL,
            bordercolor=BG_PANEL,
            arrowcolor=TEXT_DIM,
            relief="flat",
        )
        style.configure(
            "Modern.Horizontal.TScrollbar",
            background=BG_CARD,
            troughcolor=BG_PANEL,
            bordercolor=BG_PANEL,
            arrowcolor=TEXT_DIM,
            relief="flat",
        )
    except tk.TclError:
        pass
    r = root or getattr(tk, "_default_root", None)
    if r is not None:
        try:
            r.unbind_class("TCombobox", "<MouseWheel>")
            r.unbind_class("TCombobox", "<Button-4>")
            r.unbind_class("TCombobox", "<Button-5>")
            r.unbind_class("TSpinbox", "<MouseWheel>")
            r.unbind_class("TSpinbox", "<Button-4>")
            r.unbind_class("TSpinbox", "<Button-5>")
        except tk.TclError:
            pass
    _ttk_style_ready = True
    _ttk_style_root = root


class ModernFrame(tk.Frame):
    def __init__(
        self,
        parent,
        fg_color=None,
        bg_color=None,
        corner_radius=None,
        border_width=None,
        border_color=None,
        width=None,
        height=None,
        **kw,
    ):
        bg = bg_color if bg_color is not None else fg_color
        bg = _resolve_bg(parent, bg, BG_DEEP)
        super().__init__(
            parent,
            bg=bg,
            bd=0,
            highlightthickness=0,
            **_filter_kw(kw, {"padx", "pady", "cursor", "takefocus"}),
        )
        try:
            bw = int(border_width) if border_width else 0
        except (TypeError, ValueError):
            bw = 0
        if bw > 0:
            try:
                self.configure(highlightbackground=border_color or BORDER, highlightthickness=bw)
            except tk.TclError:
                pass


class ModernLabel(tk.Label):
    image: Any
    drop_target_register: Any
    dnd_bind: Any

    def __init__(
        self,
        parent,
        text="",
        font=None,
        text_color=None,
        fg_color=None,
        bg_color=None,
        corner_radius=None,
        anchor=None,
        wraplength=None,
        justify=None,
        width=None,
        height=None,
        image=None,
        compound=None,
        cursor=None,
        textvariable=None,
        **kw,
    ):
        bg = bg_color if bg_color is not None else fg_color
        bg = _resolve_bg(parent, bg, None)
        if bg is None:
            try:
                bg = parent.cget("bg")
            except tk.TclError:
                bg = BG_DEEP
        fg = text_color if text_color is not None else TEXT_MAIN
        tkw = {}
        if anchor is not None:
            tkw["anchor"] = anchor
        if wraplength is not None:
            tkw["wraplength"] = wraplength
        if justify is not None:
            tkw["justify"] = justify
        if width is not None:
            c = _tk_dim(width, _px_to_chars, None)
            if c is not None:
                tkw["width"] = c
        if height is not None:
            tkw["height"] = _tk_dim(height, _px_to_lines, 1)
        if cursor is not None:
            tkw["cursor"] = cursor
        if compound is not None:
            tkw["compound"] = compound
        if image is not None:
            tkw["image"] = image
        if textvariable is not None:
            tkw["textvariable"] = textvariable
        super().__init__(
            parent, text=text, font=font, fg=fg, bg=bg, bd=0, highlightthickness=0, **tkw
        )

    def configure(self, *args, **kw):
        if "text_color" in kw:
            kw["fg"] = kw.pop("text_color")
        if "fg_color" in kw:
            v = kw.pop("fg_color")
            try:
                kw["bg"] = _resolve_bg(self.master, v, self.cget("bg"))
            except tk.TclError:
                kw["bg"] = v
        if "bg_color" in kw:
            kw["bg"] = kw.pop("bg_color")
        kw.pop("corner_radius", None)
        kw.pop("hover_color", None)
        if "width" in kw and kw["width"] is not None:
            kw["width"] = _tk_dim(kw["width"], _px_to_chars, kw["width"])
        if "height" in kw and kw["height"] is not None:
            kw["height"] = _tk_dim(kw["height"], _px_to_lines, kw["height"])
        return super().configure(*args, **kw)

    config = configure

    def cget(self, key):
        if key == "text_color":
            return super().cget("fg")
        if key in ("fg_color", "bg_color"):
            return super().cget("bg")
        return super().cget(key)


class ModernButton(tk.Button):
    def __init__(
        self,
        parent,
        text="",
        font=None,
        fg_color=None,
        hover_color=None,
        text_color=None,
        corner_radius=None,
        border_width=None,
        border_color=None,
        width=None,
        height=None,
        state=None,
        command=None,
        cursor=None,
        image=None,
        compound=None,
        anchor=None,
        **kw,
    ):
        bg = _resolve_bg(parent, fg_color, BG_CARD)
        fg = text_color if text_color is not None else TEXT_MAIN
        hover = hover_color if hover_color is not None else BORDER
        # subtle 1px outline for definition on dark bg
        hl_bg = BORDER
        try:
            bw = int(border_width) if border_width else 0
            if bw > 0:
                hl_bg = border_color or BORDER
            else:
                # faint outline unless explicitly flat-accent button
                hl_bg = BORDER
        except (TypeError, ValueError):
            pass
        tkw = {
            "relief": "flat",
            "bd": 0,
            "highlightthickness": 1,
            "highlightbackground": hl_bg,
            "activebackground": hover,
            "activeforeground": fg,
            "cursor": cursor or "hand2",
            "padx": 8,
            "pady": 4,
        }
        if width is not None:
            c = _tk_dim(width, _px_to_chars, None)
            if c is not None:
                tkw["width"] = c
        if height is not None:
            h = _tk_dim(height, _px_to_lines, None)
            if h is not None:
                tkw["height"] = h
        if anchor is not None:
            tkw["anchor"] = anchor
        if compound is not None:
            tkw["compound"] = compound
        if image is not None:
            tkw["image"] = image
        super().__init__(parent, text=text, font=font, bg=bg, fg=fg, command=command, **tkw)  # type: ignore[arg-type]
        self._hover = hover
        self._normal_bg = bg
        if state is not None:
            try:
                self.configure(state=state)
            except tk.TclError:
                pass
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, _e=None):
        try:
            if str(self.cget("state")) != "disabled":
                self.configure(bg=self._hover)
        except tk.TclError:
            pass

    def _on_leave(self, _e=None):
        try:
            self.configure(bg=self._normal_bg)
        except tk.TclError:
            pass

    def configure(self, *args, **kw):
        if "text_color" in kw:
            kw["fg"] = kw.pop("text_color")
        if "fg_color" in kw:
            v = kw.pop("fg_color")
            try:
                bg = _resolve_bg(self.master, v, self._normal_bg)
            except tk.TclError:
                bg = v
            kw["bg"] = bg
            self._normal_bg = bg
        if "hover_color" in kw:
            self._hover = kw.pop("hover_color")
            kw["activebackground"] = self._hover
        kw.pop("corner_radius", None)
        kw.pop("border_color", None)
        kw.pop("border_width", None)
        if "width" in kw and kw["width"] is not None:
            kw["width"] = _tk_dim(kw["width"], _px_to_chars, kw["width"])
        if "height" in kw and kw["height"] is not None:
            h = _tk_dim(kw.pop("height"), _px_to_lines, None)
            if h is not None:
                kw["height"] = h
        else:
            kw.pop("height", None)
        return super().configure(*args, **kw)

    config = configure

    def cget(self, key):
        if key == "text_color":
            return super().cget("fg")
        if key == "fg_color":
            return super().cget("bg")
        if key == "hover_color":
            return getattr(self, "_hover", "")
        return super().cget(key)


class ModernEntry(tk.Entry):
    def __init__(
        self,
        parent,
        textvariable=None,
        placeholder_text="",
        fg_color=None,
        border_color=None,
        text_color=None,
        font=None,
        width=None,
        height=None,
        state=None,
        **kw,
    ):
        bg = _resolve_bg(parent, fg_color, BG_PANEL)
        fg = text_color if text_color is not None else TEXT_MAIN
        super().__init__(
            parent,
            textvariable=None,  # type: ignore[arg-type]
            font=font,
            bg=bg,
            fg=fg,
            insertbackground=fg,
            selectbackground=BORDER,
            selectforeground=TEXT_MAIN,
            relief="flat",
            highlightthickness=1,
            highlightbackground=border_color or BORDER,
            highlightcolor=ACCENT,
            bd=0,
            **_filter_kw(kw, {"show", "justify", "takefocus"}),
        )
        self._placeholder = placeholder_text or ""
        self._var = textvariable
        self._has_placeholder = False
        self._updating_var = False
        self._trace_id = None

        if self._var is not None:
            try:
                self._trace_id = self._var.trace_add("write", self._on_var_changed)
            except tk.TclError:
                pass

        val = ""
        if self._var is not None:
            try:
                val = self._var.get()
            except tk.TclError:
                val = ""

        if val:
            # Configuring textvariable already displays the variable's value;
            # inserting it again would duplicate the text (and the variable).
            super().configure(textvariable=self._var)
        elif self._placeholder:
            self._show_placeholder()
        elif self._var is not None:
            super().configure(textvariable=self._var)

        self.bind("<FocusIn>", self._clear_placeholder, add="+")
        self.bind("<FocusOut>", self._restore_placeholder, add="+")
        if state is not None:
            try:
                self.configure(state=state)
            except tk.TclError:
                pass

    def _show_placeholder(self):
        try:
            self._updating_var = True
            self._has_placeholder = True
            super().configure(textvariable="")  # type: ignore[call-overload]
            super().delete(0, "end")
            super().insert(0, self._placeholder)
            super().configure(fg=TEXT_DIM)
        except tk.TclError:
            pass
        finally:
            self._updating_var = False

    def _clear_placeholder(self, _e=None):
        if self._has_placeholder:
            try:
                self._updating_var = True
                self._has_placeholder = False
                super().delete(0, "end")
                super().configure(fg=TEXT_MAIN)
                if self._var is not None:
                    # Re-linking the variable restores its current value; no
                    # explicit insert, which would duplicate that value.
                    super().configure(textvariable=self._var)
            except tk.TclError:
                pass
            finally:
                self._updating_var = False

    def _restore_placeholder(self, _e=None):
        try:
            cur = self._var.get() if self._var is not None else super().get()
        except tk.TclError:
            cur = ""
        if not cur and self._placeholder:
            self._show_placeholder()

    def _on_var_changed(self, *args):
        if getattr(self, "_updating_var", False):
            return
        try:
            val = self._var.get() if self._var is not None else ""
            if val:
                if self._has_placeholder:
                    self._clear_placeholder()
                elif self.cget("textvariable") != str(self._var):
                    super().configure(textvariable=self._var)
            else:
                try:
                    is_focused = self == self.focus_get()
                except tk.TclError:
                    is_focused = False
                if self._placeholder and not is_focused:
                    self._show_placeholder()
        except (tk.TclError, RuntimeError):
            pass

    def get(self):
        if self._has_placeholder:
            return ""
        return super().get()

    def insert(self, index, string):
        if self._has_placeholder:
            self._clear_placeholder()
        return super().insert(index, string)

    def delete(self, first, last=None):
        if self._has_placeholder:
            self._clear_placeholder()
            return
        return super().delete(first, last)

    def destroy(self):
        self._trace_id = _detach_var_trace(self._var, self._trace_id)
        super().destroy()

    def configure(self, *args, **kw):
        if "text_color" in kw:
            kw["fg"] = kw.pop("text_color")
        if "fg_color" in kw:
            kw["bg"] = _resolve_bg(self.master, kw.pop("fg_color"), BG_PANEL)
        if "textvariable" in kw:
            new_var = kw.pop("textvariable")
            self._trace_id = _detach_var_trace(self._var, self._trace_id)
            self._var = new_var
            if self._var is not None:
                try:
                    self._trace_id = self._var.trace_add("write", self._on_var_changed)
                except tk.TclError:
                    pass
            if not self._has_placeholder and self._var is not None:
                kw["textvariable"] = self._var
        kw.pop("border_color", None)
        kw.pop("corner_radius", None)
        if "placeholder_text" in kw:
            self._placeholder = kw.pop("placeholder_text") or ""
            if self._has_placeholder:
                self._show_placeholder()
        return super().configure(*args, **kw)

    config = configure

    def cget(self, key):
        if key == "text_color":
            return super().cget("fg")
        if key == "fg_color":
            return super().cget("bg")
        if key == "textvariable":
            return self._var if self._var is not None else super().cget("textvariable")
        return super().cget(key)


class ModernCheckBox(tk.Checkbutton):
    def __init__(
        self,
        parent,
        text="",
        variable=None,
        font=None,
        text_color=None,
        fg_color=None,
        hover_color=None,
        checkmark_color=None,
        corner_radius=None,
        state=None,
        command=None,
        **kw,
    ):
        bg = _resolve_bg(parent, None, None)
        if bg is None or bg == "transparent":
            try:
                bg = parent.cget("bg")
            except tk.TclError:
                bg = BG_CARD
        fg = text_color if text_color is not None else TEXT_LABEL
        super().__init__(
            parent,
            text=text,
            variable=variable,
            font=font,
            bg=bg,
            fg=fg,
            selectcolor=BG_PANEL,
            activebackground=bg,
            activeforeground=TEXT_MAIN,
            relief="flat",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            anchor="w",
            command=command,
        )
        if state is not None:
            try:
                self.configure(state=state)
            except tk.TclError:
                pass

    def configure(self, *args, **kw):
        if "text_color" in kw:
            kw["fg"] = kw.pop("text_color")
        for k in (
            "fg_color",
            "hover_color",
            "checkmark_color",
            "corner_radius",
            "border_width",
            "border_color",
        ):
            kw.pop(k, None)
        return super().configure(*args, **kw)

    config = configure


class ModernSegmentedButton(tk.Frame):
    """Row of toggle buttons bound to a StringVar."""

    def __init__(
        self,
        parent,
        values=None,
        variable=None,
        command=None,
        selected_color=None,
        selected_hover_color=None,
        font=None,
        height=None,
        corner_radius=None,
        fg_color=None,
        **kw,
    ):
        bg = _resolve_bg(parent, fg_color, BG_PANEL)
        super().__init__(parent, bg=bg, bd=0, highlightthickness=1, highlightbackground=BORDER)
        self._values = list(values) if values else []
        self._var = variable
        self._current = self._values[0] if self._values else ""
        try:
            if self._var is not None and self._var.get():
                self._current = self._var.get()
        except tk.TclError:
            pass
        self._cmd = command
        self._font = font
        self._sel = selected_color or BORDER
        self._sel_hover = selected_hover_color or ACCENT
        self._buttons = {}
        self._trace_id = None
        self._build()
        if self._var is not None:
            try:
                self._trace_id = self._var.trace_add("write", lambda *a: self.refresh())
            except (tk.TclError, AttributeError):
                try:
                    self._trace_id = self._var.trace("w", lambda *a: self.refresh())
                except (tk.TclError, AttributeError):
                    self._trace_id = None
        self.refresh()

    def _build(self):
        for b in self._buttons.values():
            try:
                b.destroy()
            except tk.TclError:
                pass
        self._buttons = {}
        for i, v in enumerate(self._values):
            b = tk.Button(
                self,
                text=v,
                font=self._font,
                relief="flat",
                bd=0,
                highlightthickness=0,
                cursor="hand2",
                padx=8,
                pady=5,
                command=lambda val=v: self._choose(val),  # type: ignore[misc]
            )
            b.grid(row=0, column=i, sticky="nsew", padx=1, pady=1)
            self.grid_columnconfigure(i, weight=1)
            self._buttons[v] = b

    def _choose(self, val):
        self._current = val
        try:
            if self._var is not None:
                self._var.set(val)
        except tk.TclError:
            pass
        self.refresh()
        if self._cmd is not None:
            self._cmd(val)

    def refresh(self, selected=None):
        try:
            if selected is not None:
                cur = selected
            elif self._var is not None:
                try:
                    cur = self._var.get()
                    if cur:
                        self._current = cur
                    else:
                        cur = self._current
                except tk.TclError:
                    cur = self._current
            else:
                cur = self._current
        except tk.TclError:
            cur = getattr(self, "_current", None)
        for v, b in self._buttons.items():
            try:
                if v == cur:
                    b.configure(
                        bg=self._sel,
                        fg=TEXT_MAIN,
                        activebackground=self._sel_hover,
                        activeforeground=TEXT_MAIN,
                    )
                else:
                    b.configure(
                        bg=BG_PANEL,
                        fg=TEXT_DIM,
                        activebackground=BORDER,
                        activeforeground=TEXT_MAIN,
                    )
            except tk.TclError:
                pass

    def get(self):
        try:
            if self._var is not None:
                v = self._var.get()
                if v:
                    return v
        except tk.TclError:
            pass
        return getattr(self, "_current", "")

    def set(self, val):
        self._current = val
        try:
            if self._var is not None:
                self._var.set(val)
        except tk.TclError:
            pass
        self.refresh(selected=val)

    def configure(self, *args, **kw):
        if "values" in kw:
            self._values = list(kw.pop("values") or [])
            self._build()
            self.refresh()
        if "command" in kw:
            self._cmd = kw.pop("command")
        # state support: apply to every segment button
        state = kw.pop("state", None)
        if state is not None:
            for b in self._buttons.values():
                try:
                    b.configure(state=state)
                except tk.TclError:
                    pass
        kw.pop("selected_color", None)
        kw.pop("selected_hover_color", None)
        kw.pop("corner_radius", None)
        kw.pop("fg_color", None)
        if kw:
            try:
                return super().configure(*args, **kw)
            except tk.TclError:
                return None
        return None

    config = configure

    def cget(self, key):
        if key == "values":
            return list(self._values)
        try:
            return super().cget(key)
        except tk.TclError:
            return None

    def destroy(self):
        self._trace_id = _detach_var_trace(self._var, self._trace_id)
        super().destroy()


class ModernOptionMenu(ttk.Combobox):
    """Readonly dropdown that calls *command* on selection change."""

    def __init__(
        self,
        parent,
        variable=None,
        values=None,
        command=None,
        fg_color=None,
        button_color=None,
        button_hover_color=None,
        dropdown_fg_color=None,
        dropdown_hover_color=None,
        font=None,
        text_color=None,
        corner_radius=None,
        height=None,
        state="readonly",
        **kw,
    ):
        _ensure_ttk_style(parent.winfo_toplevel() if parent is not None else None)
        self._var = variable if variable is not None else tk.StringVar()
        self._cmd = command
        self._values = list(values) if values else []
        super().__init__(
            parent,
            textvariable=self._var,
            values=self._values,
            state=state,
            style="Modern.TCombobox",
            font=font or (resolve_font_family("Segoe UI"), 11),
        )
        self.bind("<<ComboboxSelected>>", self._on_select, add="+")
        # Disable scroll wheel selection cycling
        self.bind("<MouseWheel>", lambda _e: "break")
        self.bind("<Button-4>", lambda _e: "break")
        self.bind("<Button-5>", lambda _e: "break")
        # combobox height = visible dropdown rows
        try:
            self.configure(height=min(12, max(4, len(self._values))))
        except tk.TclError:
            pass

    def _on_select(self, _e=None):
        if self._cmd is not None:
            self._cmd(self.get())

    def get(self):
        try:
            return self._var.get()
        except tk.TclError:
            return super().get()

    def set(self, val):
        try:
            self._var.set(val)
        except tk.TclError:
            pass
        try:
            super().set(val)
        except tk.TclError:
            pass

    def configure(self, *args, **kw):
        if "values" in kw:
            self._values = list(kw.pop("values") or [])
            try:
                super().configure(values=self._values)
            except tk.TclError:
                pass
            try:
                super().configure(height=min(12, max(4, len(self._values))))
            except tk.TclError:
                pass
        if "command" in kw:
            self._cmd = kw.pop("command")
        if "variable" in kw:
            self._var = kw.pop("variable")
            try:
                super().configure(textvariable=self._var)
            except tk.TclError:
                pass
        if "state" in kw and kw["state"] == "normal":
            # ttk Combobox "normal" is editable; keep dropdown readonly.
            kw["state"] = "readonly"
        for k in (
            "fg_color",
            "button_color",
            "button_hover_color",
            "dropdown_fg_color",
            "dropdown_hover_color",
            "corner_radius",
            "text_color",
            "font",
        ):
            kw.pop(k, None)
        if kw or args:
            try:
                return super().configure(*args, **kw)
            except tk.TclError:
                return None
        return None

    config = configure

    def cget(self, key):
        if key == "values":
            try:
                return list(super().cget("values"))
            except tk.TclError:
                return list(self._values)
        return super().cget(key)


class ModernSlider(ttk.Scale):
    """Normalized 0.0-1.0 slider backed by ttk.Scale."""

    def __init__(
        self,
        parent,
        from_=0.0,
        to=1.0,
        command=None,
        number_of_steps=None,
        height=None,
        fg_color=None,
        progress_color=None,
        button_color=None,
        button_hover_color=None,
        corner_radius=None,
        **kw,
    ):
        _ensure_ttk_style(parent.winfo_toplevel() if parent is not None else None)
        self._cmd = command
        super().__init__(
            parent,
            from_=from_,
            to=to,
            orient="horizontal",
            style="Modern.Horizontal.TScale",
            command=self._wrapped,
            length=220,
        )
        try:
            self.configure(takefocus=True)
        except tk.TclError:
            pass

    def _wrapped(self, val):
        if self._cmd is None:
            return
        try:
            self._cmd(float(val))
        except (TypeError, ValueError):
            self._cmd(val)

    def set(self, val):
        try:
            super().set(float(val))
        except (tk.TclError, TypeError, ValueError):
            pass

    def get(self, x=None, y=None):
        try:
            return float(super().get())
        except (tk.TclError, TypeError, ValueError):
            return 0.0


class ModernProgressBar(ttk.Progressbar):
    def __init__(
        self,
        parent,
        fg_color=None,
        progress_color=None,
        corner_radius=None,
        height=None,
        mode="determinate",
        **kw,
    ):
        _ensure_ttk_style(parent.winfo_toplevel() if parent is not None else None)
        bar_mode = mode if mode in ("determinate", "indeterminate") else "determinate"
        super().__init__(
            parent,
            orient="horizontal",
            mode=bar_mode,  # type: ignore[arg-type]
            style="Modern.Horizontal.TProgressbar",
            maximum=100,
        )
        self._frac = 0.0

    def set(self, frac):
        try:
            f = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            f = 0.0
        self._frac = f
        try:
            self.configure(value=f * 100)
        except tk.TclError:
            pass

    def get(self):
        return self._frac


class ModernTextbox(tk.Text):
    def __init__(
        self,
        parent,
        fg_color=None,
        text_color=None,
        font=None,
        corner_radius=None,
        state="normal",
        wrap="word",
        height=None,
        **kw,
    ):
        bg = _resolve_bg(parent, fg_color, BG_PANEL)
        fg = text_color if text_color is not None else TEXT_MAIN
        try:
            h = max(4, min(14, int(int(height) / 12))) if height else 8
        except (TypeError, ValueError):
            h = 8
        super().__init__(
            parent,
            bg=bg,
            fg=fg,
            font=font,
            wrap=wrap,
            height=h,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            insertbackground=TEXT_MAIN,
            selectbackground=BORDER,
            selectforeground=TEXT_MAIN,
            bd=0,
            padx=8,
            pady=6,
        )
        if state is not None:
            try:
                self.configure(state=state)
            except tk.TclError:
                pass

    def configure(self, *args, **kw):
        if "text_color" in kw:
            kw["fg"] = kw.pop("text_color")
        if "fg_color" in kw:
            kw["bg"] = _resolve_bg(self.master, kw.pop("fg_color"), BG_PANEL)
        kw.pop("corner_radius", None)
        return super().configure(*args, **kw)

    config = configure


class ModernScrollableFrame(tk.Frame):
    """Scrollable container. Put content into .inner."""

    def __init__(
        self,
        parent,
        fg_color=None,
        corner_radius=None,
        scrollbar_button_color=None,
        scrollbar_button_hover_color=None,
        width=None,
        **kw,
    ):
        bg = _resolve_bg(parent, fg_color, BG_PANEL)
        super().__init__(parent, bg=bg, bd=0, highlightthickness=0)
        _ensure_ttk_style(parent.winfo_toplevel() if parent is not None else None)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview, style="Modern.Vertical.TScrollbar"
        )
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            add="+",
        )
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.itemconfig(self._win, width=e.width), add="+"
        )
        self.inner.bind("<Enter>", lambda _e: self._bind_wheel(), add="+")
        self.inner.bind("<Leave>", lambda _e: self._unbind_wheel(), add="+")
        self._wheel_bound = False

    def _on_wheel(self, e):
        try:
            raw = getattr(e, "delta", 0)
            if raw:
                if abs(raw) >= 120:
                    delta = -1 * int(raw / 120)
                else:
                    delta = -1 if raw > 0 else 1
            else:
                delta = 1 if getattr(e, "num", 4) == 5 else -1
            self.canvas.yview_scroll(delta, "units")
        except tk.TclError:
            pass
        return "break"

    def _bind_wheel(self):
        try:
            # Bind to the canvas and all children so scrolling works anywhere
            # inside the scrollable area without stealing events from other
            # scrollable frames via bind_all. Re-run on Enter so children added
            # while the pointer is inside (queue cards, layer chips) are covered.
            self.canvas.bind("<MouseWheel>", self._on_wheel)
            self.canvas.bind("<Button-4>", self._on_wheel)
            self.canvas.bind("<Button-5>", self._on_wheel)
            self._bind_wheel_recursive(self.inner)
            self._wheel_bound = True
        except tk.TclError:
            pass

    _SCROLLABLE_WIDGETS = (
        tk.Listbox,
        tk.Text,
        tk.Canvas,
        ttk.Scrollbar,
        tk.Scrollbar,
    )

    def _bind_wheel_recursive(self, widget):
        """Bind mouse wheel to widget and all descendants, skipping scrollable children."""
        if isinstance(widget, (*self._SCROLLABLE_WIDGETS, ModernScrollableFrame)):
            return
        try:
            widget.bind("<MouseWheel>", self._on_wheel)
            widget.bind("<Button-4>", self._on_wheel)
            widget.bind("<Button-5>", self._on_wheel)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._bind_wheel_recursive(child)

    def _unbind_wheel(self):
        if not self._wheel_bound:
            return
        try:
            # Enter/Leave fire when crossing into a child too (NotifyInferior),
            # so only unbind once the pointer is genuinely outside the scroll
            # area; otherwise hovering a child would disable wheel scrolling.
            if self._pointer_inside_inner():
                return
            self.canvas.unbind("<MouseWheel>")
            self.canvas.unbind("<Button-4>")
            self.canvas.unbind("<Button-5>")
            self._unbind_wheel_recursive(self.inner)
            self._wheel_bound = False
        except tk.TclError:
            pass

    def _pointer_inside_inner(self):
        try:
            x = self.winfo_pointerx() - self.inner.winfo_rootx()
            y = self.winfo_pointery() - self.inner.winfo_rooty()
            return 0 <= x < self.inner.winfo_width() and 0 <= y < self.inner.winfo_height()
        except tk.TclError:
            return False

    def _unbind_wheel_recursive(self, widget):
        """Unbind mouse wheel from widget and all descendants, skipping scrollable children."""
        if isinstance(widget, (*self._SCROLLABLE_WIDGETS, ModernScrollableFrame)):
            return
        try:
            if isinstance(widget, ModernOptionMenu):
                widget.bind("<MouseWheel>", lambda _e: "break")
                widget.bind("<Button-4>", lambda _e: "break")
                widget.bind("<Button-5>", lambda _e: "break")
            else:
                widget.unbind("<MouseWheel>")
                widget.unbind("<Button-4>")
                widget.unbind("<Button-5>")
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._unbind_wheel_recursive(child)


class ModernScrollbar(ttk.Scrollbar):
    def __init__(self, parent, **kw):
        _ensure_ttk_style(parent.winfo_toplevel() if parent is not None else None)
        orient = kw.pop("orientation", kw.pop("orient", "vertical"))
        command = kw.pop("command", None)
        super().__init__(
            parent,
            orient=orient,
            style=(
                "Modern.Vertical.TScrollbar"
                if orient == "vertical"
                else "Modern.Horizontal.TScrollbar"
            ),
            command=command,
        )


class ModernImage(ImageTk.PhotoImage):
    """PhotoImage wrapper that rasterizes a PIL image to a requested size.
    Sized once at construction: a Tk PhotoImage cannot be re-rasterized in
    place, so resizing builds a new ModernImage from the source."""

    def __init__(self, image=None, light_image=None, dark_image=None, size=None, **kw):
        pil = (
            image if image is not None else (dark_image if dark_image is not None else light_image)
        )
        self._size = tuple(size) if size else None
        if pil is not None and self._size:
            try:
                if pil.size != self._size:
                    pil = pil.copy()
                    pil.thumbnail((self._size[0] or 1, self._size[1] or 1))
                    pil = (
                        pil.resize(
                            (max(1, self._size[0]), max(1, self._size[1])),
                            getattr(Image, "Resampling", Image).LANCZOS,
                        )
                        if pil.size != self._size
                        else pil
                    )
            except (OSError, ValueError):
                pass
        elif pil is None and self._size:
            pil = Image.new("RGB", (max(1, self._size[0]), max(1, self._size[1])), (24, 24, 24))
        if self._size is None and pil is not None:
            try:
                self._size = tuple(pil.size)
            except (AttributeError, TypeError):
                self._size = (100, 100)
        super().__init__(pil)

    def cget(self, key):
        if key == "size":
            return tuple(self._size) if self._size else (0, 0)
        cget_fn = getattr(super(), "cget", None)
        if cget_fn is not None:
            try:
                return cget_fn(key)
            except (tk.TclError, AttributeError):
                return None
        return None


class Tooltip:
    """Always-on-top hint shown near a widget while the mouse hovers."""

    def __init__(self, widget, text, delay_ms=250):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip = None
        self._after_id = None
        try:
            widget.bind("<Enter>", self._schedule_show, add="+")
            widget.bind("<Leave>", self._cancel_and_hide, add="+")
            widget.bind("<ButtonPress>", self._cancel_and_hide, add="+")
        except tk.TclError:
            pass

    def _schedule_show(self, event=None):
        self._cancel_timer()
        if self.delay_ms > 0:
            try:
                self._after_id = self.widget.after(self.delay_ms, self._show)
            except tk.TclError:
                self._show()
        else:
            self._show()

    def _cancel_timer(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self.tip is not None:
            return
        try:
            if not self.widget.winfo_exists():
                return
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            try:
                self.tip.attributes("-topmost", True)
            except tk.TclError:
                pass

            container = tk.Frame(
                self.tip,
                bg=BORDER,
                bd=1,
            )
            container.pack(fill="both", expand=True)

            lbl = tk.Label(
                container,
                text=self.text,
                justify="left",
                bg=BG_PANEL,
                fg=TEXT_MAIN,
                font=(resolve_font_family("Segoe UI"), 9),
                padx=9,
                pady=6,
                wraplength=280,
            )
            lbl.pack(fill="both", expand=True)

            self.tip.update_idletasks()
            tip_w = self.tip.winfo_width()
            tip_h = self.tip.winfo_height()
            screen_w = self.widget.winfo_screenwidth()
            screen_h = self.widget.winfo_screenheight()

            # Flip above widget if overflowing bottom edge of screen
            if y + tip_h > screen_h - 12:
                y = max(10, self.widget.winfo_rooty() - tip_h - 6)
            # Shift left if overflowing right edge of screen
            if x + tip_w > screen_w - 12:
                x = max(10, screen_w - tip_w - 12)

            self.tip.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self._hide()

    def _hide(self, event=None):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None

    def _cancel_and_hide(self, event=None):
        self._cancel_timer()
        self._hide()


def _apply_modern_style(root):
    _ensure_ttk_style(root)
    try:
        root.unbind_class("TCombobox", "<MouseWheel>")
        root.unbind_class("TCombobox", "<Button-4>")
        root.unbind_class("TCombobox", "<Button-5>")
        root.unbind_class("TSpinbox", "<MouseWheel>")
        root.unbind_class("TSpinbox", "<Button-4>")
        root.unbind_class("TSpinbox", "<Button-5>")
    except tk.TclError:
        pass
    try:
        root.configure(bg=BG_DEEP)
    except tk.TclError:
        pass
    try:
        root.option_add("*Background", BG_DEEP)
        root.option_add("*Foreground", TEXT_MAIN)
        root.option_add("*Font", tk_option_font())
        root.option_add("*Entry.Background", BG_PANEL)
        root.option_add("*Entry.Foreground", TEXT_MAIN)
        root.option_add("*Text.Background", BG_PANEL)
        root.option_add("*Text.Foreground", TEXT_MAIN)
        root.option_add("*Listbox.background", BG_PANEL)
        root.option_add("*Listbox.foreground", TEXT_MAIN)
        root.option_add("*Listbox.selectBackground", BORDER)
        root.option_add("*Listbox.selectForeground", TEXT_MAIN)
    except tk.TclError:
        pass


class _SliderRow:
    """A parameter slider row with nudge controls and a value display."""

    def __init__(
        self, gui, parent, row, label, var, from_, to, is_int, tip="", curve=1.0, step=None
    ):
        self.gui = gui
        self.var = var
        self.is_int = is_int
        self.from_ = from_
        self.to = to

        if step is None:
            step = 1 if is_int else 0.01

        min_val = min(from_, to)
        max_val = max(from_, to)
        c = max(0.01, float(curve))

        self._step = step
        self._min_val = min_val
        self._max_val = max_val
        self._curve = c
        self._updating = False

        self.frame = ModernFrame(parent, fg_color="transparent")
        self.frame.grid(row=row, column=0, sticky="ew", padx=12, pady=(6, 6))
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)

        self.lbl = ModernLabel(
            self.frame,
            text=label,
            anchor="w",
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_LABEL,
        )
        self.lbl.grid(row=0, column=0, columnspan=2, sticky="w")
        if tip:
            Tooltip(self.lbl, tip)

        self.minus_btn = ModernButton(
            self.frame,
            text="−",
            width=28,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            command=lambda: self.nudge(-1),
        )
        self.minus_btn.grid(row=1, column=0, padx=(0, 6), sticky="w")

        self.val_label = ModernLabel(
            self.frame,
            anchor="e",
            font=ModernFont(family="Consolas", size=11),
            text_color=TEXT_MAIN,
            text=_fmt_slider_value(var.get(), is_int),
        )
        self.val_label.grid(row=0, column=2, sticky="e")

        self.sl = ModernSlider(
            self.frame,
            from_=0.0,
            to=1.0,
            fg_color=BG_PANEL,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self._on_slider_change,
        )
        self.sl.set(self.to_t(var.get()))
        self.sl.grid(row=1, column=1, sticky="ew", padx=0, pady=(2, 0))

        self.plus_btn = ModernButton(
            self.frame,
            text="+",
            width=28,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            command=lambda: self.nudge(1),
        )
        self.plus_btn.grid(row=1, column=2, padx=(6, 0), sticky="e")

        self._trace_id = None
        try:
            self._trace_id = var.trace_add("write", self._sync_from_var)
        except tk.TclError:
            self._trace_id = None

        for w in (self.frame, self.lbl, self.minus_btn, self.sl, self.plus_btn, self.val_label):
            w.bind("<Enter>", self._on_enter, add="+")
            w.bind("<Leave>", self._on_leave, add="+")

        self.sl.bind("<Left>", lambda e: self.nudge(-1) or "break")
        self.sl.bind("<Down>", lambda e: self.nudge(-1) or "break")
        self.sl.bind("<Right>", lambda e: self.nudge(1) or "break")
        self.sl.bind("<Up>", lambda e: self.nudge(1) or "break")

    def to_t(self, v):
        return slider_val_to_t(v, self.from_, self.to, self._curve)

    def to_val(self, t):
        return slider_t_to_val(
            t, self.from_, self.to, self._curve, is_int=self.is_int, step=self._step
        )

    def nudge(self, delta=1):
        if self.gui._proc and self.gui._proc.poll() is None:
            return
        try:
            cur = self.var.get()
        except tk.TclError:
            cur = self.from_
        new_v = cur + delta * self._step
        if self.is_int:
            new_v = int(round(new_v))
        else:
            new_v = round(new_v, 4)
        new_v = max(self._min_val, min(self._max_val, new_v))
        self.var.set(new_v)

    def refresh(self):
        """Sync slider position and value label from the current var value."""
        try:
            v = self.var.get()
            self.sl.set(self.to_t(v))
            self.val_label.configure(text=_fmt_slider_value(v, self.is_int))
        except tk.TclError:
            pass

    def dispose(self):
        """Detach the var trace so the row can't fire during/after teardown."""
        self._trace_id = _detach_var_trace(self.var, self._trace_id)

    def _on_slider_change(self, t):
        self._updating = True
        try:
            v = self.to_val(t)
            self.var.set(v)
            self.val_label.configure(text=_fmt_slider_value(v, self.is_int))
        finally:
            self._updating = False

    def _sync_from_var(self, *args):
        if self._updating:
            return
        try:
            v = self.var.get()
            self.sl.set(self.to_t(v))
            self.val_label.configure(text=_fmt_slider_value(v, self.is_int))
        except tk.TclError:
            pass

    def _on_enter(self, e=None):
        self.gui._active_hover_nudge = self.nudge

    def _on_leave(self, e=None):
        try:
            x = self.frame.winfo_pointerx() - self.frame.winfo_rootx()
            y = self.frame.winfo_pointery() - self.frame.winfo_rooty()
            if 0 <= x < self.frame.winfo_width() and 0 <= y < self.frame.winfo_height():
                return
        except tk.TclError:
            pass
        if self.gui._active_hover_nudge == self.nudge:
            self.gui._active_hover_nudge = None
