"""Widget-free coordinate mapping for zoom, focus selection, and slide reveal."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .gui_helpers import clamp01, clamp_zoom, compute_zoom_view


@dataclass(frozen=True)
class ViewerGeometry:
    """One rendered viewport snapshot shared by all pointer interactions.

    The viewport can be centered inside a larger label, and the image can be
    letterboxed inside the viewport. Both offsets must be included when mapping
    between label pixels and source pixels.
    """

    source_size: tuple[int, int]
    viewport_size: tuple[int, int]
    label_size: tuple[int, int]
    zoom: float
    center: tuple[float, float]

    @cached_property
    def view(self) -> dict[str, Any]:
        return compute_zoom_view(*self.source_size, *self.viewport_size, self.zoom, *self.center)

    def _viewport_point(self, x: float, y: float) -> tuple[float, float]:
        lw, lh = self.label_size
        vw, vh = self.viewport_size
        return float(x) - (lw - vw) // 2, float(y) - (lh - vh) // 2

    def source_point(self, x: float, y: float) -> tuple[float, float] | None:
        """Source pixels under a label point; None outside the visible image."""
        vx, vy = self._viewport_point(x, y)
        vw, vh = self.viewport_size
        if not (0 <= vx < vw and 0 <= vy < vh):
            return None
        left, top, right, bottom = self.view["crop"]
        dw, dh = self.view["size"]
        ox, oy = self.view["offset"]
        if not (ox <= vx < ox + dw and oy <= vy < oy + dh):
            return None
        return (
            left + (vx - ox) * (right - left) / max(1, dw),
            top + (vy - oy) * (bottom - top) / max(1, dh),
        )

    def normalized_point(self, x: float, y: float) -> tuple[float, float] | None:
        point = self.source_point(x, y)
        if point is None:
            return None
        sw, sh = self.source_size
        return clamp01(point[0] / sw), clamp01(point[1] / sh)

    def split_label_x(self, ratio: float) -> float | None:
        """Label position of the reveal divider, or None if it is outside the crop."""
        sw, _ = self.source_size
        left, _, right, _ = self.view["crop"]
        dw, _ = self.view["size"]
        ox, _ = self.view["offset"]
        source_x = float(ratio) * sw
        if source_x < left or source_x > right:
            return None
        lw, _ = self.label_size
        vw, _ = self.viewport_size
        return float((lw - vw) // 2 + ox + (source_x - left) * dw / max(1, right - left))

    def split_ratio(self, x: float) -> float:
        """Map a dragged divider to source fraction, clamping at image edges."""
        sw, _ = self.source_size
        left, _, right, _ = self.view["crop"]
        dw, _ = self.view["size"]
        ox, _ = self.view["offset"]
        if dw <= 0 or sw <= 0:
            return 0.5
        vx, _ = self._viewport_point(x, 0)
        clamped_x = max(ox, min(ox + dw, vx))
        source_x = left + (clamped_x - ox) * (right - left) / max(1, dw)
        return clamp01(source_x / sw)

    def zoom_target(
        self, factor: float, cursor: tuple[float, float] | None = None
    ) -> tuple[float, float, float]:
        """New zoom and center, keeping the source point under the cursor fixed."""
        zoom = clamp_zoom(self.zoom * factor)
        cx, cy = self.center
        if cursor is None:
            return zoom, cx, cy
        point = self.source_point(*cursor)
        if point is None:
            return zoom, cx, cy
        vx, vy = self._viewport_point(*cursor)
        # The new zoom can remove letterboxing. Anchor against the *new*
        # image rectangle, not the old one's offsets and displayed size.
        new_view = compute_zoom_view(*self.source_size, *self.viewport_size, zoom, *self.center)
        dw, dh = new_view["size"]
        ox, oy = new_view["offset"]
        # Zoom-out can introduce letterboxing: cursor inside the old image
        # may land outside the new one, where anchoring is impossible.
        # Keep the center instead of panning to the edge.
        if not (ox <= vx < ox + dw and oy <= vy < oy + dh):
            return zoom, cx, cy
        left, top, right, bottom = new_view["crop"]
        frac_x = (vx - ox) / max(1, dw) - 0.5
        frac_y = (vy - oy) / max(1, dh) - 0.5
        sw, sh = self.source_size
        cx = point[0] - frac_x * (right - left)
        cy = point[1] - frac_y * (bottom - top)
        return zoom, max(0.0, min(sw, cx)), max(0.0, min(sh, cy))
