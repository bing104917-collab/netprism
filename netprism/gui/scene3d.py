"""A dependency-free software-projected 3D graph view on QGraphicsView.

There is deliberately no OpenGL / WebEngine: every layer is a painted QGraphics
item.  The view stays usable with several hundred nodes by drawing a simplified
set of edges and suppressing per-node labels above a threshold.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QLabel, QToolTip

from .theme import (
    format_count,
    format_mb,
    format_shape,
    group_color,
    group_label,
    edge_color,
    esc,
    kind_color,
    kind_label,
    node_tooltip,
    shapes_text,
)

SCALE = 42.0
DEFAULT_YAW = -1.0
DEFAULT_PITCH = 0.48
GROUP_GAP = 2.6
#: Candidate presentation angles; export picks the one closest to 16:9.
EXPORT_ANGLES = (
    (-1.00, 0.48),
    (-0.86, 0.44),
    (-0.72, 0.42),
    (-0.58, 0.40),
    (-0.44, 0.38),
    (-0.30, 0.36),
)
EXPORT_ASPECT = 16.0 / 9.0

#: Drag sensitivity (radians per pixel).  A 400 px drag is a ~126° turntable
#: turn: the same order as desktop 3D viewers, but slow enough to aim.
YAW_PER_PIXEL = 0.0055
PITCH_PER_PIXEL = 0.0048
PITCH_LIMIT = 1.15
#: Live rotation repaint budget: at most one geometry refresh per frame.
FRAME_INTERVAL = 1.0 / 60.0
#: Wheel zoom factor applied per 120 units of angleDelta.
ZOOM_FACTOR_120 = 1.14
MIN_ZOOM = 0.10
MAX_ZOOM = 12.0
#: Pixels kept free around the projected model when auto-framing.
FIT_MARGIN_X = 26.0
FIT_MARGIN_Y = 32.0
#: Padding added to group boxes around their member layers, in world units.
GROUP_PAD_X = 0.52
GROUP_PAD_Y = 0.32
GROUP_PAD_Z = 0.32


def _poly(points: list[QPointF]) -> QPolygonF:
    return QPolygonF(points)


def _mix(color: QColor, factor: float) -> QColor:
    if factor >= 1:
        return color.lighter(int(factor * 100))
    return color.darker(int((1.0 / max(0.18, factor)) * 100))


class NodeItem(QGraphicsItem):
    """One layer rendered as a shaded 3D box."""

    def __init__(self, view: "Graph3DView", node: dict[str, Any], center: tuple[float, float, float],
                 dims: tuple[float, float, float], color: str) -> None:
        super().__init__()
        self.view = view
        self.node = node
        self.node_id = str(node.get("id") or "")
        self.center = center
        self.dims = dims
        self.color = QColor(color)
        self.selected = False
        self.highlighted = False
        self._faces: list[tuple[list[QPointF], float, float]] = []
        self._depth = 0.0
        self._bounds = QRectF()
        self._compute()
        self.setZValue(100 + self._depth)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(True)
        self.setToolTip("")

    def _compute(self) -> None:
        cx, cy, cz = self.center
        dx, dy, dz = self.dims
        corners = [
            (cx - dx / 2, cy - dy / 2, cz - dz / 2),
            (cx + dx / 2, cy - dy / 2, cz - dz / 2),
            (cx + dx / 2, cy + dy / 2, cz - dz / 2),
            (cx - dx / 2, cy + dy / 2, cz - dz / 2),
            (cx - dx / 2, cy - dy / 2, cz + dz / 2),
            (cx + dx / 2, cy - dy / 2, cz + dz / 2),
            (cx + dx / 2, cy + dy / 2, cz + dz / 2),
            (cx - dx / 2, cy + dy / 2, cz + dz / 2),
        ]
        projected = [self.view.project(point) for point in corners]
        points = [point for point, _depth in projected]
        depths = [depth for _point, depth in projected]
        self._depth = sum(depths) / max(1, len(depths))
        faces = [
            ([0, 1, 2, 3], 0.62),  # back z-
            ([4, 5, 6, 7], 1.02),  # front z+
            ([0, 1, 5, 4], 0.82),  # bottom y-
            ([3, 2, 6, 7], 1.18),  # top y+
            ([0, 3, 7, 4], 0.74),  # left x-
            ([1, 2, 6, 5], 0.92),  # right x+
        ]
        self._faces = []
        for indices, shade in faces:
            face_points = [points[index] for index in indices]
            mean_depth = sum(depths[index] for index in indices) / len(indices)
            self._faces.append((face_points, shade, mean_depth))
        self._faces.sort(key=lambda item: item[2], reverse=True)
        rect = QRectF(points[0], points[0])
        for point in points[1:]:
            rect = rect.united(QRectF(point, point))
        self._bounds = rect.adjusted(-2.0, -2.0, 2.0, 2.0)

    def boundingRect(self) -> QRectF:
        return self._bounds

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self._bounds)
        return path

    def refresh(self) -> None:
        """Re-project the box for the current camera without rebuilding the scene."""
        self.prepareGeometryChange()
        self._compute()
        self.setZValue(100 + self._depth)
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for face, shade, _depth in self._faces:
            color = _mix(self.color, shade)
            color.setAlpha(238 if shade > 0.8 else 220)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(4, 10, 20, 215), 0.8))
            painter.drawPolygon(_poly(face))
        if self.highlighted or self.selected:
            pen = QPen(QColor("#fff7d6") if self.selected else QColor("#fbbf24"), 2.0)
            pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            for face, _shade, _depth in self._faces:
                painter.drawPolygon(_poly(face))

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.view.node_hover_enter(self.node, event.screenPos())
        event.accept()

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.view.node_hover_leave()
        event.accept()


class GroupItem(QGraphicsItem):
    """Semi-transparent AABB around a group of layers."""

    def __init__(self, view: "Graph3DView", group: dict[str, Any], bounds: tuple[float, float, float, float, float, float]) -> None:
        super().__init__()
        self.view = view
        self.group = group
        self.bounds = bounds
        self.color = QColor(group_color(group.get("kind")))
        self._faces: list[list[QPointF]] = []
        self._rect = QRectF()
        self._compute()
        self.setZValue(-260)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def _compute(self) -> None:
        min_x, max_x, min_y, max_y, min_z, max_z = self.bounds
        corners = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        points = [self.view.project(point)[0] for point in corners]
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 2, 6, 7],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ]
        self._faces = [[points[index] for index in face] for face in faces]
        rect = QRectF(points[0], points[0])
        for point in points[1:]:
            rect = rect.united(QRectF(point, point))
        self._rect = rect.adjusted(-2, -2, 2, 2)

    def boundingRect(self) -> QRectF:
        return self._rect

    def refresh(self) -> None:
        """Re-project the group box for the current camera."""
        self.prepareGeometryChange()
        self._compute()
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fill = QColor(self.color)
        fill.setAlpha(14)
        top = QColor(self.color)
        top.setAlpha(28)
        pen = QPen(QColor(self.color.red(), self.color.green(), self.color.blue(), 115), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for index, face in enumerate(self._faces):
            painter.setBrush(top if index == 3 else fill)
            painter.drawPolygon(_poly(face))
        # 12 crisp wireframe edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        for a, b in edges:
            painter.drawLine(self._faces[0][a] if a in (0, 1, 2, 3) else self._faces[1][a - 4],
                             self._faces[0][b] if b in (0, 1, 2, 3) else self._faces[1][b - 4])


class PillLabelItem(QGraphicsItem):
    """Small dark pill label used for groups and selected layers."""

    def __init__(self, text: str, color: str, position: QPointF, *, font_size: float = 8.0) -> None:
        super().__init__()
        self.text = text
        self.font = QFont("Microsoft YaHei UI", int(font_size))
        self.font.setWeight(QFont.Weight.DemiBold)
        metrics = QFontMetricsF(self.font)
        self.width = min(190.0, metrics.horizontalAdvance(text) + 12.0)
        self.height = metrics.height() + 6.0
        self._rect = QRectF(position.x() - self.width / 2, position.y() - self.height, self.width, self.height)
        self.color = QColor(color)
        self.setZValue(800)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)

    def place(self, position: QPointF) -> None:
        """Move the pill so it sits just above ``position`` (scene pixels)."""
        self.prepareGeometryChange()
        self._rect = QRectF(
            position.x() - self.width / 2,
            position.y() - self.height,
            self.width,
            self.height,
        )
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        background = QColor(5, 11, 20, 230)
        border = QColor(self.color)
        border.setAlpha(120)
        painter.setBrush(background)
        painter.setPen(QPen(border, 0.8))
        painter.drawRoundedRect(self._rect, 3.0, 3.0)
        painter.setFont(self.font)
        painter.setPen(QColor("#c6e8f5"))
        painter.drawText(self._rect.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter, self.text)


class Graph3DView(QGraphicsView):
    """Read/write graph view with a software isometric projection."""

    nodeSelected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("view3D")
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        # The camera is recomputed explicitly in _apply_camera(), so Qt must not
        # re-anchor the transform behind our back.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setStyleSheet("QGraphicsView { background: #070b14; border: none; }")

        self.graph: dict[str, Any] = {}
        self.nodes: list[dict[str, Any]] = []
        self.node_by_id: dict[str, dict[str, Any]] = {}
        self.group_by_id: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.positions: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        self._node_items: dict[str, NodeItem] = {}
        self._group_items: list[GroupItem] = []
        self._edge_items: list[QGraphicsPathItem] = []
        self._edge_specs: list[tuple[QGraphicsPathItem, tuple[float, float, float],
                                      tuple[float, float, float], str]] = []
        self._label_items: list[QGraphicsItem] = []
        self._label_specs: list[tuple[PillLabelItem, tuple[float, float, float]]] = []
        self._selected_id: str | None = None
        self._hovered_id: str | None = None
        self._highlighted: set[str] = set()
        self._yaw = DEFAULT_YAW
        self._pitch = DEFAULT_PITCH
        self._zoom = 1.0
        self._fit_cap = 0.0
        self._world_bounds: tuple[float, float, float, float, float, float] | None = None
        self._press_pos: QPoint | None = None
        self._press_yaw = self._yaw
        self._press_pitch = self._pitch
        self._dragging = False
        self._last_frame = 0.0
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setSingleShot(True)
        self._rotation_timer.setInterval(max(1, int(FRAME_INTERVAL * 1000)))
        self._rotation_timer.timeout.connect(self._apply_pending_rotation)

        self.hint = QLabel("左键拖拽旋转   ·   滚轮缩放   ·   单击层选中   ·   双击复位", self)
        self.hint.setStyleSheet(
            "QLabel { color: #5f7b96; background: rgba(5,11,20,180); border: 1px solid #14263b;"
            " border-radius: 6px; padding: 5px 9px; font-size: 9px; }"
        )
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint.adjustSize()
        self.hint.move(12, 10)

        self.empty_label = QLabel("暂无模型", self)
        self.empty_label.setStyleSheet(
            "QLabel { color: #6d829b; font-size: 13px; background: transparent; }"
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.empty_label.show()

        self.stats = QLabel("", self)
        self.stats.setStyleSheet(
            "QLabel { color: #4e6984; background: rgba(5,11,20,160); border-radius: 5px;"
            " padding: 4px 8px; font-family: Consolas, monospace; font-size: 8px; }"
        )
        self.stats.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.stats.move(12, 44)

        self.caption = QLabel("", self)
        self.caption.setStyleSheet(
            "QLabel { color: #81a5bf; background: rgba(5,11,20,160); border-radius: 5px;"
            " padding: 4px 8px; font-family: Consolas, monospace; font-size: 8px; }"
        )
        self.caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.caption.hide()

    # ------------------------------------------------------------------ #
    # graph data
    # ------------------------------------------------------------------ #
    def set_graph(self, graph: dict[str, Any]) -> None:
        self.graph = graph if isinstance(graph, dict) else {}
        self.nodes = [node for node in (self.graph.get("nodes") or []) if isinstance(node, dict)]
        self.node_by_id = {str(node.get("id")): node for node in self.nodes if node.get("id") is not None}
        self.group_by_id = {
            str(group.get("id")): group
            for group in (self.graph.get("groups") or [])
            if isinstance(group, dict) and group.get("id") is not None
        }
        self.edges = [edge for edge in (self.graph.get("edges") or []) if isinstance(edge, dict)]
        self._selected_id = None
        self._hovered_id = None
        self._highlighted.clear()
        self._yaw = DEFAULT_YAW
        self._pitch = DEFAULT_PITCH
        self._zoom = 1.0
        self._compute_layout()
        self._rebuild(fit=True)
        self.empty_label.setVisible(not self.nodes)
        self.update()

    def _compute_layout(self) -> None:
        self.positions = {}
        self._world_bounds = None
        if not self.nodes:
            return
        raw_volumes: list[float] = []
        raw_spatial: list[float] = []
        dims: list[tuple[float, float, float]] = []
        for node in self.nodes:
            shape = _first_shape(node)
            volume, spatial, channels = _shape_metrics(shape)
            raw_volumes.append(math.log10(volume + 1))
            raw_spatial.append(math.log10(spatial + 1))
            dims.append((volume, spatial, channels))
        max_volume = max(raw_volumes) or 1.0
        max_spatial = max(raw_spatial) or 1.0
        max_channel_count = max((item[2] for item in dims), default=1.0) or 1.0
        centers: list[float] = []
        cursor = 0.0
        previous_group: str | None = None
        for index, node in enumerate(self.nodes):
            group_id = str(node.get("groupId") or "")
            if index:
                cursor += GROUP_GAP if group_id != previous_group and group_id else 2.05
            previous_group = group_id or None
            centers.append(cursor)
        # Rotate around the model's center, not its first layer.  Otherwise a
        # long model swings out of the viewport and feels uncontrollable while
        # dragging (the old pivot was the origin at the first node).
        pivot = (min(centers) + max(centers)) / 2 if centers else 0.0
        for index, node in enumerate(self.nodes):
            raw_v, raw_s, raw_c = dims[index]
            log_v = math.log10(raw_v + 1) / max_volume
            log_s = math.log10(raw_s + 1) / max_spatial
            # Channel depth is deliberately *linear*, not log-normalised:
            # 3 -> 4 -> 32 -> 64 must read as a real relative thickness.
            channel_ratio = max(0.0, min(1.0, float(raw_c) / max_channel_count))
            dx = 0.68 + 0.58 * log_v
            dy = 0.78 + 1.55 * log_s
            dz = 0.55 + 2.45 * channel_ratio
            self.positions[str(node.get("id"))] = (
                (centers[index] - pivot, 0.0, 0.0),
                (dx, dy, dz),
            )
        # Rotation-invariant world bounds (node boxes plus the padding used by
        # group boxes).  The camera framing and the scene rect are derived from
        # this once per layout, so turning the model never moves the picture.
        min_x = min_y = min_z = math.inf
        max_x = max_y = max_z = -math.inf
        for (cx, cy, cz), (dx, dy, dz) in self.positions.values():
            min_x = min(min_x, cx - dx / 2)
            max_x = max(max_x, cx + dx / 2)
            min_y = min(min_y, cy - dy / 2)
            max_y = max(max_y, cy + dy / 2)
            min_z = min(min_z, cz - dz / 2)
            max_z = max(max_z, cz + dz / 2)
        if not self.positions:
            self._world_bounds = None
            return
        self._world_bounds = (
            min_x - GROUP_PAD_X,
            max_x + GROUP_PAD_X,
            min_y - GROUP_PAD_Y,
            max_y + GROUP_PAD_Y,
            min_z - GROUP_PAD_Z,
            max_z + GROUP_PAD_Z,
        )

    def project(self, point: tuple[float, float, float]) -> tuple[QPointF, float]:
        x, y, z = point
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        y1 = y * cp - z1 * sp
        depth = y * sp + z1 * cp
        return QPointF(x1 * SCALE, -y1 * SCALE), depth

    # ------------------------------------------------------------------ #
    # scene rebuilding
    # ------------------------------------------------------------------ #
    def _rebuild(self, *, fit: bool = False) -> None:
        """Recreate every item (graph, selection or highlight changes)."""
        scene = self.scene()
        self._rotation_timer.stop()
        scene.clear()
        self._node_items.clear()
        self._group_items.clear()
        self._edge_items.clear()
        self._edge_specs.clear()
        self._label_items.clear()
        self._label_specs.clear()
        if not self.nodes:
            self._zoom = 1.0
            self._fit_cap = 0.0
            scene.setSceneRect(QRectF(-320, -240, 640, 480))
            self.stats.setText("0 layers")
            self.empty_label.setVisible(True)
            self.hint.setVisible(False)
            self._apply_camera(refit=True)
            self.viewport().update()
            return
        self.empty_label.setVisible(False)
        self.hint.setVisible(True)

        # group boxes before edges
        for group in self.group_by_id.values():
            group_nodes = [nid for nid in (group.get("nodeIds") or []) if nid in self.positions]
            if not group_nodes or group.get("kind") == "model":
                continue
            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for node_id in group_nodes:
                center, dims = self.positions[node_id]
                xs.append(center[0])
                ys.append(center[1] - dims[1] / 2)
                zs.append(center[2] - dims[2] / 2)
                ys.append(center[1] + dims[1] / 2)
                zs.append(center[2] + dims[2] / 2)
            bounds = (
                min(xs) - GROUP_PAD_X,
                max(xs) + GROUP_PAD_X,
                min(ys) - GROUP_PAD_Y,
                max(ys) + GROUP_PAD_Y,
                min(zs) - GROUP_PAD_Z,
                max(zs) + GROUP_PAD_Z,
            )
            item = GroupItem(self, group, bounds)
            scene.addItem(item)
            self._group_items.append(item)
            center_x = (bounds[0] + bounds[1]) / 2
            label = group.get("label") or group.get("name") or group.get("id")
            self._add_label(str(label), group_color(group.get("kind")),
                            (center_x, bounds[3] + 0.52, 0.0))

        for edge in self._visible_edges():
            source = self.positions.get(str(edge.get("source")))
            target = self.positions.get(str(edge.get("target")))
            if not source or not target:
                continue
            kind = str(edge.get("kind") or "data")
            path, z_value = self._edge_geometry(source[0], target[0], kind)
            item = QGraphicsPathItem(path)
            color = QColor(edge_color(kind))
            color.setAlpha(145)
            pen = QPen(color, 1.2 if kind == "data" else 1.5)
            if kind in ("residual", "skip"):
                pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(z_value)
            scene.addItem(item)
            self._edge_items.append(item)
            self._edge_specs.append((item, source[0], target[0], kind))

        for node in self.nodes:
            node_id = str(node.get("id"))
            position = self.positions.get(node_id)
            if not position:
                continue
            center, dims = position
            item = NodeItem(self, node, center, dims, kind_color(node.get("kind")))
            item.selected = node_id == self._selected_id
            item.highlighted = node_id in self._highlighted
            scene.addItem(item)
            self._node_items[node_id] = item

        self._add_labels()
        self._update_scene_rect()
        if fit:
            self._zoom = 1.0
        self._recompute_fit_cap()
        self._apply_camera()
        self.stats.setText(
            f"{len(self.nodes)} layers · {len(self.group_by_id)} groups · {len(self._edge_items)} edges · 厚度=通道数"
        )
        self.viewport().update()

    def _edge_geometry(self, source: tuple[float, float, float],
                       target: tuple[float, float, float], kind: str) -> tuple[QPainterPath, float]:
        start, start_depth = self.project(source)
        end, end_depth = self.project(target)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        bend = 8.0 + min(32.0, length * 0.14)
        if kind in ("residual", "skip"):
            bend = 22.0 + min(36.0, length * 0.18)
        # One control point on a single side gives a smooth arc.  Two
        # controls on opposite sides created a sharp S/spike for skips.
        apex = QPointF(
            start.x() + dx * 0.5 + nx * bend,
            start.y() + dy * 0.5 + ny * bend,
        )
        path = QPainterPath(start)
        path.quadTo(apex, end)
        return path, -150 + (start_depth + end_depth) * 0.02

    def _refresh_projection(self) -> None:
        """Re-project every item for the current camera without rebuilding the scene.

        Full ``scene.clear()`` + recreate on every mouse-move was the reason
        dragging felt like nothing happened until the button was released.
        """
        for item in self._node_items.values():
            item.refresh()
        for item in self._group_items:
            item.refresh()
        for item, source, target, kind in self._edge_specs:
            path, z_value = self._edge_geometry(source, target, kind)
            item.setPath(path)
            item.setZValue(z_value)
        for item, anchor in self._label_specs:
            point, _depth = self.project(anchor)
            item.place(point)
        self._apply_camera()
        self.viewport().update()

    def _add_labels(self) -> None:
        show_all = len(self.nodes) <= 18
        label_ids: list[str] = []
        if show_all:
            label_ids = [str(node.get("id")) for node in self.nodes]
        elif self._selected_id and self._selected_id in self.node_by_id:
            label_ids = [self._selected_id]
        for node_id in label_ids:
            node = self.node_by_id.get(node_id)
            position = self.positions.get(node_id)
            if not node or not position:
                continue
            center, dims = position
            text = str(node.get("name") or node_id)
            if len(text) > 22:
                text = text[:20] + "…"
            self._add_label(text, kind_color(node.get("kind")),
                            (center[0], center[1] + dims[1] / 2 + 0.25, center[2]))

    def _add_label(self, text: str, color: str, anchor: tuple[float, float, float]) -> None:
        point, _depth = self.project(anchor)
        item = PillLabelItem(text, color, point)
        self.scene().addItem(item)
        self._label_items.append(item)
        self._label_specs.append((item, anchor))

    def _visible_edges(self) -> list[dict[str, Any]]:
        if not self.edges:
            return []
        if len(self.nodes) <= 200:
            return self.edges
        order = {str(node.get("id")): index for index, node in enumerate(self.nodes)}
        major = [edge for edge in self.edges if str(edge.get("kind") or "data") != "data"]
        adjacent: list[dict[str, Any]] = []
        for edge in self.edges:
            if str(edge.get("kind") or "data") != "data":
                continue
            source = order.get(str(edge.get("source")))
            target = order.get(str(edge.get("target")))
            if source is not None and target is not None and abs(source - target) <= 1:
                adjacent.append(edge)
        result = (major + adjacent)[:340]
        return result or self.edges[:340]

    # ------------------------------------------------------------------ #
    # camera / fitting / interaction
    # ------------------------------------------------------------------ #
    def _projected_half_extent(self) -> tuple[float, float]:
        """Half width/height (scene pixels) of the model's projection.

        Computed from the rotation-invariant world bounds, so it is cheap and
        continuous while the model turns.
        """
        bounds = self._world_bounds
        if bounds is None:
            return 1.0, 1.0
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        half_x = 0.0
        half_y = 0.0
        for x in (min_x, max_x):
            for y in (min_y, max_y):
                for z in (min_z, max_z):
                    point, _depth = self.project((x, y, z))
                    half_x = max(half_x, abs(point.x()))
                    half_y = max(half_y, abs(point.y()))
        return max(1.0, half_x), max(1.0, half_y)

    def _required_scale(self) -> float:
        if self._world_bounds is None:
            return 1.0
        half_x, half_y = self._projected_half_extent()
        viewport = self.viewport()
        available_x = max(40.0, viewport.width() / 2.0 - FIT_MARGIN_X)
        available_y = max(30.0, viewport.height() / 2.0 - FIT_MARGIN_Y)
        return max(0.01, min(available_x / half_x, available_y / half_y))

    def _recompute_fit_cap(self) -> None:
        """Remember the zoom that fits the model at the default presentation angle."""
        yaw, pitch = self._yaw, self._pitch
        self._yaw, self._pitch = DEFAULT_YAW, DEFAULT_PITCH
        try:
            self._fit_cap = self._required_scale()
        finally:
            self._yaw, self._pitch = yaw, pitch

    def _apply_camera(self, *, refit: bool = False) -> None:
        """Place the model centre at the viewport centre with a stable zoom.

        The scale never grows past the default-presentation fit while orbiting
        (so turning the model cannot suddenly "zoom in"), but it does shrink as
        needed so the turned model still fits the viewport instead of being
        clipped by the edges.
        """
        if refit or self._fit_cap <= 0.0:
            self._recompute_fit_cap()
        scale = min(self._fit_cap, self._required_scale()) * self._zoom
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(0.0, 0.0)

    def _update_scene_rect(self) -> None:
        """A rotation-invariant, origin-centred scene rect.

        Letting the scene rect follow the projected items made Qt re-clamp the
        scroll bars while the model turned, which shifted the picture sideways.
        """
        bounds = self._world_bounds
        if bounds is None:
            self.scene().setSceneRect(QRectF(-320, -240, 640, 480))
            return
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        radius = 0.0
        for x in (min_x, max_x):
            for y in (min_y, max_y):
                for z in (min_z, max_z):
                    radius = max(radius, math.sqrt(x * x + y * y + z * z))
        half = max(240.0, radius * SCALE * 1.35 + 240.0)
        self.scene().setSceneRect(QRectF(-half, -half, 2 * half, 2 * half))

    def fit_scene(self) -> None:
        self._zoom = 1.0
        self._apply_camera(refit=True)

    def reset_view(self) -> None:
        self._yaw = DEFAULT_YAW
        self._pitch = DEFAULT_PITCH
        self._selected_id = self._selected_id if self._selected_id in self.node_by_id else None
        self._rebuild(fit=True)

    def _pick_export_angle(self) -> tuple[float, float]:
        """Pick the candidate whose projected bounds best fill a 16:9 canvas."""
        if not self.positions:
            return EXPORT_ANGLES[2]
        old_yaw, old_pitch = self._yaw, self._pitch
        best: tuple[float, float, float] | None = None
        try:
            for yaw, pitch in EXPORT_ANGLES:
                self._yaw, self._pitch = yaw, pitch
                xs: list[float] = []
                ys: list[float] = []
                for (cx, cy, cz), (dx, dy, dz) in self.positions.values():
                    for sx in (-0.5, 0.5):
                        for sy in (-0.5, 0.5):
                            for sz in (-0.5, 0.5):
                                point, _depth = self.project(
                                    (cx + dx * sx, cy + dy * sy, cz + dz * sz)
                                )
                                xs.append(point.x())
                                ys.append(point.y())
                if not xs:
                    continue
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                if width <= 0 or height <= 0:
                    continue
                aspect = width / height
                score = abs(math.log(max(0.01, aspect) / EXPORT_ASPECT))
                if best is None or score < best[0]:
                    best = (score, yaw, pitch)
        finally:
            self._yaw, self._pitch = old_yaw, old_pitch
        if best is None:
            return EXPORT_ANGLES[2]
        return best[1], best[2]

    def export_image(
        self,
        path: str | Path,
        *,
        width: int = 1920,
        height: int = 1080,
    ) -> Path:
        """Render the graph at an automatically chosen presentation angle into a PNG.

        The current camera angle is restored afterwards.  The exported scene
        includes labels/group boxes, not the Qt overlay widgets (hint/stats).
        """
        target = Path(path).expanduser()
        target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        width = max(640, int(width))
        height = max(360, int(height))
        old_yaw, old_pitch = self._yaw, self._pitch
        export_yaw, export_pitch = self._pick_export_angle()
        try:
            self._yaw = export_yaw
            self._pitch = export_pitch
            self._rebuild(fit=True)
            self.fit_scene()
            QApplication.processEvents()

            # The live scene rect is rotation-invariant (deliberately loose);
            # the exported picture wants the tight projected bounds instead.
            tight = self.scene().itemsBoundingRect()
            if tight.isEmpty():
                tight = QRectF(-320, -240, 640, 480)
            tight = tight.adjusted(-24, -24, 24, 24)
            self.scene().setSceneRect(tight)
            try:
                image = QImage(width, height, QImage.Format.Format_ARGB32)
                image.fill(QColor("#050b14"))
                painter = QPainter(image)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
                self.scene().render(
                    painter,
                    QRectF(0, 0, width, height),
                    tight,
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                painter.end()
            finally:
                self._update_scene_rect()
            if not image.save(str(target), "PNG"):
                raise RuntimeError(f"无法写入图片: {target}")
        finally:
            self._yaw = old_yaw
            self._pitch = old_pitch
            self._rebuild(fit=True)
            self.fit_scene()
        return target

    def set_caption(self, text: str) -> None:
        self.caption.setText(text)
        self.caption.adjustSize()
        self.caption.setVisible(bool(text))
        self._place_overlays()

    def _place_overlays(self) -> None:
        self.hint.adjustSize()
        self.hint.move(12, 10)
        self.stats.adjustSize()
        self.stats.move(12, 44)
        if self.caption.isVisible():
            self.caption.adjustSize()
            self.caption.move(max(12, self.viewport().width() - self.caption.width() - 12), 10)
        self.empty_label.setGeometry(0, 0, self.viewport().width(), self.viewport().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_overlays()
        if self.nodes:
            self._recompute_fit_cap()
            self._apply_camera()

    def _request_rotation_frame(self) -> None:
        """Repaint the rotation now if the frame budget allows, else coalesce.

        The old code re-armed a 16 ms single-shot timer on every mouse move, so
        a steady drag (moves every 5-10 ms) never let it fire: the model stayed
        frozen for the whole drag and then jumped to the final pose on release.
        """
        now = time.monotonic()
        elapsed = now - self._last_frame
        if elapsed >= FRAME_INTERVAL:
            self._last_frame = now
            self._rotation_timer.stop()
            self._refresh_projection()
        elif not self._rotation_timer.isActive():
            self._rotation_timer.start(max(1, int((FRAME_INTERVAL - elapsed) * 1000) + 1))

    def _apply_pending_rotation(self) -> None:
        self._last_frame = time.monotonic()
        self._refresh_projection()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._press_yaw = self._yaw
            self._press_pitch = self._pitch
            self._dragging = False
            self._last_frame = 0.0
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position().toPoint() - self._press_pos
            if abs(delta.x()) > 3 or abs(delta.y()) > 3:
                self._dragging = True
                self._yaw = self._press_yaw + delta.x() * YAW_PER_PIXEL
                self._pitch = max(
                    -PITCH_LIMIT,
                    min(PITCH_LIMIT, self._press_pitch + delta.y() * PITCH_PER_PIXEL),
                )
                self._request_rotation_frame()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            was_drag = self._dragging
            self._press_pos = None
            self._dragging = False
            self._rotation_timer.stop()
            if was_drag:
                # Land on the final pose even if the last move was throttled.
                self._refresh_projection()
            else:
                node = self._node_at(event.position().toPoint())
                if node is not None:
                    self.select_node(str(node.get("id")), notify=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if not delta:
            event.accept()
            return
        factor = ZOOM_FACTOR_120 ** (delta / 120.0)
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if abs(zoom - self._zoom) < 1e-9:
            event.accept()
            return
        self._zoom = zoom
        self._apply_camera()
        self.viewport().update()
        event.accept()

    def _node_at(self, pos: QPoint) -> dict[str, Any] | None:
        for item in self.items(pos):
            if isinstance(item, NodeItem):
                return item.node
        return None

    def select_node(self, node_id: str, *, notify: bool = False) -> None:
        node_id = str(node_id or "")
        if node_id not in self.node_by_id:
            return
        self._selected_id = node_id
        # rebuilding gives the node a visible label even on 400+ graphs
        self._rebuild(fit=False)
        if notify:
            self.nodeSelected.emit(node_id)

    def set_selected_node(self, node_id: str | None) -> None:
        node_id = str(node_id) if node_id is not None else ""
        self._selected_id = node_id if node_id in self.node_by_id else None
        for key, item in self._node_items.items():
            item.selected = key == self._selected_id
            item.update()

    def highlight_nodes(self, node_ids: list[str] | set[str] | None) -> None:
        self._highlighted = {str(node_id) for node_id in (node_ids or []) if str(node_id) in self.node_by_id}
        for key, item in self._node_items.items():
            item.highlighted = key in self._highlighted
            item.update()

    def node_hover_enter(self, node: dict[str, Any], screen_pos) -> None:
        self._hovered_id = str(node.get("id"))
        group = self.group_by_id.get(str(node.get("groupId") or ""))
        label = group_label(self.graph, group.get("id") if group else None)
        point = screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else QPoint(screen_pos)
        QToolTip.showText(point, node_tooltip(node, "" if label == "—" else label), self)

    def node_hover_leave(self) -> None:
        self._hovered_id = None
        QToolTip.hideText()

    # ------------------------------------------------------------------ #
    # painting backdrop
    # ------------------------------------------------------------------ #
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, QColor("#070b14"))
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor(13, 27, 46, 80))
        gradient.setColorAt(0.55, QColor(7, 13, 24, 20))
        gradient.setColorAt(1.0, QColor(22, 13, 37, 55))
        painter.fillRect(rect, gradient)
        step = SCALE
        painter.setPen(QPen(QColor(36, 62, 88, 38), 0.6))
        start_x = math.floor(rect.left() / step) * step
        start_y = math.floor(rect.top() / step) * step
        x = start_x
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = start_y
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step


def _first_shape(node: dict[str, Any]) -> list[int]:
    shapes = node.get("outShapes") or node.get("inShapes") or []
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], list):
        shape = shapes[0]
        if all(isinstance(dim, int) for dim in shape):
            return shape
    return []


def _shape_metrics(shape: list[int]) -> tuple[float, float, float]:
    if not shape:
        return 1.0, 1.0, 1.0
    dims = [max(1, int(dim)) for dim in shape]
    volume = float(math.prod(dims))
    if len(shape) >= 4:
        channels = float(dims[1])
        spatial = float(dims[2] * dims[3])
    elif len(shape) == 3:
        channels = float(dims[2])
        spatial = float(dims[1])
    elif len(shape) == 2:
        channels = float(dims[1])
        spatial = 1.0
    else:
        channels = float(dims[-1])
        spatial = 1.0
    return volume, spatial, channels


__all__ = ["Graph3DView", "NodeItem", "GroupItem", "PillLabelItem"]
