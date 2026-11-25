"""A dependency-free software-projected 3D graph view on QGraphicsView.

There is deliberately no OpenGL / WebEngine: every layer is a painted QGraphics
item.  The view stays usable with several hundred nodes by drawing a simplified
set of edges and suppressing per-node labels above a threshold.
"""

from __future__ import annotations

import math
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
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
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
        self._label_items: list[QGraphicsItem] = []
        self._selected_id: str | None = None
        self._hovered_id: str | None = None
        self._highlighted: set[str] = set()
        self._yaw = DEFAULT_YAW
        self._pitch = DEFAULT_PITCH
        self._needs_fit = True
        self._press_pos: QPoint | None = None
        self._press_yaw = self._yaw
        self._press_pitch = self._pitch
        self._dragging = False
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setSingleShot(True)
        self._rotation_timer.setInterval(16)
        self._rotation_timer.timeout.connect(lambda: self._rebuild(fit=False))

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
        self._compute_layout()
        self._needs_fit = True
        self._rebuild(fit=True)
        self.empty_label.setVisible(not self.nodes)
        self.update()

    def _compute_layout(self) -> None:
        self.positions = {}
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
        max_channel_count = max((item[2] for item in dims), default=1.0) or 1.0
        cursor = 0.0
        previous_group: str | None = None
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
            group_id = str(node.get("groupId") or "")
            if index:
                cursor += GROUP_GAP if group_id != previous_group and group_id else 2.05
            previous_group = group_id or None
            self.positions[str(node.get("id"))] = (
                (cursor, 0.0, 0.0),
                (dx, dy, dz),
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
        scene = self.scene()
        scene.clear()
        self._node_items.clear()
        self._group_items.clear()
        self._edge_items.clear()
        self._label_items.clear()
        if not self.nodes:
            scene.setSceneRect(QRectF(-320, -240, 640, 480))
            self.stats.setText("0 layers")
            self.empty_label.setVisible(True)
            self.hint.setVisible(False)
            if fit:
                self._needs_fit = True
                QTimer.singleShot(0, self.fit_scene)
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
                min(xs) - 0.52,
                max(xs) + 0.52,
                min(ys) - 0.32,
                max(ys) + 0.32,
                min(zs) - 0.32,
                max(zs) + 0.32,
            )
            item = GroupItem(self, group, bounds)
            scene.addItem(item)
            self._group_items.append(item)
            center_x = (bounds[0] + bounds[1]) / 2
            label_point, _depth = self.project((center_x, bounds[3] + 0.52, 0.0))
            label = group.get("label") or group.get("name") or group.get("id")
            self._add_label(str(label), group_color(group.get("kind")), label_point)

        edge_items = self._visible_edges()
        for edge in edge_items:
            source = self.positions.get(str(edge.get("source")))
            target = self.positions.get(str(edge.get("target")))
            if not source or not target:
                continue
            start, start_depth = self.project(source[0])
            end, end_depth = self.project(target[0])
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            kind = str(edge.get("kind") or "data")
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
            item = QGraphicsPathItem(path)
            color = QColor(edge_color(kind))
            color.setAlpha(145)
            pen = QPen(color, 1.2 if kind == "data" else 1.5)
            if kind in ("residual", "skip"):
                pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(-150 + (start_depth + end_depth) * 0.02)
            scene.addItem(item)
            self._edge_items.append(item)

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

        rect = scene.itemsBoundingRect()
        if rect.isEmpty():
            rect = QRectF(-320, -240, 640, 480)
        scene.setSceneRect(rect.adjusted(-90, -80, 90, 80))
        self.stats.setText(
            f"{len(self.nodes)} layers · {len(self.group_by_id)} groups · {len(self._edge_items)} edges · 厚度=通道数"
        )
        if fit:
            self._needs_fit = True
            QTimer.singleShot(0, self.fit_scene)
        self.update()

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
            point, _depth = self.project((center[0], center[1] + dims[1] / 2 + 0.25, center[2]))
            text = str(node.get("name") or node_id)
            if len(text) > 22:
                text = text[:20] + "…"
            self._add_label(text, kind_color(node.get("kind")), point)

    def _add_label(self, text: str, color: str, point: QPointF) -> None:
        item = PillLabelItem(text, color, point)
        self.scene().addItem(item)
        self._label_items.append(item)

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
    def fit_scene(self) -> None:
        rect = self.scene().sceneRect()
        if rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(rect.adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)
        self._needs_fit = False

    def reset_view(self) -> None:
        self._yaw = DEFAULT_YAW
        self._pitch = DEFAULT_PITCH
        self._selected_id = self._selected_id if self._selected_id in self.node_by_id else None
        self._needs_fit = True
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
        old_fit = self._needs_fit
        export_yaw, export_pitch = self._pick_export_angle()
        try:
            self._yaw = export_yaw
            self._pitch = export_pitch
            self._needs_fit = False
            self._rebuild(fit=True)
            self.fit_scene()
            QApplication.processEvents()

            image = QImage(width, height, QImage.Format.Format_ARGB32)
            image.fill(QColor("#050b14"))
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            source = self.scene().sceneRect().adjusted(-24, -24, 24, 24)
            self.scene().render(
                painter,
                QRectF(0, 0, width, height),
                source,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            painter.end()
            if not image.save(str(target), "PNG"):
                raise RuntimeError(f"无法写入图片: {target}")
        finally:
            self._yaw = old_yaw
            self._pitch = old_pitch
            self._needs_fit = old_fit or True
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
        if self._needs_fit:
            QTimer.singleShot(0, self.fit_scene)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._press_yaw = self._yaw
            self._press_pitch = self._pitch
            self._dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position().toPoint() - self._press_pos
            if abs(delta.x()) > 3 or abs(delta.y()) > 3:
                self._dragging = True
                self._yaw = self._press_yaw + delta.x() * 0.0075
                self._pitch = max(-1.15, min(1.15, self._press_pitch + delta.y() * 0.0065))
                self._rotation_timer.start()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            was_drag = self._dragging
            self._press_pos = None
            self._dragging = False
            if not was_drag:
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
        factor = 1.13 if event.angleDelta().y() > 0 else 0.885
        current = self.transform().m11()
        if current * factor < 0.12 or current * factor > 9.0:
            event.accept()
            return
        self.scale(factor, factor)
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
