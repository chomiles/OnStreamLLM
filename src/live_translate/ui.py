from __future__ import annotations

import html
import logging
import sys
import threading
import time
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path

from PySide6.QtCore import QFileInfo, QObject, QPoint, QRect, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QCloseEvent, QFont, QFontDatabase, QIcon, QKeyEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileIconProvider,
    QProgressBar,
    QRubberBand,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioCapture, list_microphones
from .config import (
    APP_DISPLAY_NAME,
    SENSEVOICE_ASR_MODEL,
    app_icon_path,
    save_settings,
    set_windows_app_user_model_id,
    settings_path,
)
from .compute import (
    format_cpu_core_ids,
    list_compute_devices,
    list_cpu_cores,
    normalize_device,
    validate_device,
)
from .engines import TranslationPipeline
from .event_bus import CaptionBus
from .models import AppSettings, Caption, CaptionStyle, SourceKind
from .model_store import (
    discover_models,
    display_path,
    delete_model,
    ensure_model_folders,
    is_model_complete,
    model_download_path,
    repository_id_from_path,
)
from .ocr import (
    KOREAN_OCR_MODEL_ID,
    OCR_STATUS_LOADING,
    OCR_STATUS_READY,
    ScreenOcr,
    download_korean_ocr_model,
    is_korean_ocr_model_ready,
    korean_ocr_model_dir,
)
from .runtime_dependencies import (
    RUNTIME_DEPENDENCIES,
    RuntimeDependency,
    all_runtime_dependencies_installed,
    clear_runtime_install_state,
    runtime_dependencies_in_install_order,
    runtime_dependencies_pending_install,
    runtime_dependency_installed,
    runtime_install_available,
)
from .gpu_driver import CUDA_RUNTIME_DEPENDENCIES, check_cuda_driver_compatibility, runtime_driver_status_message
from .i18n import (
    ENGINE_LOAD_COMPLETE_TOKENS,
    ENGINE_PRELOAD_FAIL_TOKENS,
    FAILURE_TOKENS,
    SUPPORTED_LANGUAGES,
    Translator,
    load_info_section,
    translate_driver_message,
)
from .runtime_bootstrap import release_runtime_libraries
from .runtime_install_queue import (
    clear_runtime_install_queue,
    load_runtime_install_queue,
)
from .runtime_installer import install_runtime_dependency, terminate_active_install_processes
from .server import (
    CaptionServer,
    LocalOverlayServer,
    RemoteClient,
    join_ip_list,
    list_local_ipv4_addresses,
    parse_ip_list,
)

OCR_SWITCH_COOLDOWN_SECONDS = 10
LOGGER = logging.getLogger(__name__)


def _application_icon() -> QIcon:
    icon_path = app_icon_path()
    icon = QIcon(str(icon_path)) if icon_path is not None else QIcon()
    if icon.isNull() and getattr(sys, "frozen", False):
        icon = QFileIconProvider().icon(QFileInfo(sys.executable))
    return icon


class UiBridge(QObject):
    caption = Signal(object)
    status = Signal(str)
    source_status = Signal(object, str)
    model_downloaded = Signal(str, str)
    model_download_progress = Signal(int, str)
    runtime_downloaded = Signal(str, str)
    runtime_download_progress = Signal(str, int, str)
    remote_info = Signal(object)
    inline_translation = Signal(object)


class ScreenRegionSelector(QDialog):
    region_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(None)
        self._origin = QPoint()
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def begin(self) -> None:
        screens = QApplication.screens()
        if not screens:
            return
        geometry = screens[0].geometry()
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        self.setGeometry(geometry)
        self.show()
        self.raise_()
        self.activateWindow()
        self.grabMouse()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 76))

    def mousePressEvent(self, event: object) -> None:
        self._origin = event.position().toPoint()
        self._rubber_band.setGeometry(QRect(self._origin, self._origin))
        self._rubber_band.show()

    def mouseMoveEvent(self, event: object) -> None:
        self._rubber_band.setGeometry(QRect(self._origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event: object) -> None:
        local = QRect(self._origin, event.position().toPoint()).normalized()
        self._rubber_band.hide()
        self.releaseMouse()
        self.hide()
        if local.width() < 20 or local.height() < 20:
            return
        global_top_left = self.mapToGlobal(local.topLeft())
        self.region_selected.emit(
            {
                "left": global_top_left.x(),
                "top": global_top_left.y(),
                "width": local.width(),
                "height": local.height(),
            }
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._rubber_band.hide()
            self.releaseMouse()
            self.hide()


class AutoScrollTextEdit(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self._auto_scroll = True
        self.verticalScrollBar().valueChanged.connect(self._scroll_position_changed)

    def _scroll_position_changed(self, value: int) -> None:
        scrollbar = self.verticalScrollBar()
        self._auto_scroll = value >= scrollbar.maximum() - 2

    def set_auto_scroll_html(self, value: str) -> None:
        scrollbar = self.verticalScrollBar()
        was_auto_scrolling = self._auto_scroll
        previous_position = scrollbar.value()
        self.setHtml(value)
        if was_auto_scrolling:
            scrollbar.setValue(scrollbar.maximum())
            self._auto_scroll = True
        else:
            scrollbar.setValue(min(previous_position, scrollbar.maximum()))
            self._auto_scroll = False


@lru_cache(maxsize=1)
def windows_font_families() -> tuple[str, ...]:
    return tuple(sorted(QFontDatabase.families(), key=str.casefold))


class SearchableFontCombo(QComboBox):
    def __init__(self, current: str = "", *, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        families = list(windows_font_families())
        selected = current.strip()
        if selected and selected not in families:
            families.insert(0, selected)
        self.addItems(families)
        index = self.findText(selected, Qt.MatchFlag.MatchExactly)
        if index >= 0:
            self.setCurrentIndex(index)
        elif selected:
            self.setEditText(selected)
        completer = QCompleter(families, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(completer)
        self.setMaxVisibleItems(24)
        line_edit = self.lineEdit()
        if line_edit is not None and placeholder:
            line_edit.setPlaceholderText(placeholder)

    def set_placeholder(self, placeholder: str) -> None:
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText(placeholder)

    def current_font_family(self) -> str:
        return self.currentText().strip()


class CollapsibleSection(QWidget):
    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.toggled.connect(self._on_toggle)
        self.toggle.setStyleSheet(
            "QToolButton { text-align: left; padding: 6px 8px; font-weight: 600; "
            "border: 1px solid #555; border-radius: 4px; }"
        )
        self.content = content
        self.content.setVisible(expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def set_title(self, title: str) -> None:
        self.toggle.setText(title)

    def _on_toggle(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class SwitchToggle(QCheckBox):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(26)

    def sizeHint(self) -> QSize:
        width = 54 + self.fontMetrics().horizontalAdvance(self.text())
        return QSize(width, 28)

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRect(0, (self.height() - 22) // 2, 42, 22)
        enabled = self.isEnabled()
        checked = self.isChecked()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2f9e72" if checked and enabled else "#4b5563" if enabled else "#343a40"))
        painter.drawRoundedRect(track, 11, 11)
        knob_x = track.right() - 19 if checked else track.left() + 3
        painter.setBrush(QColor("#ffffff" if enabled else "#9ca3af"))
        painter.drawEllipse(QRect(knob_x, track.top() + 3, 16, 16))
        painter.setPen(QColor("#dddddd" if enabled else "#888888"))
        painter.drawText(QRect(52, 0, self.width() - 52, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


def _html_preserve_lines(text: str) -> str:
    return "<br>".join(html.escape(line) for line in text.splitlines()) or html.escape(text)


POPUP_FONT_SIZES: tuple[int, ...] = (18, 20, 24, 28, 32, 36, 42, 48, 56, 64, 72)
POPUP_OPACITY_OPTIONS: tuple[int, ...] = (100, 90, 80, 70, 60, 50, 40, 30)


class TransparentCaptionPopup(QWidget):
    def __init__(
        self,
        settings: AppSettings,
        *,
        translate: Callable[[str], str],
        on_settings_changed: Callable[[], None],
    ) -> None:
        super().__init__()
        self.settings = settings
        self._translate = translate
        self._on_settings_changed = on_settings_changed
        self._drag_origin = QPoint()
        self._drag_active = False
        self.setWindowTitle("OnStreamLLM Overlay")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(360, 180)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._panel = QWidget(self)
        self._panel.setObjectName("popupPanel")
        self._panel.setStyleSheet(
            "#popupPanel {"
            "background-color: rgba(18, 18, 22, 160);"
            "border: 1px solid rgba(255, 255, 255, 40);"
            "border-radius: 8px;"
            "}"
        )
        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(6)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        self.lock_checkbox = QCheckBox()
        self.lock_checkbox.setChecked(self.settings.popup_locked)
        self.lock_checkbox.toggled.connect(self._lock_changed)
        self.font_combo = SearchableFontCombo(
            self.settings.popup_font_family,
            placeholder=self._translate("style.font_search"),
        )
        self.font_combo.currentTextChanged.connect(self._font_changed)
        self.font_size_combo = NoWheelComboBox()
        for size in POPUP_FONT_SIZES:
            self.font_size_combo.addItem(str(size), size)
        self._set_combo_value(self.font_size_combo, self.settings.popup_font_size)
        self.font_size_combo.currentIndexChanged.connect(self._font_size_changed)
        self.opacity_combo = NoWheelComboBox()
        for opacity in POPUP_OPACITY_OPTIONS:
            self.opacity_combo.addItem(
                self._translate("popup.opacity_percent", percent=opacity),
                opacity,
            )
        self._set_combo_value(self.opacity_combo, self.settings.popup_opacity_percent)
        self.opacity_combo.currentIndexChanged.connect(self._opacity_changed)
        font_label = QLabel()
        font_label.setObjectName("popupFontLabel")
        size_label = QLabel()
        size_label.setObjectName("popupSizeLabel")
        opacity_label = QLabel()
        opacity_label.setObjectName("popupOpacityLabel")
        controls_layout.addWidget(self.lock_checkbox)
        controls_layout.addWidget(font_label)
        controls_layout.addWidget(self.font_combo, 1)
        controls_layout.addWidget(size_label)
        controls_layout.addWidget(self.font_size_combo)
        controls_layout.addWidget(opacity_label)
        controls_layout.addWidget(self.opacity_combo)
        panel_layout.addWidget(controls)

        self.caption = AutoScrollTextEdit()
        self.caption.setReadOnly(True)
        self.caption.setFrameStyle(0)
        self.caption.setStyleSheet(
            "background: rgba(0, 0, 0, 0); border: 0; padding: 4px;"
        )
        panel_layout.addWidget(self.caption, 1)

        self._grip_row = QWidget()
        grip_layout = QHBoxLayout(self._grip_row)
        grip_layout.setContentsMargins(0, 0, 0, 0)
        grip_layout.addStretch(1)
        self._size_grip = QSizeGrip(self)
        self._size_grip.setStyleSheet("width: 16px; height: 16px;")
        grip_layout.addWidget(
            self._size_grip,
            0,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )
        panel_layout.addWidget(self._grip_row)

        root.addWidget(self._panel)
        self._apply_geometry()
        self._apply_opacity()
        self._apply_lock_state()
        self.retranslate()

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: int) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def retranslate(self) -> None:
        self.lock_checkbox.setText(self._translate("popup.lock_position"))
        font_label = self.findChild(QLabel, "popupFontLabel")
        size_label = self.findChild(QLabel, "popupSizeLabel")
        opacity_label = self.findChild(QLabel, "popupOpacityLabel")
        if font_label is not None:
            font_label.setText(self._translate("popup.font"))
        if size_label is not None:
            size_label.setText(self._translate("popup.font_size"))
        if opacity_label is not None:
            opacity_label.setText(self._translate("popup.opacity"))
        current_opacity = int(self.opacity_combo.currentData() or self.settings.popup_opacity_percent)
        for index in range(self.opacity_combo.count()):
            opacity = int(self.opacity_combo.itemData(index) or 0)
            self.opacity_combo.setItemText(
                index,
                self._translate("popup.opacity_percent", percent=opacity),
            )
        self._set_combo_value(self.opacity_combo, current_opacity)
        self.font_combo.set_placeholder(self._translate("style.font_search"))

    def set_caption_html(self, value: str) -> None:
        self.caption.set_auto_scroll_html(value)

    def _apply_geometry(self) -> None:
        width = max(self.minimumWidth(), self.settings.popup_width)
        height = max(self.minimumHeight(), self.settings.popup_height)
        self.resize(width, height)
        if self.settings.popup_x >= 0 and self.settings.popup_y >= 0:
            self.move(self.settings.popup_x, self.settings.popup_y)

    def _apply_opacity(self) -> None:
        opacity = max(0.1, min(1.0, self.settings.popup_opacity_percent / 100))
        self.setWindowOpacity(opacity)

    def _persist_geometry(self) -> None:
        self.settings.popup_x = self.x()
        self.settings.popup_y = self.y()
        self.settings.popup_width = self.width()
        self.settings.popup_height = self.height()

    def _notify_settings_changed(self) -> None:
        self._on_settings_changed()

    def _lock_changed(self, checked: bool) -> None:
        self.settings.popup_locked = checked
        self._apply_lock_state()
        self._notify_settings_changed()

    def _apply_lock_state(self) -> None:
        locked = self.settings.popup_locked
        self.font_combo.setEnabled(not locked)
        self.font_size_combo.setEnabled(not locked)
        self.opacity_combo.setEnabled(not locked)
        if locked:
            self._grip_row.hide()
            self._size_grip.setEnabled(False)
            self.setFixedSize(self.size())
            return
        self._grip_row.show()
        self._size_grip.setEnabled(True)
        self.setMinimumSize(360, 180)
        self.setMaximumSize(16777215, 16777215)
        self.updateGeometry()

    def _font_changed(self, family: str) -> None:
        selected = family.strip()
        if not selected:
            return
        self.settings.popup_font_family = selected
        self._notify_settings_changed()

    def _font_size_changed(self, _index: int) -> None:
        size = int(self.font_size_combo.currentData() or self.settings.popup_font_size)
        self.settings.popup_font_size = size
        self._notify_settings_changed()

    def _opacity_changed(self, _index: int) -> None:
        opacity = int(self.opacity_combo.currentData() or self.settings.popup_opacity_percent)
        self.settings.popup_opacity_percent = opacity
        self._apply_opacity()
        self._notify_settings_changed()

    def moveEvent(self, event: object) -> None:
        super().moveEvent(event)
        self._persist_geometry()

    def resizeEvent(self, event: object) -> None:
        if self.settings.popup_locked:
            locked_size = self.size()
            super().resizeEvent(event)
            if self.size() != locked_size:
                self.setFixedSize(locked_size)
            return
        super().resizeEvent(event)
        self._persist_geometry()

    def mousePressEvent(self, event: object) -> None:
        if self.settings.popup_locked:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        target = self.childAt(event.position().toPoint())
        if target is not None and target is not self._panel and not self._is_movable_target(target):
            return
        self._drag_active = True
        self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: object) -> None:
        if not self._drag_active or self.settings.popup_locked:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)

    def mouseReleaseEvent(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False

    def _is_movable_target(self, widget: QWidget) -> bool:
        interactive = {
            self.lock_checkbox,
            self.font_combo,
            self.font_size_combo,
            self.opacity_combo,
        }
        current: QWidget | None = widget
        while current is not None:
            if current in interactive:
                return False
            if current is self.caption or current is self._panel or isinstance(current, QLabel):
                return True
            current = current.parentWidget()
        return False


MODEL_VRAM_HINTS_GB: dict[str, float] = {
    "Qwen/Qwen3-ASR-0.6B": 2.5,
    "Qwen/Qwen3-ASR-1.7B": 4.5,
    SENSEVOICE_ASR_MODEL: 0.0,
    "Qwen/Qwen3-4B": 12.0,
    "Qwen/Qwen3-8B-AWQ": 16.0,
    "tencent/Hy-MT2-1.8B-GGUF": 3.0,
    "tencent/Hy-MT2-1.8B-2bit-GGUF": 2.0,
}

MODEL_LABEL_NOTES: dict[str, str] = {
    SENSEVOICE_ASR_MODEL: "CPU 전용",
    "Qwen/Qwen3-4B": "무거움 12GB 이상 VRAM, 게임 미가동시 사용",
    "Qwen/Qwen3-8B-AWQ": "매우 무거움 16GB 이상 VRAM에서 게임 미가동시 사용",
}


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.tr = Translator(settings.ui_language)
        self._fresh_settings = not settings_path().exists()
        ensure_model_folders()
        self.bus = CaptionBus()
        self.bridge = UiBridge()
        self.bridge.caption.connect(self._show_caption)
        self.bridge.status.connect(self._set_status)
        self.bridge.source_status.connect(self._set_source_status)
        self.bridge.model_downloaded.connect(self._model_download_finished)
        self.bridge.model_download_progress.connect(self._model_download_progress)
        self.bridge.runtime_downloaded.connect(self._runtime_download_finished)
        self.bridge.runtime_download_progress.connect(self._runtime_download_progress)
        self.bridge.remote_info.connect(self._update_remote_info)
        self.bridge.inline_translation.connect(self._inline_translation_finished)
        self.bus.subscribe(self.bridge.caption.emit)
        self._pipeline_generation = 0
        self.pipeline = TranslationPipeline(
            settings,
            self.bus,
            partial(self._emit_pipeline_source_status, self._pipeline_generation),
        )
        self.server: CaptionServer | None = None
        self.overlay_server = LocalOverlayServer(settings, self.bus)
        self.remote: RemoteClient | None = None
        self.local_running = False
        self.local_starting = False
        self.remote_connected = False
        self.remote_model_info: dict[str, object] = {}
        self.captures: dict[SourceKind, AudioCapture] = {}
        self.caption_popup = TransparentCaptionPopup(
            self.settings,
            translate=self._t,
            on_settings_changed=self._on_popup_settings_changed,
        )
        self._shutdown_done = False
        self._active_model_downloads: set[tuple[str, str]] = set()
        self._download_cancel_events: dict[tuple[str, str], threading.Event] = {}
        self._download_queue: list[tuple[str, str]] = []
        self._current_model_download: tuple[str, str] | None = None
        self._runtime_downloads: dict[str, threading.Event] = {}
        self._runtime_install_queue: list[tuple[str, bool]] = []
        self._runtime_install_active: str | None = None
        self._runtime_install_batch_total: int = 0
        self._runtime_required_install_batch_active = False
        self._runtime_restart_required_names: set[str] = set()
        self._runtime_panel_collapsed = False
        self._i18n_form_labels: dict[str, QLabel] = {}
        self._i18n_group_boxes: dict[str, QGroupBox] = {}
        self._i18n_channel_boxes: dict[SourceKind, QGroupBox] = {}
        self._i18n_collapsible_sections: dict[str, CollapsibleSection] = {}
        self._i18n_inline_labels: dict[str, QLabel] = {}
        self._runtime_desc_labels: dict[str, QLabel] = {}
        self._status_base_message = self._t("status.ready")
        self._status_busy = False
        self._status_frame_index = 0
        self._status_frames = [".", "..", "..."]
        self._pending_ocr_after_korean_download = False
        self._ocr_cooldown_remaining = 0
        self.ocr = ScreenOcr(
            partial(self._process_text, SourceKind.SCREEN),
            interval=self.settings.ocr_interval,
            auto_refresh=self.settings.ocr_auto_refresh,
            region=self._saved_ocr_region(),
            device="cpu",
            source_language=self.settings.source_language,
            status_callback=self._emit_ocr_status,
        )
        self.region_selector = ScreenRegionSelector()
        self.region_selector.region_selected.connect(self._ocr_region_selected)

        self.setWindowTitle(APP_DISPLAY_NAME)
        icon = _application_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.resize(980, 720)
        self._build_ui()
        self.status_animation = QTimer(self)
        self.status_animation.timeout.connect(self._tick_status_animation)
        self.status_animation.start(250)
        self.ocr_cooldown_timer = QTimer(self)
        self.ocr_cooldown_timer.timeout.connect(self._tick_ocr_cooldown)
        self._refresh_devices()
        if self.settings.transparent_popup_enabled:
            self._toggle_caption_popup()
        QTimer.singleShot(0, self._restore_runtime_state)
        if self._fresh_settings:
            self._apply_initial_game_light_preset()
        else:
            self._ensure_preset_device_defaults()
        QTimer.singleShot(0, self._clamp_model_tab_width)

    def _t(self, key: str, **kwargs: object) -> str:
        return self.tr.t(key, **kwargs)

    def _form_label(self, key: str) -> QLabel:
        label = QLabel(self._t(key))
        self._i18n_form_labels[key] = label
        return label

    def _make_group(self, key: str) -> QGroupBox:
        box = QGroupBox(self._t(key))
        self._i18n_group_boxes[key] = box
        return box

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), self._t("tab.monitor"))
        self.tabs.addTab(self._build_model_tab(), self._t("tab.models"))
        self.tabs.addTab(self._build_settings_tab(), self._t("tab.settings"))
        self.tabs.addTab(self._build_info_tab(), self._t("tab.info"))
        self.tabs.currentChanged.connect(lambda _index: QTimer.singleShot(0, self._clamp_model_tab_width))
        layout.addWidget(self.tabs)
        self.status = QLabel(self._t("status.prefix", message=self._status_base_message))
        self.status.setFixedHeight(28)
        self.status.setWordWrap(False)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setStyleSheet(
            "padding: 5px 8px; border-top: 1px solid #555; color: #dddddd;"
        )
        layout.addWidget(self.status)
        self.setCentralWidget(root)

    def _build_monitor_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addStretch(1)
        header_layout.addWidget(self._form_label("ui_language"))
        self.ui_language_combo = NoWheelComboBox()
        for code in SUPPORTED_LANGUAGES:
            self.ui_language_combo.addItem(self._t(f"lang.{code}"), code)
        lang_index = self.ui_language_combo.findData(self.settings.ui_language)
        self.ui_language_combo.setCurrentIndex(max(0, lang_index))
        self.ui_language_combo.currentIndexChanged.connect(self._ui_language_changed)
        header_layout.addWidget(self.ui_language_combo)
        layout.addWidget(header)
        dashboard = self._make_group("dashboard.title")
        dashboard_layout = QFormLayout(dashboard)
        self.local_models_status = QLabel()
        self.runtime_status = QLabel()
        self.remote_destination_status = QLabel()
        self.remote_models_status = QLabel(self._t("dashboard.host_models_pending"))
        self.local_start = QPushButton(self._t("btn.engine_start"))
        self.local_start.clicked.connect(self._toggle_local_mode)
        self.local_stop = QPushButton(self._t("btn.engine_stop"))
        self.local_stop.clicked.connect(self._stop_local_mode)
        self.client_start = QPushButton(self._t("btn.client_start"))
        self.client_start.clicked.connect(self._toggle_remote)
        run_buttons = QWidget()
        run_buttons_layout = QHBoxLayout(run_buttons)
        run_buttons_layout.setContentsMargins(0, 0, 0, 0)
        run_buttons_layout.addWidget(self.local_start)
        run_buttons_layout.addWidget(self.local_stop)
        run_buttons_layout.addWidget(self.client_start)
        self.input_toggle = SwitchToggle(self._t("toggle.input_detection"))
        self.input_toggle.toggled.connect(partial(self._toggle_capture, SourceKind.INPUT))
        self.output_toggle = SwitchToggle(self._t("toggle.output_detection"))
        self.output_toggle.toggled.connect(partial(self._toggle_capture, SourceKind.OUTPUT))
        self.ocr_toggle = SwitchToggle(self._t("toggle.screen_detection"))
        self.ocr_toggle.toggled.connect(self._toggle_ocr)
        detection_toggles = QWidget()
        detection_toggles_layout = QHBoxLayout(detection_toggles)
        detection_toggles_layout.setContentsMargins(0, 0, 0, 0)
        detection_toggles_layout.addWidget(self.input_toggle)
        detection_toggles_layout.addSpacing(24)
        detection_toggles_layout.addWidget(self.output_toggle)
        detection_toggles_layout.addSpacing(24)
        detection_toggles_layout.addWidget(self.ocr_toggle)
        detection_toggles_layout.addStretch(1)
        dashboard_layout.addRow(self._form_label("dashboard.current_models"), self.local_models_status)
        dashboard_layout.addRow(self._form_label("dashboard.runtime_status"), self.runtime_status)
        dashboard_layout.addRow(self._form_label("dashboard.client_destination"), self.remote_destination_status)
        dashboard_layout.addRow(self._form_label("dashboard.destination_models"), self.remote_models_status)
        dashboard_layout.addRow(run_buttons)
        dashboard_layout.addRow(self._form_label("dashboard.detection_toggles"), detection_toggles)
        self.ocr_region = QLabel(self._ocr_region_text())
        self.select_ocr_region = QPushButton(self._t("btn.select_screen_region"))
        self.select_ocr_region.clicked.connect(self.region_selector.begin)
        ocr_buttons = QWidget()
        ocr_buttons_layout = QHBoxLayout(ocr_buttons)
        ocr_buttons_layout.setContentsMargins(0, 0, 0, 0)
        ocr_buttons_layout.addWidget(self.select_ocr_region)
        ocr_buttons_layout.addStretch(1)
        dashboard_layout.addRow(self._form_label("dashboard.screen_region"), self.ocr_region)
        dashboard_layout.addRow(ocr_buttons)
        layout.addWidget(dashboard)
        self._refresh_dashboard()
        language_box = self._make_group("channel_languages.title")
        language_layout = QFormLayout(language_box)
        languages = [
            "auto", "Korean", "English", "Japanese", "Chinese", "Spanish",
            "French", "German", "Russian", "Portuguese", "Italian", "Vietnamese",
            "Thai", "Indonesian", "Arabic", "Hindi",
        ]
        self.input_device = NoWheelComboBox()
        self.output_device = NoWheelComboBox()
        self.channel_languages: dict[SourceKind, tuple[QComboBox, QComboBox]] = {}
        channel_keys = {
            SourceKind.INPUT: "channel.input_audio",
            SourceKind.OUTPUT: "channel.output_audio",
            SourceKind.SCREEN: "channel.screen",
        }
        for source, source_value, target_value in (
            (SourceKind.INPUT, self.settings.input_source_language, self.settings.input_target_language),
            (SourceKind.OUTPUT, self.settings.output_source_language, self.settings.output_target_language),
            (SourceKind.SCREEN, self.settings.source_language, self.settings.target_language),
        ):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            source_combo = NoWheelComboBox()
            target_combo = NoWheelComboBox()
            source_combo.addItems(languages)
            target_combo.addItems(languages[1:])
            source_combo.setCurrentText(source_value)
            target_combo.setCurrentText(target_value)
            source_combo.currentTextChanged.connect(self._language_settings_changed)
            target_combo.currentTextChanged.connect(self._language_settings_changed)
            source_label = QLabel(self._t("label.source"))
            target_label = QLabel(self._t("label.translation"))
            has_device_selector = source in {SourceKind.INPUT, SourceKind.OUTPUT}
            language_stretch = 2 if has_device_selector else 1
            row_layout.addWidget(source_label)
            row_layout.addWidget(source_combo, language_stretch)
            row_layout.addWidget(target_label)
            row_layout.addWidget(target_combo, language_stretch)
            if has_device_selector:
                device_label_key = (
                    "audio.input_device" if source == SourceKind.INPUT else "audio.output_device"
                )
                device_combo = self.input_device if source == SourceKind.INPUT else self.output_device
                row_layout.addWidget(QLabel(self._t(device_label_key)))
                row_layout.addWidget(device_combo, 3)
            language_layout.addRow(self._form_label(channel_keys[source]), row)
            self.channel_languages[source] = (source_combo, target_combo)
        rules = QPushButton(self._t("btn.edit_llm_rules"))
        rules.clicked.connect(self._edit_llm_rules)
        language_layout.addRow(rules)
        self.refresh_devices_button = QPushButton(self._t("audio.refresh_devices"))
        self.refresh_devices_button.clicked.connect(self._refresh_devices)
        language_layout.addRow(self.refresh_devices_button)
        layout.addWidget(language_box)
        self.monitor_labels: dict[SourceKind, QLabel] = {}
        self.caption_history: dict[SourceKind, list[Caption]] = {}
        self.source_status_labels: dict[SourceKind, QLabel] = {}
        self.source_status_history: dict[SourceKind, list[str]] = {}
        monitor_channel_keys = {
            SourceKind.INPUT: "channel.input_audio",
            SourceKind.OUTPUT: "channel.output_audio",
            SourceKind.SCREEN: "channel.screen_text",
        }
        for source in (SourceKind.INPUT, SourceKind.OUTPUT, SourceKind.SCREEN):
            box = self._make_group(monitor_channel_keys[source])
            self._i18n_channel_boxes[source] = box
            box_layout = QVBoxLayout(box)
            status = QLabel(self._t("source.stopped"))
            status.setStyleSheet("color: #aaaaaa;")
            box_layout.addWidget(status)
            label = QLabel("...")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setMinimumHeight(100)
            box_layout.addWidget(label)
            layout.addWidget(box)
            box.hide()
            self.monitor_labels[source] = label
            self.caption_history[source] = []
            self.source_status_labels[source] = status
            self.source_status_history[source] = [self._t("source.stopped")]
        output_controls = QWidget()
        output_controls_layout = QHBoxLayout(output_controls)
        output_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.omit_original = QCheckBox(self._t("checkbox.omit_original"))
        self.omit_original.setChecked(self.settings.omit_original_text)
        self.omit_original.toggled.connect(self._omit_original_changed)
        self.popup_toggle = QPushButton(self._t("btn.popup_start"))
        self.popup_toggle.clicked.connect(self._toggle_caption_popup)
        output_controls_layout.addWidget(self.omit_original)
        output_controls_layout.addWidget(self.popup_toggle)
        output_controls_layout.addStretch(1)
        layout.addWidget(output_controls)

        output_box = self._make_group("unified_output.title")
        output_layout = QVBoxLayout(output_box)
        self.unified_caption_view = AutoScrollTextEdit()
        self.unified_caption_view.setReadOnly(True)
        self.unified_caption_view.setMinimumHeight(280)
        self.unified_caption_view.setStyleSheet("background: #181818; border: 0; padding: 8px;")
        output_layout.addWidget(self.unified_caption_view)
        layout.addWidget(output_box, 1)
        self.unified_caption_history: list[Caption] = []

        layout.addWidget(self._build_inline_translator())
        self._render_unified_captions()
        return page

    def _build_inline_translator(self) -> QWidget:
        box = self._make_group("inline_translator.title")
        layout = QVBoxLayout(box)
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.inline_translator_enabled = QCheckBox(self._t("inline_translator.enable"))
        self.inline_translator_enabled.toggled.connect(self._toggle_inline_translator)
        self.inline_source_language = NoWheelComboBox()
        self.inline_target_language = NoWheelComboBox()
        languages = [
            "auto", "Korean", "English", "Japanese", "Chinese", "Spanish",
            "French", "German", "Russian", "Portuguese", "Italian", "Vietnamese",
            "Thai", "Indonesian", "Arabic", "Hindi",
        ]
        self.inline_source_language.addItems(languages)
        self.inline_target_language.addItems(languages[1:])
        self.inline_source_language.setCurrentText(self.settings.source_language)
        self.inline_target_language.setCurrentText(self.settings.target_language)
        self.inline_cross_check = QCheckBox(self._t("inline_translator.cross_check"))
        controls_layout.addWidget(self.inline_translator_enabled)
        inline_source_label = QLabel(self._t("inline_translator.source_lang"))
        inline_target_label = QLabel(self._t("inline_translator.target_lang"))
        self._i18n_inline_labels["inline_translator.source_lang"] = inline_source_label
        self._i18n_inline_labels["inline_translator.target_lang"] = inline_target_label
        controls_layout.addWidget(inline_source_label)
        controls_layout.addWidget(self.inline_source_language)
        controls_layout.addWidget(inline_target_label)
        controls_layout.addWidget(self.inline_target_language)
        controls_layout.addWidget(self.inline_cross_check)
        controls_layout.addStretch(1)
        layout.addWidget(controls)

        self.inline_body = QWidget()
        body_layout = QVBoxLayout(self.inline_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        self.inline_input = QLineEdit()
        self.inline_input.setPlaceholderText(self._t("inline_translator.placeholder"))
        self.inline_input.textChanged.connect(self._schedule_inline_translation)
        self.inline_input.returnPressed.connect(self._run_inline_translation)
        self.inline_clear_button = QPushButton(self._t("inline_translator.clear"))
        self.inline_clear_button.clicked.connect(self._clear_inline_translation)
        input_layout.addWidget(self.inline_input, 1)
        input_layout.addWidget(self.inline_clear_button)
        body_layout.addWidget(input_row)

        translated_row = QWidget()
        translated_layout = QHBoxLayout(translated_row)
        translated_layout.setContentsMargins(0, 0, 0, 0)
        translated_label = QLabel(self._t("inline_translator.translated"))
        translated_label.setFixedWidth(88)
        self._i18n_inline_labels["inline_translator.translated"] = translated_label
        self.inline_translated_text = QLineEdit()
        self.inline_translated_text.setReadOnly(True)
        self.inline_copy_button = QPushButton(self._t("inline_translator.copy"))
        self.inline_copy_button.clicked.connect(self._copy_inline_translation)
        translated_layout.addWidget(translated_label)
        translated_layout.addWidget(self.inline_translated_text, 1)
        translated_layout.addWidget(self.inline_copy_button)
        body_layout.addWidget(translated_row)

        verified_row = QWidget()
        verified_layout = QHBoxLayout(verified_row)
        verified_layout.setContentsMargins(0, 0, 0, 0)
        verified_label = QLabel(self._t("inline_translator.cross_check"))
        verified_label.setFixedWidth(88)
        self._i18n_inline_labels["inline_translator.cross_check_label"] = verified_label
        self.inline_verified_text = QLineEdit()
        self.inline_verified_text.setReadOnly(True)
        verified_layout.addWidget(verified_label)
        verified_layout.addWidget(self.inline_verified_text, 1)
        body_layout.addWidget(verified_row)
        layout.addWidget(self.inline_body)
        self.inline_body.setVisible(False)
        self.inline_translation_timer = QTimer(self)
        self.inline_translation_timer.setSingleShot(True)
        self.inline_translation_timer.setInterval(1500)
        self.inline_translation_timer.timeout.connect(self._run_inline_translation)
        self.inline_source_language.currentTextChanged.connect(self._schedule_inline_translation)
        self.inline_target_language.currentTextChanged.connect(self._schedule_inline_translation)
        self.inline_cross_check.toggled.connect(self._schedule_inline_translation)
        self._inline_request_id = 0
        self._inline_last_translated = ""
        return box

    def _build_style_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.style_controls: dict[
            SourceKind, tuple[SearchableFontCombo, QSpinBox, QLineEdit, QLineEdit, QSpinBox]
        ] = {}
        style_keys = {
            SourceKind.INPUT: "style.input_caption",
            SourceKind.OUTPUT: "style.output_caption",
            SourceKind.SCREEN: "style.screen_caption",
        }
        for source, style in (
            (SourceKind.INPUT, self.settings.input_style),
            (SourceKind.OUTPUT, self.settings.output_style),
            (SourceKind.SCREEN, self.settings.screen_style),
        ):
            box = self._make_group(style_keys[source])
            form = QFormLayout(box)
            font = SearchableFontCombo(
                style.font_family,
                placeholder=self._t("style.font_search"),
            )
            size = QSpinBox()
            size.setRange(12, 120)
            size.setValue(style.font_size)
            color = QLineEdit(style.color)
            outline_color = QLineEdit(style.outline_color)
            outline = QSpinBox()
            outline.setRange(0, 12)
            outline.setValue(style.outline_width)
            form.addRow(self._form_label("style.font"), font)
            form.addRow(self._form_label("style.size"), size)
            form.addRow(self._form_label("style.color"), color)
            form.addRow(self._form_label("style.outline_color"), outline_color)
            form.addRow(self._form_label("style.outline_width"), outline)
            layout.addWidget(box)
            self.style_controls[source] = (font, size, color, outline_color, outline)
        self.apply_styles_button = QPushButton(self._t("btn.apply_styles"))
        self.apply_styles_button.clicked.connect(self._apply_styles)
        layout.addWidget(self.apply_styles_button)
        return page

    def _build_model_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)

        self.runtime_dependency_panel = self._build_runtime_dependency_panel()
        self.runtime_dependency_group = self._section(
            "section.runtime_libs",
            self.runtime_dependency_panel,
        )
        layout.addWidget(self.runtime_dependency_group)

        model_page = QWidget()
        model_page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        form = QFormLayout(model_page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.asr_device = self._build_compute_combo(self.settings.asr_device)
        self.translation_device = self._build_compute_combo(self.settings.translation_device)
        self.asr_model = NoWheelComboBox()
        self.asr_model.setEditable(True)
        self.translation_model = NoWheelComboBox()
        self.translation_model.setEditable(True)
        for combo in (self.asr_model, self.translation_model):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(24)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            line_edit = combo.lineEdit()
            if line_edit is not None:
                line_edit.setSizePolicy(
                    QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
                )
        self.download_asr_model = self.asr_model
        self.download_translation_model = self.translation_model
        self.asr_model.currentTextChanged.connect(self._update_download_buttons)
        self.translation_model.currentTextChanged.connect(self._update_download_buttons)
        self.asr_model.currentIndexChanged.connect(self._update_model_vram_hints)
        self.translation_model.currentIndexChanged.connect(self._update_model_vram_hints)
        self.model_preset = NoWheelComboBox()
        for label, preset_id in self._model_presets():
            self.model_preset.addItem(label, preset_id)
        self.model_preset.currentIndexChanged.connect(self._model_preset_changed)
        form.addRow(self._form_label("model.recommended_preset"), self.model_preset)
        self.asr_device_hint = self._build_device_hint_label("stt")
        self.translation_device_hint = self._build_device_hint_label("llm")
        form.addRow(self._form_label("model.stt_device"), self._wrap_device_field(self.asr_device, self.asr_device_hint))
        form.addRow(self._form_label("model.llm_device"), self._wrap_device_field(self.translation_device, self.translation_device_hint))
        self.asr_cpu_controls = self._build_cpu_profile_row(
            "asr", self.settings.asr_cpu_threads, self.settings.asr_cpu_core_ids
        )
        self.translation_cpu_controls = self._build_cpu_profile_row(
            "translation",
            self.settings.translation_cpu_threads,
            self.settings.translation_cpu_core_ids,
        )
        self.asr_device.currentIndexChanged.connect(self._refresh_compute_control_visibility)
        self.translation_device.currentIndexChanged.connect(self._refresh_compute_control_visibility)
        form.addRow(self._form_label("model.stt_cpu_threads"), self.asr_cpu_controls)
        form.addRow(self._form_label("model.llm_cpu_threads"), self.translation_cpu_controls)
        self.asr_download = QPushButton(self._t("btn.download"))
        self.translation_download = QPushButton(self._t("btn.download"))
        self.asr_download.clicked.connect(partial(self._download_model, "asr"))
        self.translation_download.clicked.connect(partial(self._download_model, "translation"))
        self.asr_vram_hint = self._build_vram_hint_label()
        self.translation_vram_hint = self._build_vram_hint_label()
        form.addRow(self._form_label("model.stt_model"), self._model_select_row(self.asr_model, self.asr_download))
        form.addRow("", self.asr_vram_hint)
        form.addRow(self._form_label("model.translation_model"), self._model_select_row(self.translation_model, self.translation_download))
        form.addRow("", self.translation_vram_hint)
        refresh_models = QPushButton(self._t("model.refresh"))
        refresh_models.clicked.connect(self._refresh_model_choices)
        form.addRow(refresh_models)
        run_selected = QPushButton(self._t("model.save_settings"))
        run_selected.clicked.connect(self._apply_selected_models)
        form.addRow(run_selected)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_speed = QLabel(self._t("model.waiting"))
        self.download_speed.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.download_speed.setMaximumHeight(56)
        self.download_status = QLabel(self._t("model.select_hint"))
        self.download_status.setWordWrap(True)
        self.download_status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.download_status.setMaximumHeight(72)
        form.addRow(self._form_label("model.progress"), self.download_progress)
        form.addRow(self._form_label("model.progress_speed"), self.download_speed)
        form.addRow(self._form_label("model.status"), self.download_status)
        self.ocr_auto_refresh = QCheckBox(self._t("model.ocr_auto_refresh"))
        self.ocr_auto_refresh.setChecked(self.settings.ocr_auto_refresh)
        self.ocr_auto_refresh.toggled.connect(self._save_ocr_refresh_settings)
        self.ocr_interval = QDoubleSpinBox()
        self.ocr_interval.setRange(0.5, 60.0)
        self.ocr_interval.setDecimals(1)
        self.ocr_interval.setSingleStep(0.5)
        self.ocr_interval.setSuffix(self._t("model.seconds_suffix"))
        self.ocr_interval.setValue(self.settings.ocr_interval)
        self.ocr_interval.setEnabled(self.settings.ocr_auto_refresh)
        self.ocr_interval.valueChanged.connect(self._save_ocr_refresh_settings)
        form.addRow(self.ocr_auto_refresh)
        form.addRow(self._form_label("model.ocr_interval"), self.ocr_interval)
        layout.addWidget(self._section("section.model_download", model_page))

        delete_page = QWidget()
        delete_form = QFormLayout(delete_page)
        self.delete_kind = NoWheelComboBox()
        self.delete_kind.addItem(self._t("model.delete_stt"), "asr")
        self.delete_kind.addItem(self._t("model.delete_translation"), "translation")
        self.delete_kind.currentIndexChanged.connect(self._refresh_delete_choices)
        self.delete_model_combo = NoWheelComboBox()
        delete_button = QPushButton(self._t("model.delete_selected"))
        delete_button.clicked.connect(self._delete_selected_model)
        self.delete_cannot_delete_label = QLabel(self._t("model.cannot_delete_active"))
        delete_form.addRow(self._form_label("model.delete_kind"), self.delete_kind)
        delete_form.addRow(self._form_label("model.downloaded_models"), self.delete_model_combo)
        delete_form.addRow(delete_button)
        delete_form.addRow(self.delete_cannot_delete_label)
        layout.addWidget(self._section("section.model_delete", delete_page))

        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.model_tab_scroll = scroll
        self.model_tab_content = content
        self.model_tab_page = page
        self._refresh_model_choices()
        self._sync_model_preset_selection()
        self._ensure_preset_device_defaults()
        self._refresh_compute_control_visibility()
        self._update_model_vram_hints()
        return page

    def _build_runtime_dependency_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.runtime_install_order_hint = QLabel(self._t("runtime.install_order_hint"))
        self.runtime_install_order_hint.setWordWrap(False)
        self.runtime_install_order_hint.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addWidget(self.runtime_install_order_hint, 1)
        self.runtime_collapse_button = QPushButton("?묎린")
        self.runtime_collapse_button.setFixedWidth(72)
        self.runtime_collapse_button.clicked.connect(self._toggle_runtime_dependency_panel)
        self.runtime_collapse_button.setVisible(False)
        action_row.addWidget(self.runtime_collapse_button)
        self.runtime_install_action_button = QPushButton(self._t("runtime.install_all"))
        self.runtime_install_action_button.setFixedWidth(120)
        self.runtime_install_action_button.clicked.connect(self._handle_runtime_install_action)
        action_row.addWidget(self.runtime_install_action_button)
        layout.addLayout(action_row)
        self.runtime_dependency_body = QWidget()
        body_layout = QVBoxLayout(self.runtime_dependency_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(2)
        self.runtime_dependency_labels: dict[str, QLabel] = {}
        self.runtime_dependency_action_buttons: dict[str, QPushButton] = {}
        for dependency in runtime_dependencies_in_install_order():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            name = QLabel(f"[{dependency.name}]")
            name.setFixedWidth(95)
            description = QLabel(self.tr.runtime_description(dependency.name, dependency.description))
            description.setWordWrap(False)
            description.setToolTip(description.text())
            description.setMinimumWidth(0)
            description.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._runtime_desc_labels[dependency.name] = description
            status = QLabel()
            status.setFixedWidth(72)
            status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            status.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            installed = runtime_dependency_installed(dependency)
            status.setText(
                self._t("runtime.installed")
                if installed
                else self._t("runtime.optional")
                if dependency.optional
                else self._t("runtime.required")
            )
            self.runtime_dependency_labels[dependency.name] = status
            row_layout.addWidget(name)
            row_layout.addWidget(description, 1)
            row_layout.addWidget(status)
            if dependency.optional:
                action = QPushButton(self._t("runtime.install_optional"))
                action.setFixedWidth(72)
                action.clicked.connect(partial(self._install_optional_runtime_dependency, dependency.name))
                self.runtime_dependency_action_buttons[dependency.name] = action
                row_layout.addWidget(action)
            row.setFixedHeight(26)
            row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            body_layout.addWidget(row)
        self.runtime_driver_warning = QLabel()
        self.runtime_driver_warning.setWordWrap(True)
        self.runtime_driver_warning.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        body_layout.addWidget(self.runtime_driver_warning)
        layout.addWidget(self.runtime_dependency_body)
        self._refresh_runtime_driver_warning()
        return panel

    def _toggle_runtime_dependency_panel(self) -> None:
        self._runtime_panel_collapsed = not self._runtime_panel_collapsed
        self._apply_runtime_dependency_panel_visibility()

    def _apply_runtime_dependency_panel_visibility(self) -> None:
        if not hasattr(self, "runtime_dependency_body"):
            return
        can_collapse = (
            all_runtime_dependencies_installed()
            and not self._runtime_install_active
            and not self._runtime_downloads
        )
        if hasattr(self, "runtime_collapse_button"):
            self.runtime_collapse_button.setVisible(can_collapse)
            self.runtime_collapse_button.setText(
                "펼치기" if self._runtime_panel_collapsed else "접기"
            )
        if not can_collapse:
            self._runtime_panel_collapsed = False
        self.runtime_dependency_body.setVisible(not self._runtime_panel_collapsed)

    def _refresh_runtime_driver_warning(self) -> None:
        if not hasattr(self, "runtime_driver_warning"):
            return
        message = translate_driver_message(self.tr, runtime_driver_status_message())
        compatible, _ = check_cuda_driver_compatibility()
        self.runtime_driver_warning.setText(message)
        if compatible:
            self.runtime_driver_warning.setStyleSheet("color: #1b5e20;")
        else:
            self.runtime_driver_warning.setStyleSheet("color: #b71c1c;")

    def _set_runtime_status_label(self, label: QLabel, text: str) -> None:
        label.setToolTip(text)
        label.setText(
            label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, label.width())
        )

    def _set_download_status_text(self, text: str) -> None:
        self.download_status.setToolTip(text)
        available = max(120, self.download_status.width() - 8)
        self.download_status.setText(
            self.download_status.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, available
            )
        )

    def _set_download_speed_text(self, text: str) -> None:
        self.download_speed.setToolTip(text)
        available = max(120, self.download_speed.width() - 8)
        self.download_speed.setText(
            self.download_speed.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, available
            )
        )

    def _clamp_model_tab_width(self) -> None:
        if not hasattr(self, "model_tab_scroll") or not hasattr(self, "model_tab_content"):
            return
        viewport = self.model_tab_scroll.viewport()
        if viewport is None:
            return
        width = viewport.width()
        if width > 0:
            self.model_tab_content.setMinimumWidth(width)
            self.model_tab_content.setMaximumWidth(16777215)

    def _apply_initial_game_light_preset(self) -> None:
        if not hasattr(self, "model_preset"):
            return
        self.model_preset.blockSignals(True)
        self._apply_model_preset("game_light")
        self.settings.model_preset = "game_light"
        self.model_preset.blockSignals(False)
        save_settings(self.settings)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._clamp_model_tab_width()

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._clamp_model_tab_width)

    def _runtime_install_position(self) -> tuple[int, int]:
        if not self._runtime_install_active:
            return (0, 0)
        current = self._runtime_install_batch_total - len(self._runtime_install_queue)
        return (current, self._runtime_install_batch_total)

    def _refresh_runtime_dependency_panel(self) -> None:
        if not hasattr(self, "runtime_dependency_labels"):
            return
        self._refresh_runtime_driver_warning()
        if hasattr(self, "runtime_install_order_hint"):
            self.runtime_install_order_hint.setText(self._t("runtime.install_order_hint"))
        installing = bool(self._runtime_install_active or self._runtime_downloads)
        pending = runtime_dependencies_pending_install()
        for dependency in runtime_dependencies_in_install_order():
            status = self.runtime_dependency_labels.get(dependency.name)
            if status is None:
                continue
            installed = runtime_dependency_installed(dependency)
            downloading = dependency.name in self._runtime_downloads
            queued = any(name == dependency.name for name, _ in self._runtime_install_queue)
            if installed:
                status_text = self._t("runtime.installed")
            elif downloading:
                status_text = self._t("runtime.installing")
            elif queued:
                status_text = self._t("runtime.queued")
            elif dependency.optional:
                status_text = self._t("runtime.optional")
            else:
                status_text = self._t("runtime.required")
            self._set_runtime_status_label(status, status_text)
            action_button = self.runtime_dependency_action_buttons.get(dependency.name)
            if action_button is not None:
                action_button.setText(self._t("runtime.install_optional"))
                action_button.setEnabled(
                    not installed
                    and not installing
                    and runtime_install_available(dependency.name)
                )
            desc_label = self._runtime_desc_labels.get(dependency.name)
            if desc_label is not None:
                description = self.tr.runtime_description(dependency.name, dependency.description)
                desc_label.setText(
                    desc_label.fontMetrics().elidedText(
                        description,
                        Qt.TextElideMode.ElideRight,
                        max(160, desc_label.width() - 8),
                    )
                )
                desc_label.setToolTip(description)
        if hasattr(self, "runtime_install_action_button"):
            if installing:
                self.runtime_install_action_button.setText(self._t("runtime.cancel"))
                self.runtime_install_action_button.setEnabled(True)
            elif all_runtime_dependencies_installed():
                self.runtime_install_action_button.setText(self._t("runtime.reinstall_all"))
                self.runtime_install_action_button.setEnabled(True)
            elif pending:
                self.runtime_install_action_button.setText(self._t("runtime.install_all"))
                self.runtime_install_action_button.setEnabled(True)
            else:
                self.runtime_install_action_button.setText(self._t("runtime.install_all"))
                self.runtime_install_action_button.setEnabled(False)
        self._apply_runtime_dependency_panel_visibility()

    def _section(self, key: str, content: QWidget) -> QGroupBox:
        box = self._make_group(key)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(box)
        layout.addWidget(content)
        return box

    def _build_device_hint_label(self, kind: str) -> QLabel:
        label = QLabel(self._t(f"compute.hint.{kind}"))
        label.setWordWrap(True)
        label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    @staticmethod
    def _wrap_device_field(combo: QComboBox, hint: QLabel) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(combo)
        layout.addWidget(hint)
        return row

    def _build_vram_hint_label(self) -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setStyleSheet("color: #9aa7c2; font-size: 11px;")
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    def _lookup_model_vram_gb(self, model_id: str) -> float | None:
        normalized = model_id.replace("\\", "/").strip()
        if normalized in MODEL_VRAM_HINTS_GB:
            return MODEL_VRAM_HINTS_GB[normalized]
        for key, value in MODEL_VRAM_HINTS_GB.items():
            if key.split("/")[-1].lower() in normalized.lower():
                return value
        return None

    def _model_vram_hint_text(self, model_id: str) -> str:
        vram = self._lookup_model_vram_gb(model_id)
        if vram is None:
            return ""
        if vram <= 0:
            return self._t("model.vram_cpu")
        gb = int(vram) if vram.is_integer() else vram
        return self._t("model.vram_estimate", gb=gb)

    def _selected_model_id(self, combo: QComboBox) -> str:
        return str(combo.currentData() or combo.currentText() or "").strip()

    @staticmethod
    def _model_compute_tag_key(kind: str, model_id: str) -> str:
        normalized = model_id.replace("\\", "/").lower()
        if "sense-voice" in normalized or "sensevoice" in normalized:
            return "model.compute.cpu_only"
        if kind == "asr":
            return "model.compute.cpu_gpu"
        if "hy-mt2" in normalized and ("2bit" in normalized or "1.25bit" in normalized):
            return "model.compute.cpu_only"
        if "awq" in normalized:
            return "model.compute.gpu_only"
        return "model.compute.cpu_gpu"

    def _model_combo_label(self, kind: str, model_id: str, *, downloaded: bool = False) -> str:
        label = model_id + self._t(self._model_compute_tag_key(kind, model_id))
        note = MODEL_LABEL_NOTES.get(model_id.replace("\\", "/").strip())
        if note:
            label += f" - {note}"
        if downloaded:
            label += self._t("caption.downloaded_tag")
        return label

    def _update_model_vram_hints(self, *_args: object) -> None:
        if not hasattr(self, "asr_vram_hint"):
            return
        self.asr_vram_hint.setText(self._model_vram_hint_text(self._selected_model_id(self.asr_model)))
        self.translation_vram_hint.setText(
            self._model_vram_hint_text(self._selected_model_id(self.translation_model))
        )

    def _ensure_preset_device_defaults(self) -> None:
        if not hasattr(self, "asr_device"):
            return
        if self._resolve_model_preset() != "game_light":
            return
        self._select_compute_device(self.asr_device, "cpu")
        self._select_compute_device(self.translation_device, self._first_gpu_device())

    def _collapsible_section(
        self,
        key: str,
        content: QWidget,
        *,
        expanded: bool = False,
    ) -> CollapsibleSection:
        section = CollapsibleSection(self._t(key), content, expanded=expanded)
        self._i18n_collapsible_sections[key] = section
        return section

    @staticmethod
    def _model_select_row(combo: QComboBox, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(combo, 1)
        layout.addWidget(button)
        return row

    def _model_presets(self) -> list[tuple[str, str]]:
        return [
            (self._t("preset.game_light"), "game_light"),
            (self._t("preset.qwen_light"), "qwen_light"),
            (self._t("preset.qwen_high"), "qwen_high"),
        ]

    def _model_preset_changed(self, *_args: object) -> None:
        if not hasattr(self, "model_preset"):
            return
        preset_id = str(self.model_preset.currentData() or "")
        if not preset_id:
            return
        self._apply_model_preset(preset_id)
        self.settings.model_preset = preset_id

    def _apply_model_preset(self, preset_id: str) -> None:
        gpu_device = self._first_gpu_device()
        if preset_id == "game_light":
            self._set_combo_value(self.asr_model, SENSEVOICE_ASR_MODEL)
            self._set_combo_value(self.translation_model, "tencent/Hy-MT2-1.8B-GGUF")
            self._select_compute_device(self.asr_device, "cpu")
            self._select_compute_device(self.translation_device, gpu_device)
            self._set_cpu_thread_profile("asr", 2, "")
            self._set_cpu_thread_profile("translation", 0, "")
        elif preset_id == "qwen_light":
            self._set_combo_value(self.asr_model, "Qwen/Qwen3-ASR-0.6B")
            self._set_combo_value(self.translation_model, "Qwen/Qwen3-4B")
            self._select_compute_device(self.asr_device, gpu_device)
            self._select_compute_device(self.translation_device, gpu_device)
            self._set_cpu_thread_profile("asr", 0, "")
            self._set_cpu_thread_profile("translation", 0, "")
        elif preset_id == "qwen_high":
            self._set_combo_value(self.asr_model, "Qwen/Qwen3-ASR-1.7B")
            self._set_combo_value(self.translation_model, "Qwen/Qwen3-8B-AWQ")
            self._select_compute_device(self.asr_device, gpu_device)
            self._select_compute_device(self.translation_device, gpu_device)
            self._set_cpu_thread_profile("asr", 0, "")
            self._set_cpu_thread_profile("translation", 0, "")
        self._refresh_compute_control_visibility()
        self._update_download_buttons()

    def _resolve_model_preset(self) -> str:
        matched = self._matching_model_preset()
        if matched:
            return matched
        stored = str(self.settings.model_preset or "").strip()
        if stored in {"game_light", "qwen_light", "qwen_high"}:
            return stored
        return "game_light"

    def _sync_model_preset_selection(self) -> None:
        if not hasattr(self, "model_preset"):
            return
        preset_id = self._resolve_model_preset()
        index = self.model_preset.findData(preset_id)
        if index < 0:
            index = 0
        self.model_preset.blockSignals(True)
        self.model_preset.setCurrentIndex(index)
        self.model_preset.blockSignals(False)
        matched = self._matching_model_preset()
        if matched:
            self.settings.model_preset = matched

    def _matching_model_preset(self) -> str:
        asr = self.settings.asr_model.replace("\\", "/")
        translation = self.settings.translation_model.replace("\\", "/")
        if "sense-voice" in asr.lower() or "sensevoice" in asr.lower():
            if "hy-mt2" in translation.lower():
                return "game_light"
        if "Qwen3-ASR-1.7B" in asr and "Qwen3-8B" in translation:
            return "qwen_high"
        if "Qwen3-ASR-0.6B" in asr and "Qwen3-4B" in translation:
            return "qwen_light"
        return ""

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(value)

    def _set_cpu_thread_profile(self, prefix: str, threads: int, core_ids: str) -> None:
        combo: QComboBox = getattr(self, f"{prefix}_cpu_threads")
        index = combo.findData(threads)
        combo.setCurrentIndex(max(0, index))
        setattr(self.settings, f"{prefix}_cpu_core_ids", core_ids)
        getattr(self, f"{prefix}_cpu_core_summary").setText(format_cpu_core_ids(core_ids))
        self._cpu_profile_changed(prefix)

    def _first_gpu_device(self) -> str:
        for index in range(self.translation_device.count()):
            value = str(self.translation_device.itemData(index) or "")
            if value.startswith("cuda:") and self.translation_device.model().item(index).isEnabled():
                return value
        return "cpu"

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._collapsible_section("section.caption_style", self._build_style_tab()))
        layout.addWidget(self._collapsible_section("section.server_remote", self._build_network_tab()))
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return page

    def _build_info_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.info_view = QTextEdit()
        self.info_view.setReadOnly(True)
        self.info_view.setPlainText(load_info_section(self.settings.ui_language))
        layout.addWidget(self.info_view)
        return page

    def _ui_language_changed(self) -> None:
        language = str(self.ui_language_combo.currentData() or "ko")
        if language not in SUPPORTED_LANGUAGES or language == self.settings.ui_language:
            return
        self.settings.ui_language = language
        self.tr.language = language
        save_settings(self.settings)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.tabs.setTabText(0, self._t("tab.monitor"))
        self.tabs.setTabText(1, self._t("tab.models"))
        self.tabs.setTabText(2, self._t("tab.settings"))
        self.tabs.setTabText(3, self._t("tab.info"))
        for key, label in self._i18n_form_labels.items():
            label.setText(self._t(key))
        for key, box in self._i18n_group_boxes.items():
            box.setTitle(self._t(key))
        for key, section in self._i18n_collapsible_sections.items():
            section.set_title(self._t(key))
        if hasattr(self, "apply_styles_button"):
            self.apply_styles_button.setText(self._t("btn.apply_styles"))
        if hasattr(self, "style_controls"):
            for controls in self.style_controls.values():
                controls[0].set_placeholder(self._t("style.font_search"))
        for key, label in self._i18n_inline_labels.items():
            if key == "inline_translator.cross_check_label":
                label.setText(self._t("inline_translator.cross_check"))
            else:
                label.setText(self._t(key))
        self.local_start.setText(self._t("btn.engine_start"))
        self.local_stop.setText(self._t("btn.engine_stop"))
        if self.remote:
            self.client_start.setText(self._t("btn.client_stop"))
        else:
            self.client_start.setText(self._t("btn.client_start"))
        self.input_toggle.setText(self._t("toggle.input_detection"))
        self.output_toggle.setText(self._t("toggle.output_detection"))
        self.refresh_devices_button.setText(self._t("audio.refresh_devices"))
        self.select_ocr_region.setText(self._t("btn.select_screen_region"))
        self.omit_original.setText(self._t("checkbox.omit_original"))
        if self.caption_popup.isVisible():
            self.popup_toggle.setText(self._t("btn.popup_stop"))
        else:
            self.popup_toggle.setText(self._t("btn.popup_start"))
        self.caption_popup.retranslate()
        self.inline_translator_enabled.setText(self._t("inline_translator.enable"))
        self.inline_cross_check.setText(self._t("inline_translator.cross_check"))
        self.inline_input.setPlaceholderText(self._t("inline_translator.placeholder"))
        self.inline_clear_button.setText(self._t("inline_translator.clear"))
        self.inline_copy_button.setText(self._t("inline_translator.copy"))
        self.ocr_auto_refresh.setText(self._t("model.ocr_auto_refresh"))
        self.ocr_interval.setSuffix(self._t("model.seconds_suffix"))
        self.delete_kind.setItemText(0, self._t("model.delete_stt"))
        self.delete_kind.setItemText(1, self._t("model.delete_translation"))
        self.delete_cannot_delete_label.setText(self._t("model.cannot_delete_active"))
        if self.server:
            self.host_toggle.setText(self._t("network.host_stop"))
        else:
            self.host_toggle.setText(self._t("network.host_start"))
        if self.remote:
            self.client_toggle.setText(self._t("network.client_disconnect"))
        else:
            self.client_toggle.setText(self._t("network.client_connect"))
        if hasattr(self, "whitelist_enabled"):
            self.whitelist_enabled.setText(self._t("network.whitelist_enable"))
            self.whitelist_recommend_label.setText(self._t("network.whitelist_recommend"))
            self.whitelist_add_button.setText(self._t("network.whitelist_add"))
            self.whitelist_ip_input.setPlaceholderText(self._t("network.whitelist_add_placeholder"))
            self.network_help_label.setText(self._t("network.help"))
            self.copy_overlay_button.setText(self._t("network.copy_url"))
        self.ui_language_combo.blockSignals(True)
        for index in range(self.ui_language_combo.count()):
            code = str(self.ui_language_combo.itemData(index) or "")
            if code in SUPPORTED_LANGUAGES:
                self.ui_language_combo.setItemText(index, self._t(f"lang.{code}"))
        self.ui_language_combo.blockSignals(False)
        self.model_preset.blockSignals(True)
        self.model_preset.clear()
        for label, preset in self._model_presets():
            self.model_preset.addItem(label, preset)
        self.model_preset.blockSignals(False)
        self._sync_model_preset_selection()
        for prefix in ("asr", "translation"):
            combo: QComboBox = getattr(self, f"{prefix}_cpu_threads")
            current = int(combo.currentData() or 0)
            total = len(list_cpu_cores())
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(self._t("cpu.auto"), 0)
            for count in range(1, total + 1):
                suffix = self._t("cpu.count_suffix")
                combo.addItem(f"{count}{suffix}" if suffix else str(count), count)
            combo.addItem(self._t("cpu.custom"), -1)
            index = combo.findData(current)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
            getattr(self, f"{prefix}_cpu_core_button").setText(self._t("cpu.select_cores"))
        if hasattr(self, "info_view"):
            self.info_view.setPlainText(load_info_section(self.settings.ui_language))
        if hasattr(self, "pipeline"):
            self.pipeline.tr.language = self.settings.ui_language
        if hasattr(self, "asr_device_hint"):
            self.asr_device_hint.setText(self._t("compute.hint.stt"))
            self.translation_device_hint.setText(self._t("compute.hint.llm"))
            self._update_model_vram_hints()
        self._refresh_runtime_dependency_panel()
        self._refresh_ocr_controls()
        self._refresh_dashboard()
        self._update_download_buttons()
        self._render_unified_captions()
        self._render_status()

    def _build_compute_combo(self, selected: str) -> QComboBox:
        combo = NoWheelComboBox()
        for compute_device in list_compute_devices():
            combo.addItem(compute_device.label, compute_device.value)
            if not compute_device.available:
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        index = combo.findData(normalize_device(selected))
        combo.setCurrentIndex(max(0, index))
        return combo

    def _build_cpu_profile_row(self, prefix: str, selected_threads: int, core_ids: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = NoWheelComboBox()
        total = len(list_cpu_cores())
        combo.addItem(self._t("cpu.auto"), 0)
        for count in range(1, total + 1):
            suffix = self._t("cpu.count_suffix")
            combo.addItem(f"{count}{suffix}" if suffix else str(count), count)
        combo.addItem(self._t("cpu.custom"), -1)
        index = combo.findData(selected_threads)
        combo.setCurrentIndex(max(0, index))
        button = QPushButton(self._t("cpu.select_cores"))
        summary = QLabel(format_cpu_core_ids(core_ids))
        summary.setMinimumWidth(120)
        button.clicked.connect(partial(self._edit_cpu_cores, prefix))
        combo.currentIndexChanged.connect(partial(self._cpu_profile_changed, prefix))
        layout.addWidget(combo)
        layout.addWidget(button)
        layout.addWidget(summary, 1)
        setattr(self, f"{prefix}_cpu_threads", combo)
        setattr(self, f"{prefix}_cpu_core_button", button)
        setattr(self, f"{prefix}_cpu_core_summary", summary)
        return row

    def _cpu_profile_changed(self, prefix: str, *_args: object) -> None:
        combo: QComboBox = getattr(self, f"{prefix}_cpu_threads")
        custom = int(combo.currentData() or 0) < 0
        getattr(self, f"{prefix}_cpu_core_button").setEnabled(custom)
        if not custom:
            setattr(self.settings, f"{prefix}_cpu_core_ids", "")
            getattr(self, f"{prefix}_cpu_core_summary").setText(self._t("cpu.none_selected"))

    def _edit_cpu_cores(self, prefix: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("cpu.dialog_title"))
        layout = QVBoxLayout(dialog)
        current_ids = set(
            int(value)
            for value in getattr(self.settings, f"{prefix}_cpu_core_ids").split(",")
            if value.strip().isdigit()
        )
        checks: list[tuple[int, QCheckBox]] = []
        for core in list_cpu_cores():
            check = QCheckBox(core.label)
            check.setChecked(core.index in current_ids)
            checks.append((core.index, check))
            layout.addWidget(check)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            core_ids = ",".join(str(index) for index, check in checks if check.isChecked())
            setattr(self.settings, f"{prefix}_cpu_core_ids", core_ids)
            getattr(self, f"{prefix}_cpu_core_summary").setText(format_cpu_core_ids(core_ids))

    def _refresh_compute_control_visibility(self, *_args: object) -> None:
        for prefix, device_combo in (
            ("asr", self.asr_device),
            ("translation", self.translation_device),
        ):
            is_cpu = normalize_device(str(device_combo.currentData() or "cpu")) == "cpu"
            controls = getattr(self, f"{prefix}_cpu_controls")
            controls.setVisible(is_cpu)
            if is_cpu:
                self._cpu_profile_changed(prefix)

    @staticmethod
    def _select_compute_device(combo: QComboBox, device: str) -> None:
        index = combo.findData(device)
        if index < 0 and device.startswith("cuda:"):
            for candidate in range(combo.count()):
                value = str(combo.itemData(candidate) or "")
                item = combo.model().item(candidate)
                if value.startswith("cuda:") and item is not None and item.isEnabled():
                    index = candidate
                    break
        if index < 0:
            index = combo.findData("cpu")
        combo.setCurrentIndex(max(0, index))

    def _build_network_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(self.settings.port)
        self.password = QLineEdit(self.settings.password)
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.remote_host = QLineEdit(self.settings.remote_host)
        self.remote_port = QSpinBox()
        self.remote_port.setRange(1024, 65535)
        self.remote_port.setValue(self.settings.remote_port)
        form.addRow(self._form_label("network.host_port"), self.port)
        form.addRow(self._form_label("network.password"), self.password)

        self.whitelist_enabled = QCheckBox(self._t("network.whitelist_enable"))
        self.whitelist_enabled.setChecked(self.settings.client_ip_whitelist_enabled)
        self.whitelist_enabled.toggled.connect(self._whitelist_mode_changed)
        form.addRow(self.whitelist_enabled)

        self.whitelist_recommend_label = QLabel(self._t("network.whitelist_recommend"))
        self.whitelist_recommend_label.setWordWrap(True)
        self.whitelist_recommend_label.setStyleSheet("color: #f0c060;")
        form.addRow(self.whitelist_recommend_label)

        self._whitelist_checkboxes: dict[str, QCheckBox] = {}
        self.whitelist_ip_input = QLineEdit()
        self.whitelist_ip_input.setPlaceholderText(self._t("network.whitelist_add_placeholder"))
        self.whitelist_add_button = QPushButton(self._t("network.whitelist_add"))
        self.whitelist_add_button.clicked.connect(self._add_whitelist_ip)
        whitelist_add_row = QWidget()
        whitelist_add_layout = QHBoxLayout(whitelist_add_row)
        whitelist_add_layout.setContentsMargins(0, 0, 0, 0)
        whitelist_add_layout.addWidget(self.whitelist_ip_input, 1)
        whitelist_add_layout.addWidget(self.whitelist_add_button)

        self.whitelist_ip_container = QWidget()
        self.whitelist_ip_layout = QVBoxLayout(self.whitelist_ip_container)
        self.whitelist_ip_layout.setContentsMargins(0, 0, 0, 0)
        self.whitelist_ip_layout.setSpacing(4)
        whitelist_scroll = QScrollArea()
        whitelist_scroll.setWidgetResizable(True)
        whitelist_scroll.setMinimumHeight(120)
        whitelist_scroll.setWidget(self.whitelist_ip_container)
        whitelist_panel = QWidget()
        whitelist_panel_layout = QVBoxLayout(whitelist_panel)
        whitelist_panel_layout.setContentsMargins(0, 0, 0, 0)
        whitelist_panel_layout.addWidget(whitelist_scroll)
        whitelist_panel_layout.addWidget(whitelist_add_row)
        form.addRow(self._form_label("network.whitelist_ips"), whitelist_panel)
        self._rebuild_whitelist_checkboxes()

        form.addRow(self._form_label("network.client_ip"), self.remote_host)
        form.addRow(self._form_label("network.client_port"), self.remote_port)
        self.host_toggle = QPushButton(self._t("network.host_start"))
        self.host_toggle.clicked.connect(self._toggle_server)
        self.client_toggle = QPushButton(self._t("network.client_connect"))
        self.client_toggle.clicked.connect(self._toggle_remote)
        self.overlay_url = QLineEdit(self._overlay_url())
        self.overlay_url.setReadOnly(True)
        self.copy_overlay_button = QPushButton(self._t("network.copy_url"))
        self.copy_overlay_button.clicked.connect(self._copy_overlay_url)
        overlay_row = QWidget()
        overlay_layout = QHBoxLayout(overlay_row)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.addWidget(self.overlay_url, 1)
        overlay_layout.addWidget(self.copy_overlay_button)
        form.addRow(self.host_toggle)
        form.addRow(self.client_toggle)
        form.addRow(self._form_label("network.obs_url"), overlay_row)

        self.network_help_label = QLabel(self._t("network.help"))
        self.network_help_label.setWordWrap(True)
        self.network_help_label.setStyleSheet("color: #aaaaaa; padding-top: 8px;")
        form.addRow(self.network_help_label)
        self._update_whitelist_panel_visibility()
        return page

    def _whitelist_catalog_ips(self) -> list[str]:
        catalog = parse_ip_list(self.settings.client_ip_whitelist_catalog)
        allowed = parse_ip_list(self.settings.client_ip_whitelist_allowed)
        merged: list[str] = []
        seen: set[str] = set()
        for ip in (*catalog, *allowed, *list_local_ipv4_addresses()):
            if ip not in seen:
                seen.add(ip)
                merged.append(ip)
        return merged

    def _rebuild_whitelist_checkboxes(self) -> None:
        allowed = set(parse_ip_list(self.settings.client_ip_whitelist_allowed))
        while self.whitelist_ip_layout.count():
            item = self.whitelist_ip_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._whitelist_checkboxes.clear()
        ips = self._whitelist_catalog_ips()
        if not ips:
            empty_label = QLabel("-")
            empty_label.setStyleSheet("color: #888888;")
            self.whitelist_ip_layout.addWidget(empty_label)
            self._whitelist_empty_label = empty_label
            return
        self._whitelist_empty_label = None
        for ip in ips:
            checkbox = QCheckBox(ip)
            checkbox.setChecked(ip in allowed)
            self._whitelist_checkboxes[ip] = checkbox
            self.whitelist_ip_layout.addWidget(checkbox)
        self.whitelist_ip_layout.addStretch(1)

    def _whitelist_mode_changed(self, _enabled: bool) -> None:
        self._update_whitelist_panel_visibility()

    def _update_whitelist_panel_visibility(self) -> None:
        enabled = self.whitelist_enabled.isChecked()
        self.whitelist_ip_container.setEnabled(enabled)
        self.whitelist_ip_input.setEnabled(enabled)
        self.whitelist_add_button.setEnabled(enabled)
        self.whitelist_recommend_label.setVisible(not enabled)

    def _add_whitelist_ip(self) -> None:
        import ipaddress

        candidate = self.whitelist_ip_input.text().strip()
        if not candidate:
            return
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            QMessageBox.warning(
                self,
                self._t("network.whitelist_add"),
                self._t("msg.whitelist_ip_invalid"),
            )
            return
        if candidate in self._whitelist_checkboxes:
            QMessageBox.information(
                self,
                self._t("network.whitelist_add"),
                self._t("msg.whitelist_ip_duplicate"),
            )
            return
        catalog = parse_ip_list(self.settings.client_ip_whitelist_catalog)
        if candidate not in catalog:
            catalog.append(candidate)
            self.settings.client_ip_whitelist_catalog = join_ip_list(catalog)
        allowed = parse_ip_list(self.settings.client_ip_whitelist_allowed)
        if candidate not in allowed:
            allowed.append(candidate)
            self.settings.client_ip_whitelist_allowed = join_ip_list(allowed)
        self._rebuild_whitelist_checkboxes()
        self.whitelist_ip_input.clear()
        save_settings(self.settings)

    def _sync_whitelist_settings_from_ui(self) -> None:
        self.settings.client_ip_whitelist_enabled = self.whitelist_enabled.isChecked()
        allowed = [ip for ip, checkbox in self._whitelist_checkboxes.items() if checkbox.isChecked()]
        self.settings.client_ip_whitelist_allowed = join_ip_list(allowed)
        catalog = parse_ip_list(self.settings.client_ip_whitelist_catalog)
        for ip in self._whitelist_checkboxes:
            if ip not in catalog:
                catalog.append(ip)
        self.settings.client_ip_whitelist_catalog = join_ip_list(catalog)

    def _refresh_devices(self) -> None:
        inputs = list_microphones(False)
        outputs = list_microphones(True)
        self.input_device.clear()
        self.output_device.clear()
        for device in inputs:
            self.input_device.addItem(device.name, device.id)
        for device in outputs:
            self.output_device.addItem(device.name, device.id)
        self._restore_device_selection(
            self.input_device, self.settings.input_device_id, self.settings.input_device_name
        )
        self._restore_device_selection(
            self.output_device, self.settings.output_device_id, self.settings.output_device_name
        )
        self._set_status(self._t("msg.devices_found", inputs=len(inputs), outputs=len(outputs)))

    def _set_toggle_checked(self, toggle: QCheckBox, checked: bool) -> None:
        toggle.blockSignals(True)
        toggle.setChecked(checked)
        toggle.blockSignals(False)

    def _toggle_capture(self, source: SourceKind, checked: bool | None = None) -> None:
        toggle = self.input_toggle if source == SourceKind.INPUT else self.output_toggle
        combo = self.input_device if source == SourceKind.INPUT else self.output_device
        if checked is None:
            checked = source not in self.captures
        if not checked:
            if source in self.captures:
                self.captures.pop(source).stop()
            self._set_toggle_checked(toggle, False)
            self.bridge.source_status.emit(source, self._t("source.stopped"))
            self._set_capture_enabled(source, False)
            return
        if source in self.captures:
            self._set_toggle_checked(toggle, True)
            return
        if not self._engine_ready():
            self._set_status(self._t("msg.engine_required"))
            self._set_toggle_checked(toggle, False)
            return
        device_id = combo.currentData()
        if not device_id:
            self._set_status(self._t("msg.no_audio_device"))
            self._set_toggle_checked(toggle, False)
            return
        try:
            capture = AudioCapture(
                device_id,
                source,
                self._process_audio,
                self.bridge.source_status.emit,
                min_speech_seconds=self.settings.min_speech_seconds,
            )
            self.captures[source] = capture
            capture.start()
        except Exception as exc:
            self.captures.pop(source, None)
            self._set_toggle_checked(toggle, False)
            self._set_status(self._t("msg.audio_start_failed", error=exc))
            return
        self._set_toggle_checked(toggle, True)
        self._set_capture_enabled(source, True)

    def _stop_all_detection(self) -> None:
        for source in (SourceKind.INPUT, SourceKind.OUTPUT):
            self._toggle_capture(source, False)
        self._toggle_ocr(False)

    def _korean_ocr_required(self) -> bool:
        return self.settings.source_language.strip().casefold() == "korean"

    def _ocr_cooldown_active(self) -> bool:
        return self._ocr_cooldown_remaining > 0

    def _start_ocr_cooldown(self) -> None:
        self._ocr_cooldown_remaining = OCR_SWITCH_COOLDOWN_SECONDS
        self.ocr_cooldown_timer.start(1000)
        self._update_ocr_cooldown_ui()

    def _tick_ocr_cooldown(self) -> None:
        if self._ocr_cooldown_remaining <= 0:
            self.ocr_cooldown_timer.stop()
            self._refresh_ocr_controls()
            return
        self._ocr_cooldown_remaining -= 1
        if self._ocr_cooldown_remaining <= 0:
            self.ocr_cooldown_timer.stop()
            self._refresh_ocr_controls()
            self._set_status(self._t("status.ready"))
            return
        self._update_ocr_cooldown_ui()

    def _update_ocr_cooldown_ui(self) -> None:
        message = self._ocr_cooldown_message()
        self._set_status(message)
        self.bridge.source_status.emit(SourceKind.SCREEN, message)
        self._refresh_ocr_controls()

    def _ocr_cooldown_message(self) -> str:
        seconds = self._ocr_cooldown_remaining
        if self.settings.ui_language == "en":
            return f"Waiting to switch ({seconds:02d}s)"
        if self.settings.ui_language == "ja":
            return f"切り替え待機中 ({seconds:02d}s)"
        return f"전환 대기중 ({seconds:02d}s)"

    def _ocr_cooldown_tip(self) -> str:
        if self.settings.ui_language == "en":
            return "The OCR engine is being released. Turn it on again after the countdown."
        if self.settings.ui_language == "ja":
            return "OCRエンジンを整理しています。カウント終了後にもう一度有効にしてください。"
        return "OCR 엔진 정리 중입니다. 카운트가 끝난 뒤 다시 켜주세요."

    def _request_korean_ocr_model(self, *, start_ocr_after: bool) -> str:
        if not self._korean_ocr_required() or is_korean_ocr_model_ready():
            return "ready"
        answer = QMessageBox.question(
            self,
            self._t("msg.korean_ocr_download_title"),
            self._t("msg.korean_ocr_download_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._set_status(self._t("msg.korean_ocr_download_declined"))
            return "declined"
        self._start_korean_ocr_download(start_ocr_after=start_ocr_after)
        if is_korean_ocr_model_ready():
            return "ready"
        return "downloading"

    def _start_korean_ocr_download(self, *, start_ocr_after: bool) -> None:
        self._pending_ocr_after_korean_download = start_ocr_after
        if is_korean_ocr_model_ready():
            if start_ocr_after:
                self._enable_ocr_after_korean_model()
            return
        download_key = ("ocr", KOREAN_OCR_MODEL_ID)
        if download_key == self._current_model_download or download_key in self._download_queue:
            return
        if self._current_model_download is not None:
            self._download_queue.append(download_key)
            self._active_model_downloads.add(download_key)
            self._download_cancel_events[download_key] = threading.Event()
            self.download_status.setText(self._t("msg.korean_ocr_downloading"))
            self._set_status(self._t("msg.korean_ocr_downloading"))
            return
        self._start_model_download("ocr", KOREAN_OCR_MODEL_ID)

    def _enable_ocr_after_korean_model(self) -> None:
        if not self._engine_ready():
            self._set_status(self._t("msg.engine_required"))
            self._set_toggle_checked(self.ocr_toggle, False)
            return
        self.ocr.set_source_language(self.settings.source_language)
        try:
            self.ocr.stop()
            if self.ocr.is_running():
                raise RuntimeError("이전 OCR 엔진이 아직 종료 중입니다. 잠시 후 다시 시도해주세요.")
            self._emit_ocr_status(OCR_STATUS_LOADING)
            self.ocr.start()
        except Exception as exc:
            self.settings.ocr_enabled = False
            self._set_toggle_checked(self.ocr_toggle, False)
            self._set_status(self._t("msg.ocr_start_failed", error=exc))
            return
        self.settings.ocr_enabled = True
        self._set_toggle_checked(self.ocr_toggle, True)
        save_settings(self.settings)

    def _restart_ocr_engine(self) -> None:
        if not self.settings.ocr_enabled:
            self.ocr.set_source_language(self.settings.source_language)
            return
        self.ocr.stop()
        if self.ocr.is_running():
            self.settings.ocr_enabled = False
            self._set_toggle_checked(self.ocr_toggle, False)
            self._set_status(
                self._t(
                    "msg.ocr_start_failed",
                    error="이전 OCR 엔진이 아직 종료 중입니다. 잠시 후 다시 시도해주세요.",
                )
            )
            save_settings(self.settings)
            return
        self.ocr.set_source_language(self.settings.source_language)
        self.settings.ocr_enabled = False
        self._set_toggle_checked(self.ocr_toggle, False)
        self.bridge.source_status.emit(SourceKind.SCREEN, self._t("source.stopped"))
        self._start_ocr_cooldown()
        save_settings(self.settings)

    def _toggle_ocr(self, checked: bool | None = None) -> None:
        if checked is None:
            checked = not self.settings.ocr_enabled
        if not checked:
            self.ocr.stop()
            self.settings.ocr_enabled = False
            self._set_toggle_checked(self.ocr_toggle, False)
            self.bridge.source_status.emit(SourceKind.SCREEN, self._t("source.stopped"))
            self._start_ocr_cooldown()
            save_settings(self.settings)
            return
        if self._ocr_cooldown_active():
            self._set_toggle_checked(self.ocr_toggle, False)
            self._update_ocr_cooldown_ui()
            return
        if self.settings.ocr_enabled:
            self._set_toggle_checked(self.ocr_toggle, True)
            return
        if not self._engine_ready():
            self._set_status(self._t("msg.engine_required"))
            self._set_toggle_checked(self.ocr_toggle, False)
            return
        paddleocr_dependency = next(
            (item for item in RUNTIME_DEPENDENCIES if item.name == "PaddleOCR"),
            None,
        )
        if paddleocr_dependency is not None and not runtime_dependency_installed(
            paddleocr_dependency
        ):
            self._set_toggle_checked(self.ocr_toggle, False)
            if self._confirm_optional_runtime_install("PaddleOCR"):
                self._install_optional_runtime_dependency("PaddleOCR")
            return
        ocr_state = self._request_korean_ocr_model(start_ocr_after=True)
        if ocr_state in {"declined", "downloading"}:
            self._set_toggle_checked(self.ocr_toggle, False)
            return
        if self.ocr.is_running():
            self._set_toggle_checked(self.ocr_toggle, False)
            self._set_status(
                self._t(
                    "msg.ocr_start_failed",
                    error="이전 OCR 엔진이 아직 종료 중입니다. 잠시 후 다시 시도해주세요.",
                )
            )
            return
        self.ocr.set_source_language(self.settings.source_language)
        try:
            self._emit_ocr_status(OCR_STATUS_LOADING)
            self.ocr.start()
        except Exception as exc:
            self.settings.ocr_enabled = False
            self._set_toggle_checked(self.ocr_toggle, False)
            self._set_status(self._t("msg.ocr_start_failed", error=exc))
            return
        self.settings.ocr_enabled = True
        self._set_toggle_checked(self.ocr_toggle, True)
        save_settings(self.settings)

    def _saved_ocr_region(self) -> dict[str, int] | None:
        if self.settings.ocr_width <= 0 or self.settings.ocr_height <= 0:
            return None
        return {
            "left": self.settings.ocr_left,
            "top": self.settings.ocr_top,
            "width": self.settings.ocr_width,
            "height": self.settings.ocr_height,
        }

    def _ocr_region_text(self) -> str:
        region = self._saved_ocr_region()
        if region is None:
            return self._t("ocr.full_screen")
        return (
            f"X {region['left']}, Y {region['top']}, "
            f"{region['width']} x {region['height']}"
        )

    def _ocr_region_selected(self, region: dict[str, int]) -> None:
        self.settings.ocr_left = region["left"]
        self.settings.ocr_top = region["top"]
        self.settings.ocr_width = region["width"]
        self.settings.ocr_height = region["height"]
        self.ocr.set_region(region)
        self.ocr_region.setText(self._ocr_region_text())
        save_settings(self.settings)
        self._set_status(self._t("msg.ocr_region_saved"))

    def _toggle_server(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None
            self.host_toggle.setText(self._t("network.host_start"))
            self.settings.host_server_enabled = False
            save_settings(self.settings)
            return
        self._save_network_settings()
        if len(self.settings.password) < 12:
            QMessageBox.warning(
                self,
                self._t("msg.password_title"),
                self._t("msg.password_min_length"),
            )
            return
        if self.settings.client_ip_whitelist_enabled:
            if not parse_ip_list(self.settings.client_ip_whitelist_allowed):
                QMessageBox.warning(
                    self,
                    self._t("msg.whitelist_recommend_title"),
                    self._t("msg.whitelist_empty"),
                )
                return
        else:
            QMessageBox.information(
                self,
                self._t("msg.whitelist_recommend_title"),
                self._t("msg.whitelist_recommend_body"),
            )
            answer = QMessageBox.warning(
                self,
                self._t("msg.listen_all_title"),
                self._t("msg.listen_all_body"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        server = CaptionServer(
            self.settings,
            self.bus,
            lambda source, audio, sample_rate, source_language, target_language:
                self.pipeline.process_remote_audio(
                    source, audio, sample_rate, source_language, target_language
                ),
            lambda source, text, source_language, target_language:
                self.pipeline.process_remote_text(
                    source, text, source_language, target_language
                ),
            self._host_translate_text,
        )
        try:
            server.start()
        except RuntimeError as exc:
            self._set_status(str(exc))
            QMessageBox.warning(self, self._t("msg.server_start_failed"), str(exc))
            return
        self.server = server
        self.settings.host_server_enabled = True
        save_settings(self.settings)
        self.host_toggle.setText(self._t("network.host_stop"))
        started_key = (
            "msg.server_started_whitelist"
            if self.settings.client_ip_whitelist_enabled
            else "msg.server_started"
        )
        self._set_status(self._t(started_key))

    def _start_auto_obs(self) -> None:
        return

    def _restore_runtime_state(self) -> None:
        self.settings.host_server_enabled = False
        self.settings.remote_enabled = False
        self.settings.input_capture_enabled = False
        self.settings.output_capture_enabled = False
        self.settings.ocr_enabled = False
        self._set_toggle_checked(self.input_toggle, False)
        self._set_toggle_checked(self.output_toggle, False)
        self._set_toggle_checked(self.ocr_toggle, False)
        self._refresh_dashboard()

    def _toggle_remote(self) -> None:
        if self.remote:
            self._stop_all_detection()
            self.remote.stop()
            self.remote = None
            self.client_toggle.setText(self._t("network.client_connect"))
            self.client_start.setText(self._t("btn.client_start"))
            self.settings.remote_enabled = False
            self.remote_connected = False
            self.remote_model_info = {}
            self._refresh_dashboard()
            save_settings(self.settings)
            return
        if self.local_running or self.local_starting:
            self._stop_local_mode()
        self._save_network_settings()
        self.remote = RemoteClient(
            self.settings.remote_url,
            self.settings.password,
            self.bus,
            self.bridge.status.emit,
            self.bridge.remote_info.emit,
            self._remote_inline_translation_result,
        )
        self.remote.start()
        self.client_toggle.setText(self._t("network.client_disconnect"))
        self.client_start.setText(self._t("btn.client_stop"))
        self.settings.remote_enabled = True
        self.local_running = False
        self.local_starting = False
        self._refresh_dashboard()
        save_settings(self.settings)

    def _process_audio(self, source: SourceKind, audio: object, sample_rate: int) -> None:
        if self.remote:
            source_language, target_language = self.settings.languages_for(source)
            self.remote.send_audio(source, audio, sample_rate, source_language, target_language)
            self.bridge.source_status.emit(source, self._t("msg.audio_sending"))
        else:
            if not self.pipeline.is_ready():
                self.bridge.source_status.emit(
                    source,
                    self._t("engine.model_loading", devices=self.pipeline.runtime_device_report()),
                )
                return
            self.pipeline.process_audio(source, audio, sample_rate)

    def _process_text(self, source: SourceKind, text: str) -> None:
        if self.remote:
            source_language, target_language = self.settings.languages_for(source)
            self.remote.send_text(source, text, source_language, target_language)
        else:
            if not self.pipeline.is_ready():
                self.bridge.source_status.emit(
                    source,
                    self._t("engine.model_loading", devices=self.pipeline.runtime_device_report()),
                )
                return
            self.pipeline.process_text(source, text)

    def _toggle_inline_translator(self, checked: bool) -> None:
        self.inline_body.setVisible(checked)
        if checked:
            self._schedule_inline_translation()
        else:
            self.inline_translation_timer.stop()

    def _schedule_inline_translation(self, *_args: object) -> None:
        if not hasattr(self, "inline_translation_timer"):
            return
        self.inline_translation_timer.stop()
        if self.inline_translator_enabled.isChecked() and self.inline_input.text().strip():
            self.inline_translation_timer.start()

    def _run_inline_translation(self) -> None:
        self.inline_translation_timer.stop()
        text = self.inline_input.text().strip()
        if not text:
            return
        source_language = self.inline_source_language.currentText()
        target_language = self.inline_target_language.currentText()
        cross_check = self.inline_cross_check.isChecked()
        self._inline_request_id += 1
        request_id = self._inline_request_id
        self._inline_last_translated = ""
        self.inline_translated_text.setText(self._t("inline_translator.translating"))
        self.inline_verified_text.clear()
        if self.remote:
            if not self.remote_connected:
                self.inline_translated_text.setText(self._t("inline_translator.connect_remote"))
                self.inline_verified_text.clear()
                return
            self.remote.send_translate(
                request_id,
                text,
                source_language,
                target_language,
                cross_check,
            )
            return
        if self.local_starting:
            self.inline_translated_text.setText(self._t("inline_translator.loading"))
            self.inline_verified_text.clear()
            return
        if not self.local_running:
            self.inline_translated_text.setText(self._t("inline_translator.start_engine"))
            self.inline_verified_text.clear()
            return
        threading.Thread(
            target=self._run_inline_translation_worker,
            args=(request_id, text, source_language, target_language, cross_check),
            daemon=True,
        ).start()

    def _host_translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        cross_check: bool,
    ) -> tuple[str, str]:
        translated = self.pipeline.translate_text(text, source_language, target_language)
        verified = ""
        if cross_check and translated.strip():
            verified = self.pipeline.translate_text(translated, target_language, source_language)
        return translated, verified

    def _remote_inline_translation_result(self, message: dict[str, object]) -> None:
        self.bridge.inline_translation.emit(
            {
                "request_id": int(message.get("request_id", -1)),
                "source": "",
                "translated": str(message.get("translated", "")),
                "verified": str(message.get("verified", "")),
                "error": str(message.get("error", "")),
            }
        )

    def _run_inline_translation_worker(
        self,
        request_id: int,
        text: str,
        source_language: str,
        target_language: str,
        cross_check: bool,
    ) -> None:
        try:
            translated = self.pipeline.translate_text(text, source_language, target_language)
            verified = ""
            if cross_check and translated.strip():
                verified = self.pipeline.translate_text(translated, target_language, source_language)
            self.bridge.inline_translation.emit(
                {
                    "request_id": request_id,
                    "source": text,
                    "translated": translated,
                    "verified": verified,
                    "error": "",
                }
            )
        except Exception as exc:
            self.bridge.inline_translation.emit(
                {
                    "request_id": request_id,
                    "source": text,
                    "translated": "",
                    "verified": "",
                    "error": str(exc),
                }
            )

    def _inline_translation_finished(self, result: dict[str, object]) -> None:
        if int(result.get("request_id", -1)) != self._inline_request_id:
            return
        if result.get("error"):
            self._inline_last_translated = ""
            self.inline_translated_text.setText(self._t("error.inline", error=result["error"]))
            self.inline_verified_text.clear()
            return
        translated = str(result.get("translated", ""))
        verified = str(result.get("verified", ""))
        self._inline_last_translated = translated
        self.inline_translated_text.setText(translated)
        self.inline_verified_text.setText(verified)

    def _clear_inline_translation(self) -> None:
        self.inline_translation_timer.stop()
        self._inline_request_id += 1
        self._inline_last_translated = ""
        self.inline_input.clear()
        self.inline_translated_text.clear()
        self.inline_verified_text.clear()

    def _copy_inline_translation(self) -> None:
        QApplication.clipboard().setText(self._inline_last_translated or self.inline_translated_text.text())

    def _overlay_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.overlay_port}/overlay"

    def _copy_overlay_url(self) -> None:
        QApplication.clipboard().setText(self._overlay_url())
        self._set_status(self._t("msg.overlay_copied"))

    def _save_network_settings(self) -> None:
        self.settings.host = "0.0.0.0"
        self.settings.port = self.port.value()
        self.settings.password = self.password.text()
        self.settings.remote_host = self.remote_host.text().strip()
        self.settings.remote_port = self.remote_port.value()
        self.settings.remote_url = (
            f"ws://{self.settings.remote_host}:{self.settings.remote_port}/ws/client"
        )
        self._sync_whitelist_settings_from_ui()
        self.settings.always_run_obs_overlay = False
        self._update_whitelist_panel_visibility()
        self._refresh_dashboard()
        save_settings(self.settings)

    def _language_settings_changed(self, *_args: object) -> None:
        input_source, input_target = self.channel_languages[SourceKind.INPUT]
        output_source, output_target = self.channel_languages[SourceKind.OUTPUT]
        screen_source, screen_target = self.channel_languages[SourceKind.SCREEN]
        previous_screen_source = self.settings.source_language
        previous_screen_target = self.settings.target_language
        self.settings.input_source_language = input_source.currentText()
        self.settings.input_target_language = input_target.currentText()
        self.settings.output_source_language = output_source.currentText()
        self.settings.output_target_language = output_target.currentText()
        self.settings.source_language = screen_source.currentText()
        self.settings.target_language = screen_target.currentText()
        save_settings(self.settings)
        self.ocr.set_source_language(self.settings.source_language)
        screen_language_changed = (
            previous_screen_source != self.settings.source_language
            or previous_screen_target != self.settings.target_language
        )
        if screen_language_changed:
            if self.settings.ocr_enabled:
                self.ocr.stop()
                self.settings.ocr_enabled = False
                self._set_toggle_checked(self.ocr_toggle, False)
                self.bridge.source_status.emit(SourceKind.SCREEN, self._t("source.stopped"))
                save_settings(self.settings)
            self._start_ocr_cooldown()
        if (
            self.settings.source_language == "Korean"
            and previous_screen_source != "Korean"
            and not is_korean_ocr_model_ready()
        ):
            ocr_state = self._request_korean_ocr_model(start_ocr_after=False)
            if ocr_state == "declined":
                return
            if ocr_state == "downloading":
                return

    def _edit_llm_rules(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("msg.llm_rules_title"))
        dialog.resize(640, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(self._t("msg.llm_rules_hint")))
        editor = QTextEdit()
        editor.setPlainText(self.settings.llm_rules)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.llm_rules = editor.toPlainText().strip()
            self._reload_pipeline()
            save_settings(self.settings)
            self._set_status(self._t("msg.llm_rules_saved"))

    def _reload_pipeline(self) -> None:
        if hasattr(self, "pipeline"):
            self.pipeline.release()
        self._pipeline_generation += 1
        self.pipeline = TranslationPipeline(
            self.settings,
            self.bus,
            partial(self._emit_pipeline_source_status, self._pipeline_generation),
        )
        self.local_running = False
        self.local_starting = False
        self._refresh_dashboard()

    def _emit_pipeline_source_status(
        self,
        generation: int,
        source: SourceKind,
        message: str,
    ) -> None:
        if generation != self._pipeline_generation:
            return
        self.bridge.source_status.emit(source, message)

    def _toggle_local_mode(self) -> None:
        if self.local_running or self.local_starting:
            self._set_status(self._t("msg.engine_already_running"))
            return
        self._activate_selected_models(start_engine=True)

    def _stop_local_mode(self) -> None:
        self._stop_all_detection()
        self._reload_pipeline()
        self._set_status(self._t("msg.engine_stopped"))

    def _restore_device_selection(self, combo: QComboBox, device_id: str, device_name: str) -> None:
        index = combo.findData(device_id) if device_id else -1
        if index < 0 and device_name:
            index = combo.findText(device_name)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_capture_enabled(self, source: SourceKind, enabled: bool) -> None:
        combo = self.input_device if source == SourceKind.INPUT else self.output_device
        if source == SourceKind.INPUT:
            self.settings.input_capture_enabled = enabled
            self.settings.input_device_id = str(combo.currentData() or "")
            self.settings.input_device_name = combo.currentText()
        else:
            self.settings.output_capture_enabled = enabled
            self.settings.output_device_id = str(combo.currentData() or "")
            self.settings.output_device_name = combo.currentText()
        save_settings(self.settings)

    def _handle_runtime_install_action(self) -> None:
        if self._runtime_install_active is not None or self._runtime_downloads:
            self._cancel_runtime_install_batch()
            return
        if all_runtime_dependencies_installed():
            self._install_all_runtime_dependencies(reinstall=True)
            return
        self._install_all_runtime_dependencies(reinstall=False)

    def _install_optional_runtime_dependency(self, name: str) -> None:
        if self._runtime_install_active is not None or self._runtime_downloads:
            return
        dependency = next((item for item in RUNTIME_DEPENDENCIES if item.name == name), None)
        if dependency is None:
            return
        if runtime_dependency_installed(dependency):
            self._refresh_runtime_dependency_panel()
            return
        if not self._validate_runtime_cuda_requirements([dependency.name]):
            return
        self._runtime_install_queue = []
        self._runtime_install_batch_total = 1
        self._runtime_required_install_batch_active = False
        self._begin_runtime_install(dependency, reinstall=False)

    def _confirm_optional_runtime_install(self, name: str) -> bool:
        answer = QMessageBox.question(
            self,
            self._t("msg.optional_runtime_install_title", name=name),
            self._t("msg.optional_runtime_install_body", name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _install_all_runtime_dependencies(self, *, reinstall: bool) -> None:
        if self._runtime_install_active is not None or self._runtime_downloads:
            return
        if reinstall:
            if not self._confirm_runtime_reinstall_all():
                return
            targets = [
                dependency
                for dependency in runtime_dependencies_in_install_order()
                if not dependency.optional
                if runtime_install_available(dependency.name)
            ]
        else:
            targets = list(runtime_dependencies_pending_install())
            if not targets:
                self._set_status(self._t("runtime.all_installed"))
                self._refresh_runtime_dependency_panel()
                return
        if not targets:
            self._set_status(self._t("msg.runtime_no_source", name=""))
            return
        if not self._validate_runtime_cuda_requirements([target.name for target in targets]):
            return
        queue: list[tuple[str, bool]] = []
        for index, dependency in enumerate(targets):
            queue.append((dependency.name, reinstall and index == 0))
        first_name, first_reinstall = queue[0]
        first_dependency = next(
            item for item in RUNTIME_DEPENDENCIES if item.name == first_name
        )
        self._runtime_install_queue = queue[1:]
        self._runtime_install_batch_total = len(queue)
        self._runtime_required_install_batch_active = True
        self._begin_runtime_install(first_dependency, reinstall=first_reinstall)

    def _validate_runtime_cuda_requirements(self, names: tuple[str, ...] | list[str]) -> bool:
        for name in names:
            if name not in CUDA_RUNTIME_DEPENDENCIES:
                continue
            compatible, driver_message = check_cuda_driver_compatibility(name)
            if compatible:
                continue
            translated_message = translate_driver_message(self.tr, driver_message)
            self._set_download_status_text(translated_message)
            self._set_download_speed_text(self._t("msg.driver_update_needed"))
            self._set_status(self._t("msg.driver_low"))
            QMessageBox.warning(self, self._t("msg.driver_update_title"), translated_message)
            self._refresh_runtime_driver_warning()
            return False
        return True

    def _cancel_runtime_install_batch(self) -> None:
        active_name = self._runtime_install_active
        if active_name and active_name in self._runtime_downloads:
            cancel_event = self._runtime_downloads[active_name]
            if cancel_event.is_set():
                self._set_status(self._t("msg.runtime_cancel_wait", name=active_name))
                return
            cancel_event.set()
            terminate_active_install_processes()
            self._set_status(self._t("msg.runtime_install_cancel", name=active_name))
        self._runtime_install_queue.clear()
        self._runtime_install_batch_total = 0
        self._runtime_required_install_batch_active = False
        self._runtime_restart_required_names.clear()
        clear_runtime_install_queue()
        self._refresh_runtime_dependency_panel()
        self._update_runtime_install_status()

    def _confirm_runtime_reinstall_all(self) -> bool:
        installed_names = [
            dependency.name
            for dependency in runtime_dependencies_in_install_order()
            if runtime_dependency_installed(dependency)
        ]
        others_text = (
            self._t("msg.runtime_reinstall_others", names=", ".join(installed_names))
            if installed_names
            else ""
        )
        answer = QMessageBox.question(
            self,
            self._t("msg.runtime_reinstall_title", name=self._t("runtime.reinstall_all")),
            self._t(
                "msg.runtime_reinstall_body",
                name=self._t("runtime.reinstall_all"),
                others=others_text,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _begin_runtime_install(
        self,
        dependency: RuntimeDependency,
        *,
        reinstall: bool = False,
    ) -> None:
        cancel_event = threading.Event()
        self._runtime_downloads[dependency.name] = cancel_event
        self._runtime_install_active = dependency.name
        self.download_progress.setValue(0)
        self._set_download_speed_text(self._t("msg.runtime_preparing"))
        self._update_runtime_install_status(dependency.name)
        self._refresh_runtime_dependency_panel()
        threading.Thread(
            target=self._install_runtime_dependency_worker,
            args=(dependency, cancel_event, reinstall),
            daemon=True,
        ).start()

    @staticmethod
    def _parse_install_step(detail: str) -> tuple[int, int] | None:
        if not detail.startswith("("):
            return None
        end = detail.find(")")
        if end <= 1:
            return None
        parts = detail[1:end].split("/", 1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return None
        return int(parts[0]), int(parts[1])

    def _update_runtime_install_status(
        self,
        name: str | None = None,
        percent: int | None = None,
        *,
        detail: str = "",
    ) -> None:
        active_name = name or self._runtime_install_active
        if not active_name:
            return
        current, total = self._runtime_install_position()
        step = self._parse_install_step(detail)
        if total <= 1:
            if step is not None:
                detail_text = self._t(
                    "msg.runtime_installing_step",
                    name=active_name,
                    current=step[0],
                    total=step[1],
                )
            else:
                detail_text = self._t("msg.runtime_installing", name=active_name)
            status = (
                self._t(
                    "msg.runtime_installing_status_step",
                    name=active_name,
                    current=step[0],
                    total=step[1],
                    percent=percent,
                )
                if step is not None and percent is not None
                else self._t("msg.runtime_installing_status", name=active_name, percent=percent)
                if percent is not None
                else detail_text
            )
        else:
            detail_text = self._t(
                "msg.runtime_install_queue",
                current=current,
                total=total,
                name=active_name,
            )
            status = (
                self._t(
                    "msg.runtime_install_queue_status",
                    current=current,
                    total=total,
                    name=active_name,
                    percent=percent,
                )
                if percent is not None
                else detail_text
            )
        self._set_download_status_text(detail_text)
        self._set_status(status)

    def _install_runtime_dependency_worker(
        self,
        dependency: RuntimeDependency,
        cancel_event: threading.Event,
        reinstall: bool = False,
    ) -> None:
        try:
            target = install_runtime_dependency(
                dependency,
                progress=lambda percent, detail: self.bridge.runtime_download_progress.emit(
                    dependency.name, percent, detail
                ),
                cancel_event=cancel_event,
                reinstall=reinstall,
            )
            self.bridge.runtime_downloaded.emit(dependency.name, str(target))
        except InterruptedError as exc:
            LOGGER.info("Runtime install cancelled: %s", dependency.name)
            self.bridge.runtime_downloaded.emit(dependency.name, f"CANCELLED:{exc}")
        except Exception as exc:
            LOGGER.exception("Runtime install failed: %s", dependency.name)
            self.bridge.runtime_downloaded.emit(dependency.name, f"ERROR:{exc}")
        finally:
            release_runtime_libraries()

    def _runtime_download_progress(self, name: str, percent: int, detail: str) -> None:
        if name != self._runtime_install_active or name not in self._runtime_downloads:
            return
        self.download_progress.setValue(percent)
        speed_lines = [detail] if detail else []
        if 1 <= percent < 100:
            speed_lines.append(self._t("msg.runtime_install_wait"))
        self._set_download_speed_text("\n".join(line for line in speed_lines if line))
        self._update_runtime_install_status(name, percent, detail=detail)

    def _prompt_runtime_restart(self, completed_name: str = "") -> None:
        QMessageBox.information(
            self,
            self._t("msg.runtime_restart_title"),
            self._t("msg.runtime_restart_body", name=completed_name),
            QMessageBox.StandardButton.Ok,
        )
        self._runtime_install_batch_total = 0

    def _resume_pending_runtime_installs(self) -> None:
        if self._runtime_install_active or self._runtime_downloads:
            return
        pending = load_runtime_install_queue()
        if not pending:
            return
        clear_runtime_install_queue()
        self._runtime_install_queue = pending
        self._runtime_install_batch_total = max(1, len(pending))
        self._runtime_required_install_batch_active = True
        self._set_status(self._t("msg.runtime_resume_queue", count=len(pending)))
        self._refresh_runtime_dependency_panel()
        self._start_next_runtime_install()

    @staticmethod
    def _runtime_restart_required(dependency: RuntimeDependency) -> bool:
        for package in dependency.packages:
            root_name = package.split(".", 1)[0].replace("-", "_")
            if root_name in sys.modules:
                return True
        return False

    def _start_next_runtime_install(self) -> None:
        self._runtime_install_active = None
        while self._runtime_install_queue:
            next_name, next_reinstall = self._runtime_install_queue.pop(0)
            LOGGER.info("Runtime install queue advancing to: %s", next_name)
            dependency = next(
                (item for item in RUNTIME_DEPENDENCIES if item.name == next_name),
                None,
            )
            if dependency is None or (
                runtime_dependency_installed(dependency) and not next_reinstall
            ):
                if self._runtime_install_batch_total > 0:
                    self._runtime_install_batch_total = max(
                        1,
                        self._runtime_install_batch_total - 1,
                    )
                continue
            self._begin_runtime_install(dependency, reinstall=next_reinstall)
            return
        LOGGER.info("Runtime install queue drained")
        show_required_restart_notice = self._runtime_required_install_batch_active
        self._runtime_install_batch_total = 0
        self._runtime_required_install_batch_active = False
        clear_runtime_install_queue()
        if all_runtime_dependencies_installed():
            self.download_progress.setValue(100)
            self._set_download_speed_text(self._t("state.complete"))
            self._set_download_status_text(self._t("runtime.all_installed"))
            self._set_status(self._t("runtime.all_installed"))
        self._refresh_runtime_dependency_panel()
        if show_required_restart_notice and all_runtime_dependencies_installed():
            self._prompt_runtime_restart()
            self._runtime_restart_required_names.clear()
        elif self._runtime_restart_required_names:
            completed_name = ", ".join(sorted(self._runtime_restart_required_names))
            self._prompt_runtime_restart(completed_name)
            self._runtime_restart_required_names.clear()

    def _runtime_download_finished(self, name: str, result: str) -> None:
        self._runtime_downloads.pop(name, None)
        if result.startswith("ERROR:"):
            self._set_download_speed_text(self._t("state.failed"))
            self._set_download_status_text(self._t("msg.runtime_failed", name=name, error=result[6:]))
            self._set_status(self._t("msg.runtime_failed_status", name=name, error=result[6:]))
            self._start_next_runtime_install()
            self._refresh_runtime_dependency_panel()
            return
        if result.startswith("CANCELLED:"):
            dependency = next(
                (item for item in RUNTIME_DEPENDENCIES if item.name == name),
                None,
            )
            if dependency is not None:
                clear_runtime_install_state(dependency)
            self._set_download_speed_text(self._t("state.cancelled"))
            self._set_download_status_text(
                self._t("msg.runtime_cancelled", name=name, detail=result[10:])
            )
            self._set_status(self._t("msg.runtime_cancelled_status", name=name))
            if self._runtime_install_batch_total > 0:
                self._runtime_install_batch_total = max(
                    1,
                    self._runtime_install_batch_total - 1,
                )
            self._start_next_runtime_install()
            self._refresh_runtime_dependency_panel()
            return
        self.download_progress.setValue(100)
        self._set_download_speed_text(self._t("state.complete"))
        self._set_download_status_text(self._t("msg.runtime_complete", name=name))
        self._set_status(self._t("msg.runtime_complete_status", name=name))
        dependency = next((item for item in RUNTIME_DEPENDENCIES if item.name == name), None)
        if dependency is not None:
            if self._runtime_restart_required(dependency):
                self._runtime_restart_required_names.add(name)
            else:
                from .runtime_bootstrap import configure_runtime_paths

                configure_runtime_paths()
        self._start_next_runtime_install()
        self._refresh_runtime_dependency_panel()

    def _download_model(self, kind: str) -> None:
        combo = self.download_asr_model if kind == "asr" else self.download_translation_model
        model_id, destination = self._download_target(kind, combo)
        download_key = (kind, model_id)
        if download_key == self._current_model_download:
            cancel_event = self._download_cancel_events.get(download_key)
            if cancel_event:
                cancel_event.set()
            self.download_status.setText(self._t("msg.download_cancel_request", model=model_id))
            self._set_status(self._t("msg.download_cancel_requested", model=model_id))
            self._update_download_buttons()
            return
        if download_key in self._download_queue:
            self._download_queue = [item for item in self._download_queue if item != download_key]
            self._active_model_downloads.discard(download_key)
            self._download_cancel_events.pop(download_key, None)
            self.download_status.setText(self._t("msg.download_queue_cancelled", model=model_id))
            self._set_status(self._t("msg.download_queue_cancel", model=model_id))
            self._update_download_buttons()
            return
        if is_model_complete(destination):
            self._model_download_finished(kind, str(Path(destination).resolve()))
            self._update_download_buttons()
            return
        if not model_id or "/" not in model_id:
            self.download_status.setText(self._t("msg.enter_hf_model"))
            self._set_status(self._t("msg.enter_hf_model_error"))
            return
        if self._current_model_download is not None:
            self._download_queue.append(download_key)
            self._active_model_downloads.add(download_key)
            self._download_cancel_events[download_key] = threading.Event()
            self.download_status.setText(self._t("msg.download_queued", model=model_id))
            self._set_status(self._t("msg.download_waiting", model=model_id))
            self._update_download_buttons()
            return
        self._start_model_download(kind, model_id)

    def _start_model_download(self, kind: str, model_id: str) -> None:
        download_key = (kind, model_id)
        cancel_event = self._download_cancel_events.get(download_key) or threading.Event()
        self._active_model_downloads.add(download_key)
        self._download_cancel_events[download_key] = cancel_event
        self._current_model_download = download_key
        if kind == "asr":
            button = self.asr_download
        elif kind == "translation":
            button = self.translation_download
        else:
            button = None
        if button is not None:
            button.setEnabled(True)
            button.setText(self._t("btn.download_cancel"))
        self.download_progress.setValue(0)
        self._set_download_speed_text(self._t("msg.download_preparing"))
        if kind == "ocr":
            status = self._t("msg.korean_ocr_downloading")
        else:
            status = self._t("msg.downloading", model=model_id)
        self._set_download_status_text(status)
        self._set_status(status)
        threading.Thread(
            target=self._download_model_worker,
            args=(kind, model_id, cancel_event),
            daemon=True,
        ).start()

    def _start_next_model_download(self) -> None:
        if self._current_model_download is not None or not self._download_queue:
            return
        kind, model_id = self._download_queue.pop(0)
        if kind == "ocr" and is_korean_ocr_model_ready():
            self._active_model_downloads.discard((kind, model_id))
            self._download_cancel_events.pop((kind, model_id), None)
            QTimer.singleShot(0, self._start_next_model_download)
            return
        if is_model_complete(model_download_path(kind, model_id)):
            self._active_model_downloads.discard((kind, model_id))
            self._download_cancel_events.pop((kind, model_id), None)
            QTimer.singleShot(0, self._start_next_model_download)
            return
        self._start_model_download(kind, model_id)

    def _download_model_worker(
        self, kind: str, model_id: str, cancel_event: threading.Event
    ) -> None:
        completion_reported = threading.Event()
        try:
            if kind == "ocr":
                self._download_korean_ocr_worker(cancel_event, completion_reported)
                return
            from huggingface_hub import HfApi

            model_dir = model_download_path(kind, model_id)
            if is_model_complete(model_dir):
                completion_reported.set()
                self.bridge.model_downloaded.emit(kind, str(model_dir.resolve()))
                return
            try:
                info = HfApi().model_info(model_id, files_metadata=True)
                expected_files = [
                    (str(sibling.rfilename), int(sibling.size or 0))
                    for sibling in info.siblings
                    if sibling.rfilename and self._should_download_model_file(model_id, sibling.rfilename)
                ]
                total_size = sum(size for _name, size in expected_files)
            except Exception:
                expected_files = []
                total_size = 0
            if not expected_files:
                raise RuntimeError("紐⑤뜽 ?뚯씪 紐⑸줉??媛?몄삤吏 紐삵뻽?듬땲??")
            downloaded_size = self._downloaded_model_bytes(model_dir, expected_files)
            previous_size = downloaded_size
            previous_time = time.monotonic()
            self.bridge.model_download_progress.emit(
                int(min(99, downloaded_size * 100 / total_size)) if total_size else 0,
                (
                    f"{self._format_bytes(downloaded_size)} / {self._format_bytes(total_size)}"
                    if total_size
                    else self._format_bytes(downloaded_size)
                ),
            )
            for filename, size in expected_files:
                if cancel_event.is_set():
                    raise InterruptedError("?ㅼ슫濡쒕뱶媛 痍⑥냼?섏뿀?듬땲??")
                self._download_hf_file(
                    model_id,
                    filename,
                    size,
                    model_dir,
                    expected_files,
                    total_size,
                    cancel_event,
                )
                downloaded_size = self._downloaded_model_bytes(model_dir, expected_files)
                now = time.monotonic()
                rate = max(0, downloaded_size - previous_size) / max(0.1, now - previous_time)
                percent = int(min(99, downloaded_size * 100 / total_size)) if total_size else 0
                detail = (
                    f"{self._format_bytes(downloaded_size)} / {self._format_bytes(total_size)}"
                    if total_size
                    else self._format_bytes(downloaded_size)
                )
                if rate > 0:
                    detail += f" 쨌 {self._format_bytes(rate)}/s"
                self.bridge.model_download_progress.emit(percent, detail)
                previous_size = downloaded_size
                previous_time = now
            path = model_dir
            if not completion_reported.is_set():
                completion_reported.set()
                self.bridge.model_downloaded.emit(kind, str(Path(path).resolve()))
        except InterruptedError as exc:
            self.bridge.model_downloaded.emit(kind, f"CANCELLED:{exc}")
        except Exception as exc:
            if is_model_complete(model_download_path(kind, model_id)):
                if not completion_reported.is_set():
                    completion_reported.set()
                    self.bridge.model_downloaded.emit(
                        kind, str(model_download_path(kind, model_id).resolve())
                    )
            else:
                self.bridge.model_downloaded.emit(kind, f"ERROR:{exc}")

    def _download_korean_ocr_worker(
        self,
        cancel_event: threading.Event,
        completion_reported: threading.Event,
    ) -> None:
        model_dir = korean_ocr_model_dir()
        if is_korean_ocr_model_ready():
            completion_reported.set()
            self.bridge.model_downloaded.emit("ocr", str(model_dir.resolve()))
            return

        stop_event = threading.Event()
        monitor = threading.Thread(
            target=self._monitor_korean_ocr_download,
            args=(model_dir, stop_event, cancel_event),
            daemon=True,
        )
        monitor.start()
        try:
            if cancel_event.is_set():
                raise InterruptedError("?ㅼ슫濡쒕뱶媛 痍⑥냼?섏뿀?듬땲??")
            download_korean_ocr_model()
            if not is_korean_ocr_model_ready():
                raise RuntimeError("?쒓뎅??OCR 紐⑤뜽 ?ㅼ슫濡쒕뱶媛 ?꾨즺?섏? ?딆븯?듬땲??")
            if not completion_reported.is_set():
                completion_reported.set()
                self.bridge.model_downloaded.emit("ocr", str(model_dir.resolve()))
        except InterruptedError as exc:
            self.bridge.model_downloaded.emit("ocr", f"CANCELLED:{exc}")
        except Exception as exc:
            if is_korean_ocr_model_ready():
                if not completion_reported.is_set():
                    completion_reported.set()
                    self.bridge.model_downloaded.emit("ocr", str(model_dir.resolve()))
            else:
                self.bridge.model_downloaded.emit("ocr", f"ERROR:{exc}")
        finally:
            stop_event.set()
            monitor.join(timeout=1)

    def _monitor_korean_ocr_download(
        self,
        model_dir: Path,
        stop_event: threading.Event,
        cancel_event: threading.Event,
    ) -> None:
        previous_size = 0
        previous_time = time.monotonic()
        while not stop_event.wait(0.5):
            if cancel_event.is_set():
                return
            current_size = self._directory_size(model_dir)
            now = time.monotonic()
            rate = max(0, current_size - previous_size) / max(0.1, now - previous_time)
            detail = self._format_bytes(current_size)
            if rate > 0:
                detail += f" 쨌 {self._format_bytes(rate)}/s"
            percent = 99 if is_korean_ocr_model_ready() else min(95, int(current_size / (14 * 1024 * 1024) * 100))
            self.bridge.model_download_progress.emit(percent, detail)
            previous_size = current_size
            previous_time = now

    @staticmethod
    def _directory_size(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    continue
        return total

    def _download_hf_file(
        self,
        model_id: str,
        filename: str,
        expected_size: int,
        model_dir: Path,
        expected_files: list[tuple[str, int]],
        total_size: int,
        cancel_event: threading.Event,
    ) -> None:
        from huggingface_hub import hf_hub_url
        import requests

        destination = model_dir / Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and (expected_size <= 0 or destination.stat().st_size >= expected_size):
            return

        partial_path = destination.with_name(destination.name + ".part")
        downloaded = partial_path.stat().st_size if partial_path.exists() else 0
        if expected_size > 0 and downloaded > expected_size:
            partial_path.unlink(missing_ok=True)
            downloaded = 0

        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        response = requests.get(
            hf_hub_url(repo_id=model_id, filename=filename),
            headers=headers,
            stream=True,
            timeout=(15, 60),
        )
        if response.status_code == 416:
            partial_path.replace(destination)
            return
        response.raise_for_status()
        if downloaded and response.status_code == 200:
            partial_path.unlink(missing_ok=True)
            downloaded = 0
        mode = "ab" if downloaded else "wb"
        chunk_since_update = 0
        last_update = time.monotonic()
        with partial_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if cancel_event.is_set():
                    raise InterruptedError("?ㅼ슫濡쒕뱶媛 痍⑥냼?섏뿀?듬땲??")
                if not chunk:
                    continue
                handle.write(chunk)
                chunk_since_update += len(chunk)
                now = time.monotonic()
                if now - last_update >= 0.5:
                    last_update = now
                    current = self._downloaded_model_bytes(model_dir, expected_files)
                    percent = int(min(99, current * 100 / total_size)) if total_size else 0
                    self.bridge.model_download_progress.emit(
                        percent,
                        (
                            f"{self._format_bytes(current)} / {self._format_bytes(total_size)}"
                            if total_size
                            else self._format_bytes(current)
                        )
                        + f" 쨌 {self._format_bytes(chunk_since_update * 2)}/s",
                    )
                    chunk_since_update = 0
        if expected_size > 0 and partial_path.stat().st_size < expected_size:
            raise RuntimeError(f"{filename} ?ㅼ슫濡쒕뱶媛 ?꾨즺?섏? ?딆븯?듬땲??")
        partial_path.replace(destination)

    @staticmethod
    def _should_download_model_file(model_id: str, filename: str) -> bool:
        normalized_model = model_id.lower()
        normalized_file = filename.replace("\\", "/").lower()
        if normalized_model == "tencent/hy-mt2-1.8b-gguf":
            return normalized_file.endswith("hy-mt2-1.8b-q4_k_m.gguf")
        return True

    def _monitor_model_download(
        self,
        kind: str,
        model_dir: Path,
        expected_files: list[tuple[str, int]],
        total_size: int,
        stop_event: threading.Event,
        completion_reported: threading.Event,
    ) -> None:
        previous_size = 0
        previous_time = time.monotonic()
        complete_seen_at: float | None = None
        stable_seen_at: float | None = None
        while not stop_event.wait(0.5):
            current_size = self._downloaded_model_bytes(model_dir, expected_files)
            current_size = max(previous_size, current_size)
            now = time.monotonic()
            rate = max(0, current_size - previous_size) / max(0.1, now - previous_time)
            percent = int(min(99, current_size * 100 / total_size)) if total_size else 0
            nearly_complete = bool(total_size and current_size >= total_size * 0.995)
            detail = (
                f"{self._format_bytes(current_size)} / {self._format_bytes(total_size)}"
                if total_size
                else self._format_bytes(current_size)
            )
            if rate > 0:
                detail += f" 쨌 {self._format_bytes(rate)}/s"
            elif percent >= 99 and is_model_complete(model_dir):
                detail += " - 마무리 확인 중"
            self.bridge.model_download_progress.emit(percent, detail)
            if is_model_complete(model_dir) and (not total_size or current_size >= total_size):
                if complete_seen_at is None:
                    complete_seen_at = now
                elif now - complete_seen_at >= 2.0 and not completion_reported.is_set():
                    completion_reported.set()
                    self.bridge.model_downloaded.emit(kind, str(model_dir.resolve()))
                    return
            else:
                complete_seen_at = None
            if is_model_complete(model_dir) and nearly_complete and rate <= 1024:
                if stable_seen_at is None:
                    stable_seen_at = now
                elif now - stable_seen_at >= 8.0 and not completion_reported.is_set():
                    completion_reported.set()
                    self.bridge.model_downloaded.emit(kind, str(model_dir.resolve()))
                    return
            else:
                stable_seen_at = None
            previous_size = current_size
            previous_time = now

    @staticmethod
    def _downloaded_model_bytes(model_dir: Path, expected_files: list[tuple[str, int]]) -> int:
        if not model_dir.exists():
            return 0
        if not expected_files:
            return MainWindow._folder_size_without_cache(model_dir)

        cache_dir = model_dir / ".cache" / "huggingface" / "download"
        cache_files = tuple(cache_dir.iterdir()) if cache_dir.exists() else ()
        total = 0
        for relative_name, expected_size in expected_files:
            relative_path = Path(relative_name)
            final_path = model_dir / relative_path
            best_size = 0
            try:
                if final_path.is_file():
                    best_size = final_path.stat().st_size
            except OSError:
                best_size = 0
            partial_path = final_path.with_name(final_path.name + ".part")
            try:
                if partial_path.is_file():
                    best_size = max(best_size, partial_path.stat().st_size)
            except OSError:
                pass
            file_name = relative_path.name
            for cached_path in cache_files:
                if not cached_path.name.startswith(file_name):
                    continue
                if cached_path.name.endswith(".metadata") or cached_path.name.endswith(".lock"):
                    continue
                try:
                    if cached_path.is_file():
                        best_size = max(best_size, cached_path.stat().st_size)
                except OSError:
                    continue
            if expected_size > 0:
                best_size = min(best_size, expected_size)
            total += best_size
        return total

    @staticmethod
    def _folder_size_without_cache(model_dir: Path) -> int:
        total = 0
        for path in model_dir.rglob("*"):
            if ".cache" in path.parts:
                continue
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    @staticmethod
    def _format_bytes(value: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"

    def _model_download_finished(self, kind: str, result: str) -> None:
        finished_key = self._current_model_download
        if finished_key and finished_key[0] == kind:
            self._active_model_downloads.discard(finished_key)
            self._download_cancel_events.pop(finished_key, None)
            self._current_model_download = None
        elif not result.startswith(("ERROR:", "CANCELLED:")):
            repository_id = repository_id_from_path(result)
            if repository_id:
                finished_key = (kind, repository_id)
                self._active_model_downloads.discard(finished_key)
                self._download_cancel_events.pop(finished_key, None)
        if kind == "asr":
            button = self.asr_download
        elif kind == "translation":
            button = self.translation_download
        else:
            button = None
        if button is not None:
            button.setEnabled(True)
        if result.startswith("ERROR:"):
            self.download_status.setText(self._t("msg.download_failed", error=result[6:]))
            self._set_download_speed_text(self._t("state.failed"))
            self._set_status(self._t("msg.download_failed_status", error=result[6:]))
            self._update_download_buttons()
            QTimer.singleShot(0, self._start_next_model_download)
            return
        if result.startswith("CANCELLED:"):
            self.download_status.setText(self._t("msg.download_cancelled", error=result[10:]))
            self._set_download_speed_text(self._t("state.cancelled"))
            self._set_status(self._t("msg.download_cancelled", error=result[10:]))
            self._update_download_buttons()
            QTimer.singleShot(0, self._start_next_model_download)
            return
        self.download_progress.setValue(100)
        self._set_download_speed_text(self._t("state.complete"))
        self._refresh_model_choices()
        self._refresh_dashboard()
        self._set_download_status_text(self._t("msg.download_complete", path=display_path(result)))
        self._set_status(self._t("msg.download_complete", path=display_path(result)))
        self._update_download_buttons()
        if kind == "ocr":
            if self._pending_ocr_after_korean_download:
                self._pending_ocr_after_korean_download = False
                self._enable_ocr_after_korean_model()
            QTimer.singleShot(0, self._start_next_model_download)
            return
        QTimer.singleShot(0, self._start_next_model_download)

    def _model_download_progress(self, percent: int, speed: str) -> None:
        self.download_progress.setValue(percent)
        self._set_download_speed_text(speed)
        if self._current_model_download and self._current_model_download[0] == "ocr":
            status = self._t("msg.korean_ocr_downloading")
        else:
            status = self._t("msg.download_progress", percent=percent, speed=speed)
        self._set_download_status_text(status)
        self._set_status(status)

    def _apply_selected_models(self) -> None:
        self._activate_selected_models(start_engine=False)

    def _activate_selected_models(self, start_engine: bool = True) -> None:
        selected_asr = self._selected_model_path(self.asr_model)
        selected_translation = self._selected_model_path(self.translation_model)
        if start_engine:
            missing = [
                label
                for label, kind, model in (
                    (self._t("label.stt"), "asr", selected_asr),
                    (self._t("label.translation_kind"), "translation", selected_translation),
                )
                if not is_model_complete(self._download_path_for_selection(kind, model))
            ]
            if missing:
                self.download_status.setText(
                    self._t("msg.missing_models", labels=", ".join(missing))
                )
                self._set_status(self._t("msg.download_models_first"))
                self._update_download_buttons()
                return
        if start_engine and not self._local_engine_supported(selected_asr, selected_translation):
            self.download_status.setText(self._t("msg.unsupported_combo"))
            self._set_status(self._t("msg.unsupported_combo_short"))
            return
        if "Qwen3.5" in selected_translation:
            self.download_status.setText(self._t("msg.qwen35_unsupported"))
            return
        asr_device = str(self.asr_device.currentData() or "cpu")
        if "sense-voice" in selected_asr.lower() or "sensevoice" in selected_asr.lower():
            asr_device = "cpu"
            self._select_compute_device(self.asr_device, "cpu")
        translation_device = str(self.translation_device.currentData() or "cpu")
        for selected_device in (asr_device, translation_device):
            valid, error = validate_device(selected_device)
            if not valid:
                self.download_status.setText(error)
                return
        self.settings.asr_device = asr_device
        self.settings.translation_device = translation_device
        self.settings.device = translation_device
        self.settings.asr_cpu_threads = int(self.asr_cpu_threads.currentData() or 0)
        self.settings.translation_cpu_threads = int(
            self.translation_cpu_threads.currentData() or 0
        )
        self.settings.asr_model = selected_asr
        self.settings.translation_model = selected_translation
        matched_preset = self._matching_model_preset()
        self.settings.model_preset = matched_preset or str(
            self.model_preset.currentData() or "game_light"
        )
        self.ocr.device = "cpu"
        self.settings.demo_mode = False
        self._reload_pipeline()
        if start_engine:
            self.local_starting = True
            self._refresh_dashboard()
            self._set_status(self._t("msg.model_loading"))
            self.pipeline.preload()
        if self.remote:
            self._toggle_remote()
        if self.server:
            self.server.broadcast_host_info()
        save_settings(self.settings)
        self.download_status.setText(
            self._t(
                "msg.selected_models",
                asr=display_path(self.settings.asr_model),
                translation=display_path(self.settings.translation_model),
            )
        )
        if start_engine:
            self._set_status(self._t("msg.model_loading"))
        else:
            self._set_status(self._t("msg.settings_saved"))
        self._refresh_dashboard()

    @staticmethod
    def _local_engine_supported(asr_model: str, translation_model: str) -> bool:
        asr_name = asr_model.replace("\\", "/")
        translation_name = translation_model.replace("\\", "/")
        asr_supported = (
            "Qwen3-ASR" in asr_name
            or "sense-voice" in asr_name.lower()
            or "sensevoice" in asr_name.lower()
        )
        translation_supported = "Qwen3" in translation_name or "hy-mt2" in translation_name.lower()
        return asr_supported and translation_supported

    def _refresh_model_choices(self) -> None:
        selections = (
            (self.asr_model, "asr", discover_models("asr"), self.settings.asr_model),
            (self.translation_model, "translation", discover_models("translation"), self.settings.translation_model),
        )
        counts: list[int] = []
        for combo, kind, models, selected in selections:
            counts.append(len(models))
            combo.blockSignals(True)
            combo.clear()
            downloaded_by_repo = {
                repository_id: path
                for path in models
                if (repository_id := repository_id_from_path(path))
            }
            added_paths: set[Path] = set()
            for model_id in self._recommended_model_ids(kind):
                downloaded_path = downloaded_by_repo.get(model_id)
                if downloaded_path:
                    combo.addItem(
                        self._model_combo_label(kind, model_id, downloaded=True),
                        model_id,
                    )
                    added_paths.add(downloaded_path.resolve())
                else:
                    combo.addItem(self._model_combo_label(kind, model_id), model_id)
            for path in models:
                if path.resolve() in added_paths:
                    continue
                repo_id = repository_id_from_path(path)
                display_id = repo_id or display_path(path)
                combo.addItem(self._model_combo_label(kind, display_id), str(path))
            selected_path = str(Path(selected).resolve()) if Path(selected).exists() else selected
            index = combo.findData(selected_path)
            if index < 0:
                selected_names = {Path(selected).name, selected.replace("/", "--")}
                index = next(
                    (i for i, path in enumerate(models) if path.name in selected_names),
                    -1,
                )
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                repo_id = repository_id_from_path(Path(selected)) if Path(selected).exists() else ""
                combo.setEditText(repo_id or selected)
                combo.setToolTip(selected)
            combo.blockSignals(False)
        self.download_status.setText(
            self._t("msg.detected_models", stt=counts[0], llm=counts[1])
        )
        self._update_download_buttons()
        self._refresh_delete_choices()

    @staticmethod
    def _recommended_model_ids(kind: str) -> list[str]:
        if kind == "asr":
            return [
                "Qwen/Qwen3-ASR-0.6B",
                "Qwen/Qwen3-ASR-1.7B",
                SENSEVOICE_ASR_MODEL,
            ]
        return [
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B-AWQ",
            "tencent/Hy-MT2-1.8B-GGUF",
            "tencent/Hy-MT2-1.8B-2bit-GGUF",
        ]

    def _update_download_buttons(self, *_args: object) -> None:
        if not hasattr(self, "translation_download"):
            return
        for combo, button in (
            (self.download_asr_model, self.asr_download),
            (self.download_translation_model, self.translation_download),
        ):
            kind = "asr" if combo is self.download_asr_model else "translation"
            model_id, destination = self._download_target(kind, combo)
            complete = is_model_complete(destination)
            key = (kind, model_id)
            active = key == self._current_model_download
            queued = key in self._download_queue
            button.setEnabled(bool(model_id.strip()) and (not complete or active or queued))
            button.setText(
                self._t("btn.download_done")
                if complete
                else self._t("btn.download_cancel")
                if active
                else self._t("btn.queue_cancel")
                if queued
                else self._t("btn.download")
            )

    def _download_target(self, kind: str, combo: QComboBox) -> tuple[str, str]:
        selected = self._selected_model_path(combo)
        model_id = repository_id_from_path(selected) if Path(selected).exists() else selected
        if model_id and "/" in model_id:
            return model_id, str(model_download_path(kind, model_id))
        return str(model_id or ""), selected

    def _download_path_for_selection(self, kind: str, selected: str) -> str:
        if Path(selected).exists():
            return selected
        if "/" in selected:
            return str(model_download_path(kind, selected))
        return selected

    def _refresh_delete_choices(self, *_args: object) -> None:
        if not hasattr(self, "delete_model_combo"):
            return
        kind = str(self.delete_kind.currentData() or "asr")
        self.delete_model_combo.clear()
        for path in discover_models(kind):
            self.delete_model_combo.addItem(display_path(path), str(path))

    def _delete_selected_model(self) -> None:
        kind = str(self.delete_kind.currentData() or "asr")
        selected = str(self.delete_model_combo.currentData() or "")
        if not selected:
            self.download_status.setText(self._t("msg.no_model_to_delete"))
            return
        active = self.settings.asr_model if kind == "asr" else self.settings.translation_model
        if Path(selected).resolve() == Path(active).resolve():
            self.download_status.setText(self._t("msg.cannot_delete_active"))
            return
        answer = QMessageBox.question(
            self,
            self._t("msg.delete_confirm_title"),
            self._t("msg.delete_confirm_body", model=display_path(selected)),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_model(kind, selected)
            self.download_status.setText(self._t("msg.delete_done", model=display_path(selected)))
            self._refresh_model_choices()
        except (OSError, ValueError) as exc:
            self.download_status.setText(self._t("msg.delete_failed", error=exc))

    def _selected_model_path(self, combo: QComboBox) -> str:
        data = combo.currentData()
        item_text = combo.itemText(combo.currentIndex()) if combo.currentIndex() >= 0 else ""
        if data and (
            combo.currentText() == item_text
            or combo.currentText() == display_path(data)
            or combo.currentText() == str(data)
        ):
            return str(data)
        return combo.currentText().strip()

    def _apply_styles(self) -> None:
        for source, controls in self.style_controls.items():
            font, size, color, outline_color, outline = controls
            style = CaptionStyle(
                font.current_font_family(),
                size.value(),
                color.text(),
                outline_color.text(),
                outline.value(),
            )
            setattr(self.settings, f"{source.value}_style", style)
            self._apply_label_style(source, style)
        save_settings(self.settings)
        self._set_status(self._t("msg.styles_applied"))

    def _apply_label_style(self, source: SourceKind, style: CaptionStyle) -> None:
        label = self.monitor_labels[source]
        label.setFont(QFont(style.font_family, max(8, style.font_size // 2)))
        label.setStyleSheet(f"color: {style.color}; background: #181818; padding: 12px;")
        self._render_unified_captions()

    def _show_caption(self, caption: Caption) -> None:
        if not self.unified_caption_history or (
            self.unified_caption_history[-1].source != caption.source
            or self.unified_caption_history[-1].original != caption.original
            or self.unified_caption_history[-1].translated != caption.translated
        ):
            self.unified_caption_history.append(caption)
            del self.unified_caption_history[:-3]
        self._render_unified_captions()

    def _render_unified_captions(self) -> None:
        source_names = {
            SourceKind.INPUT: self._t("caption.input"),
            SourceKind.OUTPUT: self._t("caption.output"),
            SourceKind.SCREEN: self._t("caption.screen"),
        }
        blocks: list[str] = []
        for item in self.unified_caption_history:
            style = getattr(self.settings, f"{item.source.value}_style")
            original = ""
            if not self.settings.omit_original_text:
                original = (
                    f"<div style='font-size:11px;color:#aaaaaa'>{html.escape(self._t('caption.original_tag'))}</div>"
                    f"<div style='font-size:15px;color:#dddddd'>{_html_preserve_lines(item.original)}</div>"
                )
            blocks.append(
                "<div style='margin-bottom:14px'>"
                f"<div style='font-size:10px;color:#999999'>[{source_names[item.source]}]</div>"
                f"{original}"
                f"<div style='font-size:11px;color:#aaaaaa'>{html.escape(self._t('caption.translation_tag'))}</div>"
                f"<div style='font-family:{html.escape(style.font_family)};"
                f"font-size:{max(16, style.font_size // 2)}px;color:{style.color};font-weight:700'>"
                f"{_html_preserve_lines(item.translated)}</div></div>"
            )
        rendered = "".join(blocks) or (
            f"<div style='color:#888888'>{html.escape(self._t('caption.waiting'))}</div>"
        )
        self.unified_caption_view.set_auto_scroll_html(rendered)
        self.caption_popup.set_caption_html(self._render_popup_caption_html())

    def _render_popup_caption_html(self) -> str:
        source_names = {
            SourceKind.INPUT: self._t("caption.input"),
            SourceKind.OUTPUT: self._t("caption.output"),
            SourceKind.SCREEN: self._t("caption.screen"),
        }
        font_family = html.escape(self.settings.popup_font_family)
        font_size = max(14, self.settings.popup_font_size)
        blocks: list[str] = []
        for item in self.unified_caption_history:
            style = getattr(self.settings, f"{item.source.value}_style")
            original = ""
            if not self.settings.omit_original_text:
                original = (
                    f"<div style='font-size:11px;color:rgba(170,170,170,0.9)'>"
                    f"{html.escape(self._t('caption.original_tag'))}</div>"
                    f"<div style='font-size:15px;color:rgba(221,221,221,0.95)'>"
                    f"{_html_preserve_lines(item.original)}</div>"
                )
            blocks.append(
                "<div style='margin-bottom:14px'>"
                f"<div style='font-size:10px;color:rgba(153,153,153,0.9)'>"
                f"[{source_names[item.source]}]</div>"
                f"{original}"
                f"<div style='font-size:11px;color:rgba(170,170,170,0.9)'>"
                f"{html.escape(self._t('caption.translation_tag'))}</div>"
                f"<div style='font-family:{font_family};font-size:{font_size}px;"
                f"color:{html.escape(style.color)};font-weight:700'>"
                f"{_html_preserve_lines(item.translated)}</div></div>"
            )
        return "".join(blocks) or (
            f"<div style='color:rgba(136,136,136,0.95)'>{html.escape(self._t('caption.waiting'))}</div>"
        )

    def _on_popup_settings_changed(self) -> None:
        save_settings(self.settings)
        if self.caption_popup.isVisible():
            self._render_unified_captions()

    def _omit_original_changed(self, checked: bool) -> None:
        self.settings.omit_original_text = checked
        save_settings(self.settings)
        self._render_unified_captions()

    def _toggle_caption_popup(self) -> None:
        if self.caption_popup.isVisible():
            self.caption_popup.hide()
            self.popup_toggle.setText(self._t("btn.popup_start"))
            self.settings.transparent_popup_enabled = False
        else:
            self._render_unified_captions()
            self.caption_popup._apply_geometry()
            self.caption_popup._apply_opacity()
            self.caption_popup.show()
            self.caption_popup.raise_()
            self.popup_toggle.setText(self._t("btn.popup_stop"))
            self.settings.transparent_popup_enabled = True
        save_settings(self.settings)

    def _save_ocr_refresh_settings(self, *_args: object) -> None:
        self.settings.ocr_auto_refresh = self.ocr_auto_refresh.isChecked()
        self.settings.ocr_interval = self.ocr_interval.value()
        self.ocr.set_refresh(self.settings.ocr_auto_refresh, self.settings.ocr_interval)
        self.ocr_interval.setEnabled(self.settings.ocr_auto_refresh)
        save_settings(self.settings)

    def _emit_ocr_status(self, state: str) -> None:
        if state == OCR_STATUS_LOADING:
            message = self._t("msg.screen_module_loading")
        elif state == OCR_STATUS_READY:
            message = self._t("msg.screen_detecting")
        else:
            message = state
        self.bridge.source_status.emit(SourceKind.SCREEN, message)

    def _set_source_status(self, source: SourceKind, message: str) -> None:
        label = self.source_status_labels.get(source)
        if not label:
            return
        history = self.source_status_history[source]
        if not history or history[-1] != message:
            history.append(message)
            del history[:-3]
        label.setText("\n".join(history))
        label.setMinimumHeight(label.fontMetrics().lineSpacing() * 3 + 6)
        color = "#ff7777" if any(self.tr.is_error_message(item) for item in history) else "#72d572"
        label.setStyleSheet(f"color: {color}; font-weight: 700;")
        if self.tr.is_error_message(message) or self.tr.is_busy_message(message):
            self._set_status(message)
        if source == SourceKind.SCREEN and any(token in message for token in ENGINE_LOAD_COMPLETE_TOKENS):
            self.local_starting = False
            self.local_running = True
            self._set_status(message)
            self._refresh_dashboard()
        elif source == SourceKind.SCREEN and any(token in message for token in ENGINE_PRELOAD_FAIL_TOKENS):
            self.local_starting = False
            self.local_running = False
            self._set_status(message)
            self._refresh_dashboard()

    def _set_status(self, message: str) -> None:
        self._status_base_message = message
        self._status_busy = self._is_busy_status(message) or bool(
            self._runtime_install_active or self._runtime_install_queue
        )
        self._render_status()

    def _tick_status_animation(self) -> None:
        if not self._status_busy:
            return
        self._status_frame_index = (self._status_frame_index + 1) % len(self._status_frames)
        self._render_status()

    def _render_status(self) -> None:
        message = self._status_base_message
        if self._status_busy:
            message = f"{self._status_base_message} {self._status_frames[self._status_frame_index]}"
        full_text = self._t("status.prefix", message=message)
        self.status.setToolTip(full_text)
        self.status.setText(self.status.fontMetrics().elidedText(full_text, Qt.TextElideMode.ElideRight, 920))
        is_error = self.tr.is_error_message(self._status_base_message) or any(
            token in self._status_base_message for token in FAILURE_TOKENS
        )
        color = "#ff7777" if is_error else "#72d572"
        if self._status_busy or self.tr.is_busy_message(self._status_base_message):
            color = "#e7d36f" if self._status_busy else color
        self.status.setStyleSheet(
            f"padding: 5px 8px; border-top: 1px solid #555; color: {color};"
        )

    def _is_busy_status(self, message: str) -> bool:
        return self.tr.is_busy_message(message)

    def _update_remote_info(self, info: dict[str, object]) -> None:
        self.remote_connected = bool(info.get("connected"))
        if self.remote_connected:
            self.remote_model_info = info
        else:
            self.remote_model_info = {}
        self._refresh_dashboard()

    @staticmethod
    def _status_label(label: QLabel, text: str, active: bool) -> None:
        color = "#72d572" if active else "#ff7777"
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-weight: 700;")

    def _refresh_dashboard(self) -> None:
        if not hasattr(self, "local_models_status"):
            return
        self.local_models_status.setText(
            f"STT: {display_path(self.settings.asr_model)}\n"
            f"LLM: {display_path(self.settings.translation_model)}"
        )
        if self.remote:
            self._status_label(
                self.runtime_status,
                self._t("runtime.client_mode_connected")
                if self.remote_connected
                else self._t("runtime.client_mode_waiting"),
                self.remote_connected,
            )
        else:
            self._status_label(
                self.runtime_status,
                (
                    self._t("runtime.local_running")
                    if self.local_running
                    else self._t("runtime.local_loading")
                    if self.local_starting
                    else self._t("runtime.stopped")
                ),
                self.local_running,
            )
        destination = f"{self.settings.remote_host}:{self.settings.remote_port}"
        self._status_label(
            self.remote_destination_status,
            f"{destination} - {self._t('runtime.connected') if self.remote_connected else self._t('runtime.disconnected')}",
            self.remote_connected,
        )
        if self.remote_connected:
            self.remote_models_status.setText(
                f"STT: {display_path(str(self.remote_model_info.get('asr_model', self._t('runtime.unknown'))))}\n"
                f"LLM: {display_path(str(self.remote_model_info.get('translation_model', self._t('runtime.unknown'))))}"
            )
        else:
            self.remote_models_status.setText(self._t("dashboard.host_models_pending"))
        self._refresh_run_controls()
        self._refresh_ocr_controls()

    def _engine_ready(self) -> bool:
        return (self.local_running and self.pipeline.is_ready()) or self.remote_connected

    def _refresh_run_controls(self) -> None:
        if not hasattr(self, "local_start"):
            return
        self.local_start.setEnabled(not self.local_running and not self.local_starting and not self.remote)
        self.local_stop.setEnabled(self.local_running or self.local_starting)
        detection_ready = self._engine_ready()
        self.input_toggle.setEnabled(detection_ready)
        self.output_toggle.setEnabled(detection_ready)
        if not detection_ready:
            for source in list(self.captures):
                self.captures.pop(source).stop()
                self._set_capture_enabled(source, False)
                self.bridge.source_status.emit(source, self._t("source.stopped"))
            if self.settings.ocr_enabled:
                self.ocr.stop()
                self.settings.ocr_enabled = False
                self.bridge.source_status.emit(SourceKind.SCREEN, self._t("source.stopped"))
            self._set_toggle_checked(self.input_toggle, False)
            self._set_toggle_checked(self.output_toggle, False)

    def _refresh_ocr_controls(self) -> None:
        if not hasattr(self, "ocr_toggle"):
            return
        ready = self._engine_ready()
        cooldown = self._ocr_cooldown_active()
        self.ocr_toggle.setEnabled(ready and not cooldown)
        self.select_ocr_region.setEnabled(not cooldown)
        if hasattr(self, "channel_languages"):
            screen_source, screen_target = self.channel_languages[SourceKind.SCREEN]
            screen_source.setEnabled(not cooldown)
            screen_target.setEnabled(not cooldown)
        if cooldown:
            self._set_toggle_checked(self.ocr_toggle, False)
            self.ocr_toggle.setText(self._ocr_cooldown_message())
            self.ocr_toggle.setToolTip(self._ocr_cooldown_tip())
        elif ready:
            self.ocr_toggle.setText(self._t("toggle.screen_detection"))
            self.ocr_toggle.setToolTip("")
        else:
            self._set_toggle_checked(self.ocr_toggle, False)
            self.ocr_toggle.setText(self._t("toggle.screen_detection_wait"))
            self.ocr_toggle.setToolTip(self._t("toggle.screen_detection_tip"))

    def closeEvent(self, event: QCloseEvent) -> None:
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        for cancel_event in list(self._runtime_downloads.values()):
            cancel_event.set()
        for cancel_event in list(self._download_cancel_events.values()):
            cancel_event.set()
        terminate_active_install_processes()
        self.settings.input_capture_enabled = False
        self.settings.output_capture_enabled = False
        self.settings.ocr_enabled = False
        self.settings.remote_enabled = False
        self.settings.host_server_enabled = False
        self.settings.input_device_id = str(self.input_device.currentData() or "")
        self.settings.input_device_name = self.input_device.currentText()
        self.settings.output_device_id = str(self.output_device.currentData() or "")
        self.settings.output_device_name = self.output_device.currentText()
        save_settings(self.settings)
        for capture in self.captures.values():
            capture.stop()
        self.ocr.stop()
        if hasattr(self, "inline_translation_timer"):
            self.inline_translation_timer.stop()
        self.local_running = False
        self.local_starting = False
        if hasattr(self, "pipeline"):
            self.pipeline.release()
        self.caption_popup.close()
        self.overlay_server.stop()
        if self.remote:
            self.remote.stop()
        if self.server:
            self.server.stop()
        release_runtime_libraries()

    def _start_overlay_server(self) -> None:
        try:
            self.overlay_server.start()
        except RuntimeError as exc:
            self._set_status(str(exc))


def run_app(settings: AppSettings) -> int:
    set_windows_app_user_model_id()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setOrganizationName("chomiles")
    icon = _application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = MainWindow(settings)
    app.aboutToQuit.connect(window._shutdown)
    window.show()
    QTimer.singleShot(0, window._start_overlay_server)
    QTimer.singleShot(500, window._resume_pending_runtime_installs)
    return app.exec()
