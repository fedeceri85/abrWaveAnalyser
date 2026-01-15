"""
Modern dark theme stylesheet for ABR Wave Analyser.
Provides a cohesive, professional appearance while maintaining readability.
"""

# Color palette
COLORS = {
    'background': '#1e1e1e',
    'background_alt': '#252526',
    'surface': '#2d2d2d',
    'surface_light': '#3c3c3c',
    'border': '#404040',
    'text': '#e0e0e0',
    'text_secondary': '#a0a0a0',
    'accent': '#0078d4',
    'accent_hover': '#1a8cff',
    'accent_pressed': '#005a9e',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'threshold_above': '#2d2d2d',  # Traces above threshold
    'threshold_below': '#803030',  # Traces below threshold (reddish)
}

# Main application stylesheet
DARK_STYLESHEET = f"""
/* ==================== Main Window ==================== */
QMainWindow {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
}}

QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: Helvetica, Arial, sans-serif;
    font-size: 12px;
}}

/* ==================== Dock Widgets ==================== */
QDockWidget {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    titlebar-close-icon: url(close.png);
    border: 1px solid {COLORS['border']};
}}

QDockWidget::title {{
    background-color: {COLORS['surface_light']};
    color: {COLORS['text']};
    padding: 8px 12px;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid {COLORS['border']};
}}

QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 2px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {COLORS['surface_light']};
    border-radius: 3px;
}}

/* ==================== Splitter ==================== */
QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 3px;
}}

QSplitter::handle:vertical {{
    height: 3px;
}}

QSplitter::handle:hover {{
    background-color: {COLORS['accent']};
}}

/* ==================== Buttons ==================== */
QPushButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 500;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {COLORS['surface_light']};
    border-color: {COLORS['accent']};
}}

QPushButton:pressed {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
    color: white;
}}

QPushButton:disabled {{
    background-color: rgba(255, 255, 255, 0.05);
    color: {COLORS['text_secondary']};
    border-color: {COLORS['border']};
}}

/* Primary action buttons */
QPushButton[primary="true"] {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
    color: white;
}}

QPushButton[primary="true"]:hover {{
    background-color: {COLORS['accent_hover']};
    border-color: {COLORS['accent_hover']};
}}

/* ==================== Parameter Tree ==================== */
QTreeView, QTreeWidget {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    outline: none;
    padding: 4px;
}}

QTreeView::item {{
    padding: 6px;
    border-radius: 4px;
    min-height: 30px; /* Taller rows for modern look */
    margin: 1px 4px;
}}

QTreeView::item:hover {{
    background-color: {COLORS['surface_light']};
}}

QTreeView::item:selected {{
    background-color: {COLORS['accent']};
    color: white;
}}

/* Hide the branch indicators for a cleaner look */
QTreeView::branch {{
    background-color: transparent;
}}

QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {{
    image: none;
}}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {{
    image: none;
}}

/* ==================== Embedded Widgets in Tree ==================== */
/* Make widgets inside the tree blend in seamlessly */
QTreeView QSpinBox, QTreeView QDoubleSpinBox, QTreeView QComboBox, QTreeView QLineEdit {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
}}

QTreeView QSpinBox:hover, QTreeView QDoubleSpinBox:hover, QTreeView QComboBox:hover, QTreeView QLineEdit:hover,
QTreeView QSpinBox:focus, QTreeView QDoubleSpinBox:focus, QTreeView QComboBox:focus, QTreeView QLineEdit:focus {{
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['accent']};
}}

QTreeView QPushButton {{
    margin: 2px 4px;
    padding: 4px 12px;
    background-color: {COLORS['surface_light']};
    border: 1px solid {COLORS['border']};
}}

QTreeView QPushButton:hover {{
    background-color: {COLORS['accent']};
    color: white;
    border: 1px solid {COLORS['accent']};
}}

/* Header styles for tree/table views */
QHeaderView::section {{
    background-color: {COLORS['surface_light']};
    color: {COLORS['text']};
    padding: 6px 12px;
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    font-weight: bold;
}}

/* ==================== Scroll Bars ==================== */
QScrollBar:vertical {{
    background-color: {COLORS['surface']};
    width: 12px;
    border-radius: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['surface_light']};
    border-radius: 6px;
    min-height: 30px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['border']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['surface']};
    height: 12px;
    border-radius: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['surface_light']};
    border-radius: 6px;
    min-width: 30px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {COLORS['border']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ==================== Input Fields ==================== */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {COLORS['accent']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent']};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {COLORS['surface_light']};
    border: none;
    width: 18px;
}}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {COLORS['accent']};
}}

/* ==================== Combo Box ==================== */
QComboBox {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 100px;
}}

QComboBox:hover {{
    border-color: {COLORS['accent']};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent']};
}}

/* ==================== Check Box ==================== */
QCheckBox {{
    color: {COLORS['text']};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS['border']};
    border-radius: 3px;
    background-color: {COLORS['surface']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['accent']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}

/* ==================== Group Box ==================== */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
}}

/* ==================== Status Bar ==================== */
QStatusBar {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border-top: 1px solid {COLORS['border']};
    padding: 4px 12px;
}}

QStatusBar::item {{
    border: none;
}}

/* ==================== Menu Bar ==================== */
QMenuBar {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 4px;
}}

QMenuBar::item {{
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['surface_light']};
}}

QMenuBar::item:pressed {{
    background-color: {COLORS['accent']};
}}

QMenu {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS['accent']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {COLORS['border']};
    margin: 6px 12px;
}}

/* ==================== Tab Widget ==================== */
QTabWidget::pane {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
}}

QTabBar::tab {{
    background-color: {COLORS['background']};
    color: {COLORS['text_secondary']};
    padding: 8px 16px;
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border-bottom: 2px solid {COLORS['accent']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS['surface_light']};
}}

/* ==================== Tool Tips ==================== */
QToolTip {{
    background-color: {COLORS['surface_light']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
}}

/* ==================== Labels ==================== */
QLabel {{
    color: {COLORS['text']};
    background-color: transparent;
}}

QLabel[heading="true"] {{
    font-size: 14px;
    font-weight: bold;
    color: {COLORS['text']};
    padding: 8px 0;
}}

/* ==================== File Dialog ==================== */
QFileDialog {{
    background-color: {COLORS['background']};
}}

/* ==================== Progress Bar ==================== */
QProgressBar {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    height: 20px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 3px;
}}
"""


def apply_dark_theme(app):
    """Apply the dark theme stylesheet to a QApplication."""
    app.setStyleSheet(DARK_STYLESHEET)


def get_plot_colors():
    """Get colors for pyqtgraph plots that work with the dark theme."""
    return {
        'background': 'w',  # Keep white for scientific data clarity
        'foreground': 'k',  # Black text/axes on white
        'trace_above': 'k',  # Black traces above threshold
        'trace_below': '#cc3333',  # Red traces below threshold
        'peak_marker': '#cc0000',  # Red peak markers
        'highlight': '#0078d4',  # Blue highlight for selected trace
    }
