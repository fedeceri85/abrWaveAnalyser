from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton, QDialogButtonBox
from PyQt5.QtCore import Qt

HELP_CONTENT = """
<h1 style="color: #0078d4;">ABR Wave Analyser User Guide</h1>

<p>Welcome to the ABR Wave Analyser. This tool helps you analyze Auditory Brainstem Response (ABR) waveforms.</p>

<h2 style="color: #0078d4;">Getting Started</h2>
<ul>
    <li><b>Single File Mode</b>: Analyze one ABR recording at a time.
        <ul>
            <li>Click <i>Open file</i> to load a .csv or .rst file.</li>
            <li>The traces will appear in the main view.</li>
        </ul>
    </li>
    <li><b>Multiple File Mode</b>: Batch process an experiment list.
        <ul>
            <li>Create a .csv list with columns for Filename and IDs.</li>
            <li>Click <i>Open experiment list</i> to load it.</li>
            <li>Use <i>Next/Previous file</i> buttons to navigate through recordings.</li>
        </ul>
    </li>
</ul>

<h2 style="color: #0078d4;">Interaction</h2>
<ul>
    <li><b>Select Trace</b>: Click on any trace in the main view to select it.</li>
    <li><b>View Waveform</b>: The selected trace appears in the "Current Waveform" panel (top right).</li>
    <li><b>Zoom/Pan</b>: 
        <ul>
            <li>Right-click and drag to zoom.</li>
            <li>Left-click and drag to pan.</li>
            <li>Click "A" in the bottom-left corner of a plot to auto-scale.</li>
        </ul>
    </li>
</ul>

<h2 style="color: #0078d4;">Peak Analysis</h2>
<p>Use the <b>Peak Controls</b> panel (bottom right tab) to mark peaks:</p>
<ol>
    <li>Select the peak type (e.g., P1, N1) from the dropdown.</li>
    <li>Click on a point in the "Current Waveform" plot to mark it.</li>
</ol>
<h2 style="color: #0078d4;">Keyboard Shortcuts</h2>
<table border="0" cellpadding="4" cellspacing="0" width="100%">
    <tr>
        <td colspan="2"><b>Navigation</b></td>
        <td colspan="2"><b>Peak Selection</b></td>
    </tr>
    <tr>
        <td width="20%"><b>W / S</b></td><td width="30%">Previous/Next Trace</td>
        <td width="20%"><b>1 / 2</b></td><td width="30%">P1 / N1</td>
    </tr>
    <tr>
        <td><b>A / D</b></td><td>Left/Right</td>
        <td><b>3 / 4</b></td><td>P2 / N2</td>
    </tr>
    <tr>
        <td colspan="2"><br><b>Actions</b></td>
        <td><b>5 / 6</b></td><td>P3 / N3</td>
    </tr>
    <tr>
        <td><b>R</b></td><td>Guess Higher</td>
        <td><b>7 / 8</b></td><td>P4 / N4</td>
    </tr>
    <tr>
        <td><b>F</b></td><td>Guess Lower</td>
        <td><b>9 / 0</b></td><td>P5 / N5</td>
    </tr>
    <tr>
        <td><b>Z</b></td><td>Set Threshold</td>
        <td><b>E</b></td><td>Cycle Peak Type</td>
    </tr>
</table>

<h2 style="color: #0078d4;">Saving Results</h2>
<ul>
    <li><b>Save ABR traces</b>: Exports the processed traces to CSV.</li>
    <li><b>Save results</b>: Saves peak locations, amplitudes, and latencies.</li>
</ul>

<hr>
<p style="font-size: 10px; color: #888;">ABR Wave Analyser v2.0 | Sheffield Hearing Research Group</p>
"""

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("User Guide")
        self.resize(600, 700)
        
        layout = QVBoxLayout(self)
        
        # Help text browser
        self.textBrowser = QTextBrowser()
        self.textBrowser.setHtml(HELP_CONTENT)
        self.textBrowser.setOpenExternalLinks(True)
        layout.addWidget(self.textBrowser)
        
        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)
