import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

class ClickableTextItem(pg.TextItem):
    sigClicked = QtCore.pyqtSignal(int, float, float, str)
    
    def __init__(self, index, start_time, end_time, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.name = name
        
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.sigClicked.emit(self.index, self.start_time, self.end_time, self.name)
            ev.accept()
        else:
            ev.ignore()

class AudioCanvas(pg.PlotWidget):
    seek_requested = QtCore.pyqtSignal(float)
    add_to_queue_requested = QtCore.pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Standard Rekordbox-style aesthetics
        self.setBackground('#121212')  # Dark background
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(x=True, y=False)  # Enable pan/zoom on X-axis only
        self.setMenuEnabled(False)  # Disable default right-click menu
        
        # Initialize an empty PlotDataItem for the waveform curve
        pen = pg.mkPen(color='#00d0ff', width=1)  # Vibrant neon blue
        self.waveform_curve = pg.PlotDataItem(pen=pen)
        self.addItem(self.waveform_curve)
        
        # Initialize an empty list to manage InfiniteLine items (beat markers)
        self.beat_markers = []
        
        # Initialize an empty list to manage InfiniteLine items (phrase markers)
        self.phrase_markers = []
        self.phrase_texts = []

        # Playhead
        playhead_pen = pg.mkPen(color='y', width=2)
        self.playhead = pg.InfiniteLine(pos=0, angle=90, pen=playhead_pen, movable=False)
        self.addItem(self.playhead)
        
        # Selection Region
        self.selection_region = pg.LinearRegionItem(movable=False)
        self.selection_region.setBrush(pg.mkBrush(255, 255, 0, 50))
        self.selected_phrase = None

        # Connect mouse click event
        self.scene().sigMouseClicked.connect(self.mouse_clicked)

    def set_playhead_position(self, time_sec):
        self.playhead.setValue(time_sec)

    def mouse_clicked(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Map the scene click coordinate to the plot's view coordinates
            pos = self.plotItem.vb.mapSceneToView(event.scenePos())
            time_sec = pos.x()
            self.seek_requested.emit(time_sec)
            event.accept()

    def select_phrase(self, index, start_time, end_time, name):
        if self.selection_region not in self.items():
            self.addItem(self.selection_region)
            
        self.selection_region.setRegion((start_time, end_time))
        self.selected_phrase = {
            'phrase_name': name,
            'start': start_time,
            'end': end_time
        }

    def set_waveform_data(self, y, sr):
        if len(y) == 0:
            return
            
        # Compute a downsampled amplitude envelope for performance
        target_points = 20000  # Number of points for the envelope
        if len(y) > target_points:
            chunk_size = len(y) // (target_points // 2)
            num_chunks = len(y) // chunk_size
            y_trunc = y[:num_chunks * chunk_size]
            y_reshaped = y_trunc.reshape((num_chunks, chunk_size))
            y_max = np.max(y_reshaped, axis=1)
            y_min = np.min(y_reshaped, axis=1)
            env_y = np.empty(num_chunks * 2, dtype=y.dtype)
            env_y[0::2] = y_max
            env_y[1::2] = y_min
            duration = len(y) / sr
            env_t = np.linspace(0, duration, len(env_y))
            self.waveform_curve.setData(env_t, env_y)
        else:
            time_axis = np.linspace(0, len(y) / sr, len(y))
            self.waveform_curve.setData(time_axis, y)

    def set_beat_markers(self, beat_times):
        for marker in self.beat_markers:
            self.removeItem(marker)
        self.beat_markers.clear()
        
        dash_pen = pg.mkPen(color='r', style=QtCore.Qt.PenStyle.DashLine, width=1.5)
        for t in beat_times:
            line = pg.InfiniteLine(pos=t, angle=90, pen=dash_pen)
            self.addItem(line)
            self.beat_markers.append(line)
            
        self.autoRange()

    def set_phrase_markers(self, phrase_times, prefix="B"):
        for marker in self.phrase_markers:
            self.removeItem(marker)
        self.phrase_markers.clear()
        
        for text_item in self.phrase_texts:
            self.removeItem(text_item)
        self.phrase_texts.clear()
        
        if self.selection_region in self.items():
            self.removeItem(self.selection_region)
        self.selected_phrase = None
        
        solid_pen = pg.mkPen(color='#ff00ff', style=QtCore.Qt.PenStyle.SolidLine, width=3)
        
        x_data, y_data = self.waveform_curve.getData()
        total_duration = x_data[-1] if (x_data is not None and len(x_data) > 0) else (phrase_times[-1] + 10.0 if len(phrase_times) > 0 else 0)
        y_max = np.max(y_data) if (y_data is not None and len(y_data) > 0) else 1.0
        
        for i, t in enumerate(phrase_times):
            mins = int(t // 60)
            secs = int(t % 60)
            
            line = pg.InfiniteLine(pos=t, angle=90, pen=solid_pen)
            self.addItem(line)
            self.phrase_markers.append(line)
            
            end_time = phrase_times[i+1] if i + 1 < len(phrase_times) else total_duration
            name = f"{prefix}{i+1} ({mins}:{secs:02d})"
            
            text_item = ClickableTextItem(
                index=i, 
                start_time=t, 
                end_time=end_time, 
                name=name,
                text=name, 
                anchor=(0, 0), # Top-Left anchor so text draws downwards
                color='#ffffff',
                fill=pg.mkBrush('#ff00ff')
            )
            # Position it at the peak of the waveform
            text_item.setPos(t, y_max)
            text_item.sigClicked.connect(self.select_phrase)
            self.addItem(text_item)
            self.phrase_texts.append(text_item)
