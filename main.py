import sys
import os
import librosa
import numpy as np
import qdarktheme
import soundfile as sf

from PyQt6.QtWidgets import (QMainWindow, QApplication, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QWidget, QFileDialog, 
                             QSlider, QComboBox, QLabel, QTabWidget, QListWidget, QGroupBox, QLineEdit, QListWidgetItem, QProgressDialog, QMessageBox)
from PyQt6.QtCore import QUrl, QTimer, Qt
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from canvas import AudioCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("smple - Professional Audio Analyzer")
        self.resize(1200, 600)

        self.render_queue = []
        self.current_file_path = None

        # Audio Player Setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # Timer for Playhead
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(50)
        self.playback_timer.timeout.connect(self.update_playhead)

        # Tabs Setup
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.setup_analyze_tab()
        self.setup_render_tab()

        # Analysis State Caching
        self.y = None
        self.sr = None
        self.beat_times = None
        self.beat_frames = None
        self.onset_env = None

    def setup_analyze_tab(self):
        self.analyze_tab = QWidget()
        self.analyze_layout = QVBoxLayout(self.analyze_tab)

        # Control Bar
        self.control_bar = QWidget()
        self.control_layout = QHBoxLayout(self.control_bar)
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_button = QPushButton("Load Track")
        self.load_button.clicked.connect(self.load_track)
        self.control_layout.addWidget(self.load_button)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_playback)
        self.control_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        self.control_layout.addWidget(self.stop_button)

        self.control_layout.addSpacing(10)

        self.add_queue_button = QPushButton("Add to Queue")
        self.add_queue_button.clicked.connect(self.add_selected_to_queue)
        self.add_queue_button.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.control_layout.addWidget(self.add_queue_button)

        self.control_layout.addSpacing(20)

        # Sensitivity Slider
        self.control_layout.addWidget(QLabel("Anchor Sensitivity:"))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(1, 100)
        self.sens_slider.setValue(80) # Default
        self.sens_slider.setFixedWidth(120)
        self.sens_slider.sliderReleased.connect(self.recalculate_phrases)
        self.control_layout.addWidget(self.sens_slider)

        self.control_layout.addSpacing(10)

        # Grid Size Combo
        self.control_layout.addWidget(QLabel("Grid Size:"))
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["1 Bar", "2 Bars", "4 Bars", "8 Bars"])
        self.grid_combo.currentIndexChanged.connect(self.recalculate_phrases)
        self.control_layout.addWidget(self.grid_combo)

        self.control_layout.addStretch()
        
        self.analyze_layout.addWidget(self.control_bar)

        # Audio Canvas
        self.canvas = AudioCanvas()
        self.canvas.seek_requested.connect(self.seek_player)
        self.canvas.add_to_queue_requested.connect(self.add_to_queue)
        self.analyze_layout.addWidget(self.canvas)

        self.tabs.addTab(self.analyze_tab, "Analyze")

    def setup_render_tab(self):
        self.render_tab = QWidget()
        self.render_layout = QVBoxLayout(self.render_tab)
        
        # Queue List
        self.queue_list = QListWidget()
        self.render_layout.addWidget(self.queue_list)
        
        # Remove button
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_from_queue)
        self.render_layout.addWidget(self.remove_btn)
        
        # Export Settings Group
        self.export_group = QGroupBox("Export Settings")
        self.export_layout = QVBoxLayout(self.export_group)
        
        self.dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_btn = QPushButton("Browse")
        self.dir_btn.clicked.connect(self.browse_export_dir)
        self.dir_layout.addWidget(self.dir_input)
        self.dir_layout.addWidget(self.dir_btn)
        self.export_layout.addLayout(self.dir_layout)
        
        self.format_layout = QHBoxLayout()
        self.format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["WAV", "FLAC"])
        self.format_layout.addWidget(self.format_combo)
        self.format_layout.addStretch()
        self.export_layout.addLayout(self.format_layout)
        
        self.render_layout.addWidget(self.export_group)
        
        # Render All button
        self.render_btn = QPushButton("RENDER ALL")
        self.render_btn.setMinimumHeight(50)
        self.render_btn.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.render_btn.clicked.connect(self.execute_render_queue)
        self.render_layout.addWidget(self.render_btn)
        
        self.tabs.addTab(self.render_tab, "Render")

    def start_playback(self):
        self.player.play()
        self.playback_timer.start()

    def stop_playback(self):
        self.player.stop()
        self.playback_timer.stop()
        self.canvas.set_playhead_position(0)

    def update_playhead(self):
        pos_ms = self.player.position()
        self.canvas.set_playhead_position(pos_ms / 1000.0)

    def seek_player(self, time_sec):
        self.player.setPosition(int(time_sec * 1000))
        self.canvas.set_playhead_position(time_sec)

    def add_selected_to_queue(self):
        if self.canvas.selected_phrase:
            self.add_to_queue(self.canvas.selected_phrase)
        else:
            print("No phrase selected to add to queue.")

    def recalculate_phrases(self):
        if self.beat_times is None or self.onset_env is None:
            return

        # Map slider 1-100 to delta 3.0 to 0.1
        val = self.sens_slider.value()
        delta = 3.0 - ((val / 100.0) * 2.9)

        # Grid size in beats
        grid_idx = self.grid_combo.currentIndex()
        bars = [1, 2, 4, 8][grid_idx]
        beat_interval = bars * 4

        boundary_frames = librosa.onset.onset_detect(onset_envelope=self.onset_env, sr=self.sr, delta=delta, pre_max=30, post_max=30)
        
        phrase_times_set = set()
        if len(self.beat_times) > 0:
            if len(boundary_frames) == 0:
                master_anchor_frame = self.beat_frames[0]
            else:
                strengths = self.onset_env[boundary_frames]
                master_anchor_frame = boundary_frames[np.argmax(strengths)]
                
            b_time = librosa.frames_to_time(master_anchor_frame, sr=self.sr)
            closest_beat_idx = np.argmin(np.abs(self.beat_times - b_time))
            
            # Extrapolate forward
            idx = closest_beat_idx
            while idx < len(self.beat_times):
                phrase_times_set.add(self.beat_times[idx])
                idx += beat_interval
                
            # Extrapolate backward
            idx = closest_beat_idx - beat_interval
            while idx >= 0:
                phrase_times_set.add(self.beat_times[idx])
                idx -= beat_interval
                
        phrase_times = np.array(sorted(list(phrase_times_set)))
        
        # Change label prefix depending on selected grid size
        prefix = "B" if bars == 1 else "P"
        self.canvas.set_phrase_markers(phrase_times, prefix=prefix)

    def load_track(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Audio File", 
            "", 
            "Audio Files (*.wav *.mp3)"
        )
        if file_path:
            print(f"Selected file: {file_path}")
            print("Analyzing track, please wait...")
            
            self.current_file_path = file_path
            self.player.setSource(QUrl.fromLocalFile(file_path))

            try:
                y, sr = librosa.load(file_path, sr=None)
                self.y = y
                self.sr = sr
                
                self.canvas.set_waveform_data(y, sr)
                
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                self.beat_frames = beat_frames
                bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
                
                self.beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                print(f"Detected Tempo: {bpm:.2f} BPM")
                
                self.canvas.set_beat_markers(self.beat_times)
                
                print("Detecting boundaries...")
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                self.onset_env = librosa.onset.onset_strength(S=S, sr=sr)
                
                # Apply markers using current UI tool settings
                self.recalculate_phrases()
                print("Analysis complete.")
                
            except Exception as e:
                print(f"Error analyzing track: {e}")

    def add_to_queue(self, phrase_data):
        if not self.current_file_path: return
        
        item_data = {
            'file_path': self.current_file_path,
            'phrase_name': phrase_data['phrase_name'],
            'start': phrase_data['start'],
            'end': phrase_data['end']
        }
        self.render_queue.append(item_data)
        
        filename = os.path.basename(self.current_file_path)
        
        start_mins = int(item_data['start'] // 60)
        start_secs = int(item_data['start'] % 60)
        end_mins = int(item_data['end'] // 60)
        end_secs = int(item_data['end'] % 60)
        
        display_text = f"{filename} - {item_data['phrase_name']} [{start_mins}:{start_secs:02d} - {end_mins}:{end_secs:02d}]"
        
        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.ItemDataRole.UserRole, item_data)
        self.queue_list.addItem(list_item)
        
        print(f"Added to queue: {display_text}")

    def remove_from_queue(self):
        selected_items = self.queue_list.selectedItems()
        if not selected_items: return
        
        for item in selected_items:
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data in self.render_queue:
                self.render_queue.remove(item_data)
            self.queue_list.takeItem(self.queue_list.row(item))
            
    def browse_export_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if dir_path:
            self.dir_input.setText(dir_path)

    def execute_render_queue(self):
        export_dir = self.dir_input.text().strip()
        if not export_dir or not os.path.isdir(export_dir):
            QMessageBox.warning(self, "Export Error", "Please select a valid export directory.")
            return
            
        if not self.render_queue:
            QMessageBox.information(self, "Export Queue Empty", "There are no phrases in the render queue.")
            return
            
        export_format = self.format_combo.currentText().lower()
        
        progress = QProgressDialog("Rendering files...", "Cancel", 0, len(self.render_queue), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Rendering")
        
        for i, item_data in enumerate(self.render_queue):
            if progress.wasCanceled():
                break
                
            file_path = item_data['file_path']
            phrase_name = item_data['phrase_name'].replace(" ", "").replace(":", "-").replace("(", "_").replace(")", "")
            start_time = item_data['start']
            end_time = item_data['end']
            
            try:
                # Load full audio at native sample rate
                y, sr = librosa.load(file_path, sr=None)
                
                # Convert time to samples
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Slice audio
                audio_slice = y[start_sample:end_sample]
                
                # Construct filename
                original_filename = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = f"{original_filename}_{phrase_name}.{export_format}"
                output_path = os.path.join(export_dir, output_filename)
                
                # Write file
                sf.write(output_path, audio_slice, sr)
                
            except Exception as e:
                print(f"Error rendering {file_path}: {e}")
                
            progress.setValue(i + 1)
            
        if not progress.wasCanceled():
            QMessageBox.information(self, "Render Complete", f"Successfully rendered {len(self.render_queue)} file(s).")
            self.render_queue.clear()
            self.queue_list.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
