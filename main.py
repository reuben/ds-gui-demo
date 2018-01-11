#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import csv
import numpy as np
import os
import sys
import textwrap
import time
import tempfile
import wave

from deepspeech.model import Model
from queue import Queue
from threading import Thread

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *

BEAM_WIDTH = 512
LM_BINARY_PATH = 'data/lm/lm.binary'
LM_TRIE_PATH = 'data/lm/trie'
ALPHABET_CONFIG_PATH = 'data/alphabet.txt'
LM_WEIGHT = 1.50
WORD_COUNT_WEIGHT = 1.00
VALID_WORD_COUNT_WEIGHT = 1.50

N_INPUT = 26
N_CONTEXT = 9

model_file = sys.argv[1]
if not os.path.exists(model_file):
    print('Invalid model file {}'.format(model_file))
    exit(1)

# check if this is a release model and adjust paths automagically
base = os.path.dirname(model_file)
if os.path.exists(os.path.join(base, 'lm.binary')):
    LM_BINARY_PATH = os.path.join(base, 'lm.binary')
    LM_TRIE_PATH = os.path.join(base, 'trie')
    ALPHABET_CONFIG_PATH = os.path.join(base, 'alphabet.txt')


class Sample(QObject):
    def __init__(self, wav_path, transcription, source, extra_text, color):
        super().__init__()
        self._wav_path = wav_path
        self._transcription = transcription
        self._source = source
        self._extra_text = extra_text
        self._color = color
        self._button = None

    @property
    def wav_path(self):
        return self._wav_path

    @property
    def transcription(self):
        return self._transcription

    @property
    def source(self):
        return self._source

    @property
    def extra_text(self):
        return self._extra_text

    @property
    def color(self):
        return self._color

    def set_button(self, button):
        self._button = button

    @property
    def button(self):
        return self._button


def wav_length(wav_path):
    file = wave.open(wav_path)
    return file.getnframes() / file.getframerate()


class InferenceRunner(QObject):
    inference_done = pyqtSignal(Sample, str)

    def __init__(self):
        super().__init__()
        self._queue = Queue()
        self._thread = Thread(target=self._worker_thread)
        self._thread.daemon = True
        self._thread.start()

    def _worker_thread(self):
        print('restoring from {}'.format(model_file))
        model = Model(
            model_file,
            N_INPUT,
            N_CONTEXT,
            ALPHABET_CONFIG_PATH,
            BEAM_WIDTH)
        model.enableDecoderWithLM(
            ALPHABET_CONFIG_PATH,
            LM_BINARY_PATH,
            LM_TRIE_PATH,
            LM_WEIGHT,
            WORD_COUNT_WEIGHT,
            VALID_WORD_COUNT_WEIGHT)

        while True:
            cmd, *args = self._queue.get()
            if cmd == 'sample':
                sample = args[0]
                file = wave.open(sample.wav_path)
                audio = np.frombuffer(
                    file.readframes(
                        file.getnframes()),
                    dtype=np.int16)
                fs = file.getframerate()
                start = time.time()
                result = model.stt(audio, fs)
                inference_time = time.time() - start
                wav_time = wav_length(sample.wav_path)
                print('wav length: {}\ninference time: {}\nRTF: {:2f}'.format(
                    wav_time, inference_time, inference_time / wav_time))
                self.inference_done.emit(sample, result)
            elif cmd == 'stop':
                break

        sess.close()

    def inference(self, sample):
        self._queue.put(('sample', sample))

    def stop(self):
        self._queue.put(('stop', 'stop'))


class RichTextRadioButton(QRadioButton):
    def __init__(self, richLabel):
        # strip HTML from rich label
        xml = QXmlStreamReader(richLabel)
        plainLabel = ''
        while not xml.atEnd():
            if xml.readNext() == QXmlStreamReader.Characters:
                plainLabel += xml.text()
        super().__init__(plainLabel)
        self._richLabel = richLabel

    def paintEvent(self, event):
        super().paintEvent(event)

        rect = event.rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.eraseRect(
            rect.topLeft().x() + 18,
            rect.topLeft().y(),
            rect.width() - 18,
            rect.height())
        painter.translate(QPointF(18, 0))

        label = QTextDocument()
        font = label.defaultFont()
        font.setPixelSize(16)
        label.setDefaultFont(font)
        label.setHtml(self._richLabel)
        label.drawContents(painter)
        painter.end()


class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        self._tasksInProgress = 0
        self._recording = False

        audioFormat = QAudioFormat()
        audioFormat.setCodec('audio/pcm')
        audioFormat.setSampleRate(16000)
        audioFormat.setSampleSize(16)
        audioFormat.setChannelCount(1)
        audioFormat.setByteOrder(QAudioFormat.LittleEndian)
        audioFormat.setSampleType(QAudioFormat.SignedInt)

        inputDeviceInfo = QAudioDeviceInfo.defaultInputDevice()
        if not inputDeviceInfo.isFormatSupported(audioFormat):
            print('Can\'t record audio in 16kHz 16-bit signed PCM format.')
            self._audioInput = None
        else:
            self._audioInput = QAudioInput(audioFormat)

        self._samples = []
        with open('samples.csv', 'r') as csvfile:
            sampleReader = csv.reader(csvfile)
            next(sampleReader, None)  # skip header
            for wav_path, transcription, source, extra_text, color in sampleReader:
                self._samples.append(
                    Sample(
                        wav_path,
                        transcription,
                        source,
                        extra_text,
                        color))

        self._inferenceRunner = InferenceRunner()
        self._inferenceRunner.inference_done.connect(self._on_inference_done)

        self.create_UI()

    def create_UI(self):
        self.resize(1440, 880)
        self.setWindowTitle('Deep Speech Demo')

        quitAction = QAction('Quit')
        quitAction.setShortcut('Ctrl-Q')
        quitAction.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        appMenu = menubar.addMenu('File')
        appMenu.addAction(quitAction)

        sampleSelectionLabel = QLabel(
            '<span style="font-size:20px; font-style: bold;">' +
            'Click a sample below to hear it and see the transcription from the model:' +
            '</span>')
        sampleSelectionLabel.setStyleSheet('max-height: 30px; height: 30px;')
        sampleSelectionLabel.setAlignment(Qt.AlignCenter)

        if self._audioInput is not None:
            self._micButton = QPushButton(QIcon('microphone.png'), '')
            self._micButton.setCheckable(True)
            self._micButton.clicked.connect(self._on_mic_clicked)

        sampleSelectionAndMicInputHbox = QHBoxLayout()
        sampleSelectionAndMicInputHbox.addStretch(1)
        sampleSelectionAndMicInputHbox.addWidget(sampleSelectionLabel)
        sampleSelectionAndMicInputHbox.addStretch(1)
        if self._audioInput is not None:
            sampleSelectionAndMicInputHbox.addWidget(self._micButton)

        sampleSelectionGrid = QGridLayout()

        positions = [(j, i) for i in range(3) for j in range(5)]
        for i in range(0, min(15, len(self._samples))):
            transcription = self._samples[i].transcription
            if len(transcription) > 336:
                transcription = transcription[:336]
                transcription += '...'
            btn = QPushButton(textwrap.fill(transcription, 70))
            self._samples[i].set_button(btn)
            btn.clicked.connect(
                (lambda s: lambda: self._sample_clicked(s))(
                    self._samples[i]))
            btn.setStyleSheet(
                'min-height: 100px; min-width: 300px; border: 2px solid ' +
                self._samples[i].color +
                ';')
            sampleSelectionGrid.addWidget(btn, *positions[i])

        self._progressBar = QProgressBar(self)
        self._progressBar.setOrientation(Qt.Horizontal)
        self._progressBar.setFormat('Running inference...')
        self._progressBar.setRange(0, 0)
        self._progressBar.setVisible(False)

        self._transcriptionResult = QTextEdit()
        self._transcriptionResult.setReadOnly(True)
        self._transcriptionResult.setStyleSheet('height: 120px;')

        centralWidget = QWidget(self)

        topWidget = QWidget(centralWidget)
        topWidgetLayout = QVBoxLayout()
        topWidgetLayout.addLayout(sampleSelectionAndMicInputHbox)
        topWidget.setLayout(topWidgetLayout)
        topWidget.setFixedHeight(150)

        bottomWidget = QWidget(centralWidget)
        bottomWidgetLayout = QVBoxLayout()
        bottomWidgetLayout.addWidget(self._progressBar)
        bottomWidgetLayout.addWidget(self._transcriptionResult)
        bottomWidget.setLayout(bottomWidgetLayout)
        bottomWidget.setFixedHeight(130)

        vbox = QVBoxLayout()
        vbox.addWidget(topWidget)
        vbox.addStretch(1)
        vbox.addLayout(sampleSelectionGrid)
        vbox.addStretch(1)
        vbox.addWidget(bottomWidget)

        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)

        self.show()

    def _on_mic_clicked(self):
        if not self._recording:
            self._recording = True
            self._recordingDuration = 0
            self._recordingTimer = QTimer()
            self._recordingTimer.setTimerType(Qt.PreciseTimer)
            self._recordingTimer.timeout.connect(self._timer_timeout)
            self._recordingBuffer = QByteArray()
            self._inputIODevice = self._audioInput.start()
            self._inputIODevice.readyRead.connect(self._input_bytes_available)
            self._recordingTimer.start(100)
        else:
            self._recording = False
            self._recordingTimer.stop()
            self._micButton.setText('')
            self._audioInput.stop()
            f = tempfile.NamedTemporaryFile(delete=False)
            wav.write(
                f.name,
                16000,
                np.frombuffer(
                    self._recordingBuffer.data(),
                    np.int16))
            self._sample_recorded(f.name)

    def _input_bytes_available(self):
        self._recordingBuffer.append(self._inputIODevice.readAll())

    def _timer_timeout(self):
        self._recordingDuration += 100
        self._micButton.setText(
            '{:.1f}'.format(
                self._recordingDuration /
                1000))

    def _sample_recorded(self, wav_path):
        self._progressBar.setVisible(True)
        self._tasksInProgress += 1
        self._soundEffect = QSoundEffect()
        self._soundEffect.setSource(QUrl.fromLocalFile(wav_path))
        self._soundEffect.setLoopCount(0)
        self._soundEffect.setVolume(1.0)
        self._soundEffect.play()
        sample = Sample(wav_path, None, None, None, None)
        self._inferenceRunner.inference(sample)

    def _sample_clicked(self, sample):
        self._progressBar.setVisible(True)
        self._tasksInProgress += 1
        self._soundEffect = QSoundEffect()
        self._soundEffect.setSource(QUrl.fromLocalFile(sample.wav_path))
        self._soundEffect.setLoopCount(0)
        self._soundEffect.setVolume(1.0)
        self._soundEffect.play()
        sample.button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._soundEffect.playingChanged.connect(
            (lambda sample: lambda: self._on_playing_changed(sample))(sample))
        self._inferenceRunner.inference(sample)

    def _on_inference_done(self, sample, transcription):
        self._tasksInProgress -= 1
        self._progressBar.setVisible(self._tasksInProgress != 0)
        self._transcriptionResult.setHtml(
            '<p style="font-size: 20px; text-align: center;">Transcription: ' +
            transcription +
            '</p>')

    def _on_playing_changed(self, sample):
        sample.button.setIcon(QIcon())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    demo = MainWidget()

    sys.exit(app.exec_())
