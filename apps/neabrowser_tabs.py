from AnyQt.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QRadioButton
from AnyQt.QtCore import Qt, QEvent, QObject
from AnyQt.QtWidgets import QWidget, QSplitter, QTabWidget

from orangecanvas.scheme.signalmanager import SignalManager
from orangecanvas.registry.description import WidgetDescription

from orangewidget.workflow.widgetsscheme import WidgetsScheme

from Orange.widgets.data.owfile import OWFile
from Orange.widgets.data.owtable import OWTable
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.owhyper import OWHyper
from orangecontrib.spectroscopy.widgets.owfilebrowser import OWFileBrowser
from orangecontrib.snom.widgets.owpreprocessimage import OWPreprocessImage


class EventFilterEscape(QObject):

    def __init__(self, parent):
        super().__init__(parent)
        parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        if not obj.__dict__:  # object destruction in progress
            return True
        if event.type() == QEvent.KeyPress and event.key() in [Qt.Key_Escape]:
            return True
        else:
            return obj.eventFilter(obj, event)


class mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SNOM data browser')
        self.workflow = WidgetsScheme()
        # Create a splitter
        splitter = QSplitter(Qt.Horizontal)

        # Create Quasar widgets
        # File viewer
        self.file_node, self.file = self.create_widget(OWFileBrowser)
        self.file.toggle_infobox()
        self.file.set_fixed_readers(["NeaSPEC single image","NeaSPEC spectrum and ifg files"])
        EventFilterEscape(self.file)
        # Preprocess widgets
        self.image_process_node, self.image_process = self.create_widget(OWPreprocessImage)
        self.spectra_process_node, self.spectra_process = self.create_widget(OWPreprocess)
        EventFilterEscape(self.image_process)
        EventFilterEscape(self.spectra_process)
        self.spectra_process_node.state_changed.connect(self.check_node_states)
        # Viewer widget
        self.hyperspectra_node, self.hyperspectra = self.create_widget(OWHyper)
        self.hyperspectra_image_node, self.hyperspectra_image = self.create_widget(OWHyper)
        EventFilterEscape(self.hyperspectra)
        EventFilterEscape(self.hyperspectra_image)

        # Set up layout
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        splitter.addWidget(self.file)

        self.tabs = QTabWidget(self)
        self.tab_preprocess = QWidget()
        self.tab_display = QWidget()
        self.tabs.addTab(self.tab_preprocess,"Preprocess")
        self.tabs.addTab(self.tab_display,"Display")

        splitter.addWidget(self.tabs)
        self.tab_preprocess.layout = QHBoxLayout(self.tab_preprocess)
        self.tab_preprocess.layout.addWidget(self.image_process)
        self.tab_preprocess.layout.addWidget(self.spectra_process)

        self.tab_display.layout = QHBoxLayout(self.tab_display)
        self.tab_display.layout.addWidget(self.hyperspectra)
        self.tab_display.layout.addWidget(self.hyperspectra_image)

        self.spectra_process.setHidden(True)
        self.hyperspectra.setHidden(True)

        widget = QWidget()
        widget.setLayout(layout)

        self.workflow.new_link(self.file_node, "data", self.image_process_node, "data")
        self.workflow.new_link(self.file_node, "data", self.spectra_process_node, "data")
        self.workflow.new_link(self.spectra_process_node, "preprocessed_data", self.hyperspectra_node, "data")
        self.workflow.new_link(self.image_process_node, "preprocessed_data", self.hyperspectra_image_node, "data")
        
        sm = SignalManager()
        sm.set_workflow(self.workflow)

        self.file.show()
        self.image_process.show()
        # self.spectra_process.show()
        # self.hyperspectra.show()
        self.hyperspectra_image.show()

        sm.start()

        self.setCentralWidget(widget)
        self.hyperspectra._OWBaseWidget__toggleControlArea()
        self.hyperspectra_image._OWBaseWidget__toggleControlArea()

    def create_widget(self, cl):
        node = self.workflow.new_node(WidgetDescription(**cl.get_widget_description()))
        widget = self.workflow.widget_for_node(node)
        return node, widget
    
    def check_node_states(self,v):
        if self.file.reader:
            if self.file.reader.DESCRIPTION == "NeaSPEC single image":
                self.spectra_process.setHidden(True)
                self.hyperspectra.setHidden(True)
                self.image_process.setHidden(False)
                self.hyperspectra_image.setHidden(False)

            elif self.file.reader.DESCRIPTION == "NeaSPEC spectrum and ifg files":
                self.image_process.setHidden(True)
                self.hyperspectra_image.setHidden(True)
                self.spectra_process.setHidden(False)
                self.hyperspectra.setHidden(False)


app = QApplication([])
window = mainwindow()

window.show()
app.exec()