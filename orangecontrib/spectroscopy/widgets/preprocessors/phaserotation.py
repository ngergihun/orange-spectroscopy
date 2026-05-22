from AnyQt.QtWidgets import QVBoxLayout, QFormLayout

from Orange.widgets import gui
from orangecontrib.spectroscopy.preprocess import RotatePhase as RotatePhasePreprocess
from orangecontrib.spectroscopy.widgets.gui import XPosLineEdit
from orangecontrib.spectroscopy.widgets.preprocessors.registry import preprocess_editors
from orangecontrib.spectroscopy.widgets.preprocessors.utils import (
    BaseEditorOrange,
    PreviewMinMaxMixin,
)


class RotatePhaseEditor(BaseEditorOrange, PreviewMinMaxMixin):
    name = "Rotate Phase"
    qualname = "orangecontrib.infrared.rotatephase"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        self.degree = 0.0
        self.wn_ref = 1000.0

        form = QFormLayout()
        self.degree_spin = gui.doubleSpin(
            None,
            self,
            "degree",
            minv=-360.0,
            maxv=360.0,
            step=0.1,
            callback=self.edited.emit,
        )
        form.addRow("Degree", self.degree_spin)
        self.wn_ref_line = None

        self.preview_data = None
        form.addRow("Wavenumber", self.add_wn_selection_ui())
        self.controlArea.layout().addLayout(form)


    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if (
            self.wn_ref_line
            and self.wn_ref_line.line not in self.parent_widget.curveplot.markings
        ):
            self.wn_ref_line.line.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(self.wn_ref_line.line)

    def add_wn_selection_ui(self):
        linelayout = gui.hBox(self)
        pmin, pmax = self.preview_min_max()
        e = XPosLineEdit(label="")
        e.set_default((pmin + pmax) / 2)
        linelayout.layout().addWidget(e)
        e.edited.connect(self.edited)
        e.focusIn.connect(self.activateOptions)
        self.wn_ref_line = e
        return linelayout

    def setParameters(self, params):
        self.degree = params.get("degree", 0.0)
        self.wn_ref = params.get("wn_ref", self.wn_ref)
        if self.wn_ref_line is not None:
            self.wn_ref_line.position = self.wn_ref

    def parameters(self):
        parameters = super().parameters()
        parameters["wn_ref"] = (
            float(self.wn_ref_line.position)
            if self.wn_ref_line is not None
            else float(self.wn_ref)
        )
        return parameters

    @staticmethod
    def createinstance(params):
        degree = params.get("degree", 0.0)
        wn_ref = params.get("wn_ref", 1000.0)
        return RotatePhasePreprocess(degree=degree, wn_ref=wn_ref)

    def set_preview_data(self, data):
        self.preview_data = data


preprocess_editors.register(RotatePhaseEditor, 126)
