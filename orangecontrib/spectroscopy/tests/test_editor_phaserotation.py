import unittest

from orangecontrib.spectroscopy.preprocess import RotatePhase
from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.phaserotation import (
    RotatePhaseEditor,
)


class TestRotatePhaseEditor(PreprocessorEditorTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(RotatePhaseEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data

    def test_no_interaction(self):
        p = self.commit_get_preprocessor()
        self.assertIsInstance(p, RotatePhase)
        self.assertEqual(p.degree, 0.0)
        self.assertEqual(p.wn_ref, 1000.0)

    def test_basic(self):
        self.editor.degree = 42.5
        self.editor.wn_ref_line.position = 1234.0
        self.editor.edited.emit()

        p = self.commit_get_preprocessor()
        self.assertEqual(p.degree, 42.5)
        self.assertEqual(p.wn_ref, 1234.0)


if __name__ == "__main__":
    unittest.main()
