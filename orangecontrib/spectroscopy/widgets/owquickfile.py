import os
import re
import logging
from warnings import catch_warnings

from AnyQt.QtWidgets import \
    QStyle, QVBoxLayout, QHBoxLayout, QVBoxLayout, QFileDialog, QFileSystemModel, QLabel, \
    QLineEdit, QTreeView, QTreeWidget, QTreeWidgetItem, QHeaderView, QSizePolicy as Policy, QCompleter
from AnyQt.QtCore import QDir, QSortFilterProxyModel

from Orange.data.io import FileFormat
from Orange.data.table import Table
from Orange.widgets import widget, gui
from Orange.widgets.data.owfile import add_origin
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler, SettingProvider
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.widget import Msg, Output

import orangecontrib.spectroscopy # to import get all reader namespaces

log = logging.getLogger(__name__)


class FileFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_string = ""

    def setFilterString(self, filter_string):
        self.filter_string = filter_string
        self.invalidateFilter()  # Trigger re-filtering

    def filterAcceptsRow(self, source_row, source_parent):
        idx = self.sourceModel().index(source_row, 0, source_parent)
        name = idx.data()

        if re.search(self.filter_string,name)==None and self.sourceModel().isDir(idx)==False:
            return False
        
        return True
    

class OWQuickfile(widget.OWWidget):
    name = "Quick File"
    id = "orangecontrib.spectroscopy.widgets.quickfile"
    # TODO: We will need to change the icon
    icon = "icons/quickfile.svg"
    description = "Read data from an input file selected from the file tree" \
                  "and send a data table to the output."
    priority = 100

    class Outputs:
        data = Output("Data", Table,
                      doc="Attribute-valued dataset read from the input file.")
        
    want_main_area = False

    SIZE_LIMIT = 0

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    domain_editor = SettingProvider(DomainEditor)

    class Warning(widget.OWWidget.Warning):
        file_too_big = widget.Msg("The file is too large to load automatically."
                                  " Press Reload to load.")
        load_warning = widget.Msg("Read warning:\n{}")
        performance_warning = widget.Msg(
            "Categorical variables with >100 values may decrease performance.")

    class Error(widget.OWWidget.Error):
        missing_reader = Msg("No tile-by-tile reader for this file.")
        file_not_found = widget.Msg("File not found.")
        sheet_error = widget.Msg("Error listing available sheets.")
        unknown = widget.Msg("Read error:\n{}")

    class NoFileSelected:
        pass
    
    def __init__(self):
        super().__init__()

        self.domain = None
        self.data = None
        self.reader_description = ["NeaSPEC single image"]
        self.reader = None

        initial_directory = os.getcwd()
        self.selected_folder = initial_directory
        self.base_folder = ""
        self.file_list = []
        self.filter_string = ""

        layout = QVBoxLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)
        
        browse_layout = QHBoxLayout()
        layout.addLayout(browse_layout)

        browse_label = QLabel("Select project folder:", self)
        browse_button = gui.button(self, self, ' ... ', callback=self.browse_folder, autoDefault=False)
        browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse_layout.addWidget(browse_label)
        browse_layout.addWidget(browse_button)
        browse_layout.addStretch()

        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Enter a string to filter files...")
        self.filter_input.textChanged.connect(self.filter_files)
        layout.addWidget(self.filter_input)

        self.treeview = QTreeView(self)
        layout.addWidget(self.treeview)

        # Source model
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setReadOnly(True)
        self.root = self.fileSystemModel.setRootPath(initial_directory)
        self.fileSystemModel.setFilter(QDir.Files|QDir.AllDirs|QDir.NoDotAndDotDot)

        # Proxy model for filtering
        self.proxy_model = FileFilterProxyModel()
        self.proxy_model.setDynamicSortFilter(True)
        self.proxy_model.setSourceModel(self.fileSystemModel)

        self.treeview.setModel(self.proxy_model)
        self.treeview.setRootIndex(self.proxy_model.mapFromSource(self.fileSystemModel.index(initial_directory)))
        self.treeview.clicked.connect(self.load_selected_file)

        self.treeview.header().setStretchLastSection(False)
        self.treeview.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        box = gui.vBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, '')

        # self.Warning.file_too_big()

    def browse_folder(self):
        fname = QFileDialog.getExistingDirectory(self)
        if fname:
            self.selected_folder = fname
            self.treeview.setRootIndex(self.proxy_model.mapFromSource(self.fileSystemModel.index(self.selected_folder)))
    
    def filter_files(self):
        filter_text = self.filter_input.text().strip()
        self.proxy_model.setFilterString(filter_text)
    
    def load_selected_file(self):
        idx = self.treeview.currentIndex()
        source_index = self.proxy_model.mapToSource(idx)
        indexItem = self.fileSystemModel.index(source_index.row(), 0, source_index.parent())
        self.filepath = self.fileSystemModel.filePath(indexItem)
        if not self.fileSystemModel.isDir(indexItem):
            self._try_load()

    def _try_load(self):
        # pylint: disable=broad-except

        if self.filepath and not os.path.exists(self.filepath):
            return self.Error.file_not_found

        try:
            self.reader = self.get_the_reader(self.filepath)
            # assert self.reader is not None
        except Exception:
            return self.Error.missing_reader

        if self.reader is self.NoFileSelected:
            self.Outputs.data.send(None)
            return None

        with catch_warnings(record=True) as warnings:
            try:
                data = self.reader.read()
            except Exception as ex:
                log.exception(ex)
                return lambda x=ex: self.Error.unknown(str(x))
            if warnings:
                self.Warning.load_warning(warnings[-1].message.args[0])

        self.infolabel.setText(self._describe(data))

        add_origin(data, self.filepath)
        self.data = data
        self.openContext(data.domain)
        self.Outputs.data.send(self.data)

        return None
    
########## GET THE READERS ##########
    def get_the_reader(self, filename):
        """Return reader instance that reads NeaSPEC single image files.

        Parameters
        ----------
        filename : str

        Returns
        -------
        FileFormat
        """

        readers = [f for f in FileFormat.formats
                   if getattr(f, "DESCRIPTION", None) in self.reader_description and getattr(f, "EXTENSIONS", None)]

        for reader in readers:
            if os.path.splitext(filename)[1] in reader.EXTENSIONS:
                return reader(filename)

        raise IOError('No readers for file "{}"'.format(filename))

    @staticmethod
    def _describe(table):
        def missing_prop(prop):
            if prop:
                return f"({prop * 100:.1f}% missing values)"
            else:
                return "(no missing values)"

        domain = table.domain
        text = ""

        attrs = getattr(table, "attributes", {})
        descs = [attrs[desc]
                 for desc in ("Name", "Description") if desc in attrs]
        if len(descs) == 2:
            descs[0] = f"<b>{descs[0]}</b>"
        if descs:
            text += f"<p>{'<br/>'.join(descs)}</p>"

        text += f"<p>{len(table)} instance(s)"

        missing_in_attr = missing_prop(table.has_missing_attribute()
                                       and table.get_nan_frequency_attribute())
        missing_in_class = missing_prop(table.has_missing_class()
                                        and table.get_nan_frequency_class())
        text += f"<br/>{len(domain.attributes)} feature(s) {missing_in_attr}"
        if domain.has_continuous_class:
            text += f"<br/>Regression; numerical class {missing_in_class}"
        elif domain.has_discrete_class:
            text += "<br/>Classification; categorical class " \
                f"with {len(domain.class_var.values)} values {missing_in_class}"
        elif table.domain.class_vars:
            text += "<br/>Multi-target; " \
                f"{len(table.domain.class_vars)} target variables " \
                f"{missing_in_class}"
        else:
            text += "<br/>Data has no target variable."
        text += f"<br/>{len(domain.metas)} meta attribute(s)"
        text += "</p>"

        if 'Timestamp' in table.domain:
            # Google Forms uses this header to timestamp responses
            text += f"<p>First entry: {table[0, 'Timestamp']}<br/>" \
                f"Last entry: {table[-1, 'Timestamp']}</p>"
        return text

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWQuickfile).run()