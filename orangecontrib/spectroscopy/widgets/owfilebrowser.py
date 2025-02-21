import os
import re
import logging
from warnings import catch_warnings
import pathlib

from AnyQt.QtWidgets import \
    QStyle, QVBoxLayout, QHBoxLayout, QVBoxLayout, QFileDialog, QFileSystemModel, QScrollArea, QLabel, QWidget, QFrame,\
    QLineEdit, QTreeView, QComboBox, QHeaderView, QSizePolicy as Policy, QCompleter
from AnyQt.QtCore import QSize, QDir, QPoint, Qt, Signal, QSortFilterProxyModel

from orangewidget.utils.filedialogs import format_filter

from Orange.data.io import FileFormat, class_from_qualified_name
from Orange.data.io_base import MissingReaderException
from Orange.data.table import Table
from Orange.widgets import widget, gui
from Orange.widgets.data.owfile import add_origin
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler, SettingProvider
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.widget import Msg, Output

import orangecontrib.spectroscopy

DEFAULT_READER_TEXT = "Automatically detect type"

log = logging.getLogger(__name__)

class AddressBarLabel(QLabel):
    clicked = Signal(QPoint)

    def __init__(self, txt):
        super().__init__(txt)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel{ border-radius: 5px; }
            QLabel:hover{ background-color: #d4d2d2; }
        """)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.mapToParent(ev.pos()))

class AddressBar(QWidget):
    directoryClicked = Signal(str)  # New signal

    def __init__(self):
        super().__init__()

        # self.setContentsMargins(0, 0, 0, 0)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        self.setObjectName("main_frame")
        self.setStyleSheet("""
            QWidget#main_frame { border: 1px grey;
                                border-radius: 2px;
                                background-color: white;}
        """)

        self.sub_layout = QHBoxLayout()
        self.sub_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sub_layout.setContentsMargins(5, 2, 5, 2)
        self.setLayout(self.sub_layout)

    def stripAddressBar(self):
        # Remove all widgets and spacer items from the sub-layout
        for i in reversed(range(self.sub_layout.count())):
            item = self.sub_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            else:
                pass

    def get_dirlabels_width(self):
        width = 0
        for i in range(self.sub_layout.count()):
            item = self.sub_layout.itemAt(i)
            width += item.geometry().width()
        return width

    def updateAddressBar(self, path: pathlib.PurePath):
        # Remove any existing content from the address bar
        self.stripAddressBar()

        self.path_parts = path.parts
        self.displayed_parts = list(self.path_parts)
        for i, part in enumerate(self.displayed_parts):
            if "\\" in part:
                slashindex = part.find("\\")
                self.displayed_parts[i] = part[0:slashindex]

        self.sub_dir_labels = []

        # Add a QLabel widget for each subdirectory and a separator after each one
        for i, name in enumerate(self.displayed_parts):
            sub_dir_l = AddressBarLabel(name)

            self.sub_dir_labels.append(sub_dir_l)
            sub_dir_l.clicked.connect(self.onSubDirectoryClicked)
            self.sub_layout.addWidget(sub_dir_l)

            if i < len(self.path_parts) - 1:
                sep = QLabel("/")
                self.sub_layout.addWidget(sep)

        self.sub_dir_labels[-1].setStyleSheet("font-weight: bold")

        self.sub_layout.setSpacing(0)
        # if self.get_dirlabels_width() < self.sub_frame.geometry().width():
        if self.get_dirlabels_width() < self.geometry().width():
            self.sub_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            self.sub_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

    def onSubDirectoryClicked(self,ev):
        # labelItem = self.sub_frame.childAt(ev)
        labelItem = self.childAt(ev)
        index = int(self.sub_layout.indexOf(labelItem)/2)
        new_directory_path = pathlib.PurePath(*self.path_parts[:index + 1])
        if new_directory_path is not None:
            self.updateAddressBar(new_directory_path)
            self.directoryClicked.emit(str(new_directory_path))

    def resizeEvent(self, event):
      print("Widget resized to:", event.size())


class FileFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_string = ""
        self.filtered_extensions = ()

    def setFilterString(self, filter_string):
        self.filter_string = filter_string
        self.invalidateFilter()  # Trigger re-filtering

    def setExtensionFilter(self, extensions_tuple):
        self.filtered_extensions = extensions_tuple
        self.invalidateFilter()  # Trigger re-filtering

    def filterAcceptsRow(self, source_row, source_parent):
        idx = self.sourceModel().index(source_row, 0, source_parent)
        name = idx.data()

        if self.filtered_extensions == ():
            pass
        else:
            try:
                if (re.search(self.filter_string,name)==None or os.path.splitext(name)[1] not in self.filtered_extensions) and self.sourceModel().isDir(idx)==False:
                    return False
            except:
                pass
        
        return True
    

class OWFileBrowser(widget.OWWidget):
    name = "File Browser"
    id = "orangecontrib.protosec.widgets.filebrowser"
    icon = "icons/quickfile.svg"
    description = "Read data from an input file selected from the file tree" \
                  "and send a data table to the output."
    priority = 10

    class Outputs:
        data = Output("Data", Table,
                      doc="Attribute-valued dataset read from the input file.")
        
    want_main_area = False

    SIZE_LIMIT = 1e7

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    class Warning(widget.OWWidget.Warning):
        file_too_big = widget.Msg("The file is too large to load automatically."
                                  " Press Reload to load.")
        load_warning = widget.Msg("Read warning:\n{}")
        performance_warning = widget.Msg(
            "Categorical variables with >100 values may decrease performance.")

    class Error(widget.OWWidget.Error):
        missing_reader = Msg("No reader for this file.")
        file_not_found = widget.Msg("File not found.")
        sheet_error = widget.Msg("Error listing available sheets.")
        unknown = widget.Msg("Read error:\n{}")

    class NoFileSelected:
        pass
    
    def __init__(self,fixed_reader=False,reader_description="NeaSPEC single image"):
        super().__init__()

        self.domain = None
        self.data = None
        self.reader = None
        self.auto_reader = False

        if fixed_reader == True:
            self.reader_description = reader_description
            self.reader = self.get_described_reader()
        else:
            readers = [f for f in FileFormat.formats
                   if getattr(f, 'read', None)
                   and getattr(f, "EXTENSIONS", None)]
            
            def group_readers_per_addon_key(w):
                # readers from Orange.data.io should go first
                def package(w):
                    package = w.qualified_name().split(".")[:-1]
                    package = package[:2]
                    if ".".join(package) == "Orange.data":
                        return ["0"]  # force "Orange" to come first
                    return package
                return package(w), w.DESCRIPTION

            self.available_readers = sorted(set(readers),
                                        key=group_readers_per_addon_key)

        initial_directory = os.getcwd()
        self.selected_folder = initial_directory
        self.base_folder = ""
        self.file_list = []
        self.filter_string = ""

        layout = QVBoxLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)
        
        browse_layout = QHBoxLayout()
        layout.addLayout(browse_layout)

        # home_folder_button = gui.button(self, self, '', autoDefault=False)
        # home_folder_button.clicked.connect(lambda: self.jump_to_folder(initial_directory))
        # home_folder_button.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        # browse_layout.addWidget(home_folder_button)
        size = 30
        self.adr_bar = AddressBar()
        self.adr_bar.setMinimumSize(QSize(100, size))
        self.adr_bar.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.adr_bar.updateAddressBar(pathlib.PurePath(initial_directory))
        self.adr_bar.directoryClicked.connect(self.jump_to_folder)
        browse_layout.addWidget(self.adr_bar)

        # folder_up_button = gui.button(self, self, '', callback=self.folder_jump_up, autoDefault=False)
        # folder_up_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        # browse_layout.addWidget(folder_up_button)

        browse_button = gui.button(self, self, ' ... ', callback=self.browse_folder, autoDefault=False)
        browse_layout.addWidget(browse_button)
        # browse_layout.addStretch()

        if fixed_reader is False:
            box = gui.hBox(None, addToLayout=False, margin=0)
            box.setSizePolicy(Policy.Expanding, Policy.Fixed)
            self.reader_combo = QComboBox(self)
            self.reader_combo.setSizePolicy(Policy.Expanding, Policy.Fixed)
            self.reader_combo.setMinimumSize(QSize(100, 1))
            self.reader_combo.activated[int].connect(self.select_reader)
            box.layout().addWidget(self.reader_combo)
            layout.addWidget(box)
            self._initialize_reader_combo()

        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Enter a string to filter files...")
        self.filter_input.textChanged.connect(self.filter_files)
        layout.addWidget(self.filter_input)

        self.treeview = QTreeView(self)
        layout.addWidget(self.treeview)

        # Source model
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setReadOnly(True)
        self.fileSystemModel.setRootPath(initial_directory)
        self.fileSystemModel.setFilter(QDir.Files|QDir.AllDirs|QDir.NoDotAndDotDot)

        # Proxy model for filtering
        self.proxy_model = FileFilterProxyModel()
        self.proxy_model.setDynamicSortFilter(True)
        self.proxy_model.setSourceModel(self.fileSystemModel)
        if self.reader is not None:
            self.proxy_model.setExtensionFilter(self.reader.EXTENSIONS)

        self.treeview.setModel(self.proxy_model)
        self.treeview.setRootIndex(self.proxy_model.mapFromSource(self.fileSystemModel.index(initial_directory)))
        self.treeview.clicked.connect(self.load_selected_file)
        self.treeview.doubleClicked.connect(self.on_double_click)

        self.treeview.header().setStretchLastSection(False)
        self.treeview.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        box = gui.vBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, '')

    def browse_folder(self):
        fname = QFileDialog(self,"Select folder",self.selected_folder).getExistingDirectory()
        if fname:
            self.jump_to_folder(fname)

    def folder_jump_up(self):
        self.jump_to_folder(os.path.abspath(os.path.dirname(self.selected_folder)))

    def jump_to_folder(self,folder):
        self.selected_folder = folder
        self.adr_bar.updateAddressBar(pathlib.PurePath(folder))
        self.treeview.setRootIndex(self.proxy_model.mapFromSource(self.fileSystemModel.index(self.selected_folder)))
    
    def filter_files(self):
        filter_text = self.filter_input.text().strip()
        self.proxy_model.setFilterString(filter_text)
        if self.reader is not None and self.auto_reader is False:
            self.proxy_model.setExtensionFilter(self.reader.EXTENSIONS)
        else:
            self.proxy_model.setExtensionFilter(())
    
    def load_selected_file(self):
        idx = self.treeview.currentIndex()
        source_index = self.proxy_model.mapToSource(idx)
        indexItem = self.fileSystemModel.index(source_index.row(), 0, source_index.parent())
        self.filepath = self.fileSystemModel.filePath(indexItem)
        if not self.fileSystemModel.isDir(indexItem):
            error = self._try_load()
            if error:
                error()
                self.data = None
                self.Outputs.data.send(None)
                self.infolabel.setText("No data.")

    def on_double_click(self):
        idx = self.treeview.currentIndex()
        source_index = self.proxy_model.mapToSource(idx)
        indexItem = self.fileSystemModel.index(source_index.row(), 0, source_index.parent())

        if self.proxy_model.sourceModel().isDir(source_index):
            self.jump_to_folder(self.fileSystemModel.filePath(indexItem))
            # self.treeview.setRootIndex(self.proxy_model.mapFromSource(self.fileSystemModel.index(self.selected_folder)))

    def _try_load(self):

        self.clear_messages()

        if self.filepath and not os.path.exists(self.filepath):
            return self.Error.file_not_found
        
        if self.auto_reader: # if no reader is specified than try autofind it based of file extension
            try:
                self.reader_combo.setCurrentIndex(0)
                reader = FileFormat.get_reader(self.filepath)
                qname = reader.qualified_name()
                self.reader = class_from_qualified_name(qname)
            except:
                return self.Error.missing_reader
            
        if os.path.splitext(self.filepath)[1] in self.reader.EXTENSIONS:
           reader = self.reader(self.filepath)

        if reader is self.NoFileSelected:
            self.Outputs.data.send(None)
            return None

        with catch_warnings(record=True) as warnings:
            try:
                data = reader.read()
            except Exception as ex:
                log.exception(ex)
                return lambda x=ex: self.Error.unknown(str(x))
            if warnings:
                self.Warning.load_warning(warnings[-1].message.args[0])

        if os.path.getsize(self.filepath) > self.SIZE_LIMIT:
                return self.Warning.file_too_big

        self.infolabel.setText(self._describe(data))

        add_origin(data, self.filepath)
        self.data = data
        self.openContext(data.domain)
        self.Outputs.data.send(self.data)

        return None
    
########## GET THE READERS ##########
    def get_described_reader(self):
        """Return reader instance that reads the file given by the read description.

        Parameters
        ----------
        filename : str

        Returns
        -------
        FileFormat
        """
        try:
            reader = [f for f in FileFormat.formats
                    if getattr(f, "DESCRIPTION", None) == self.reader_description and getattr(f, "EXTENSIONS", None)]
            qname = reader[0].qualified_name()
            reader_class = class_from_qualified_name(qname)
            return reader_class
        except:
            raise IOError(f'No readers for {self.reader_description} files.')

    def select_reader(self, n):
        if n == 0:  # default
            self.auto_reader = True
        elif n <= len(self.available_readers):
            self.auto_reader = False
            self.reader = self.available_readers[n - 1]
        else:  # the rest include just qualified names
            self.auto_reader = False
            reader = self.reader_combo.itemText(n)
            qname = reader[0].qualified_name()
            self.reader = class_from_qualified_name(qname)

        self.filter_files()

    def _initialize_reader_combo(self):
        self.reader_combo.clear()
        filters = [format_filter(f) for f in self.available_readers]
        self.reader_combo.addItems([DEFAULT_READER_TEXT] + filters)
        self.reader_combo.setCurrentIndex(0)
        self.auto_reader = True
        # self.reader_combo.setDisabled(True)
        # additional readers may be added in self._get_reader()

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
    WidgetPreview(OWFileBrowser).run()