import numpy as np
import pyqtgraph as pg
import bottleneck as bn

from scipy.ndimage import sobel
# from scipy.ndimage.interpolation import shift
from scipy.ndimage import shift, fourier_shift
from skimage.registration import phase_cross_correlation

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel

import Orange.data
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings
from Orange.widgets.visualize.utils.plotutils import PlotWidget

from orangecontrib.spectroscopy.data import getx, build_spec_table
from orangecontrib.spectroscopy.io.util import _spectra_from_image
from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange
from orangecontrib.spectroscopy.utils import NanInsideHypercube, InvalidAxisException, \
    get_hypercube, get_ndim_hyperspec


# the following line imports the copied code so that
# we do not need to depend on scikit-learn
from orangecontrib.spectroscopy.utils.skimage.register_translation import register_translation
from orangecontrib.spectroscopy.widgets.owspectra import InteractiveViewBox

from orangecontrib.spectroscopy.utils import (
    InvalidAxisException,
    values_to_linspace,
    index_values,
)
# instead of from skimage.feature import register_translation

# stack alignment code originally from: https://github.com/jpacold/STXM_live

class NotMatchingFeatures(Exception):
    pass

class MissingReference(Exception):
    pass

class RegisterTranslation:

    def __init__(self, upsample_factor=1):
        self.upsample_factor = upsample_factor

    def __call__(self, base, shifted):
        """Return the shift (in each axis) needed to align to the base.
        Shift down and right are positive. First coordinate belongs to
        the first axis (rows in numpy)."""
        # s, _, _ = register_translation(base, shifted, upsample_factor=self.upsample_factor)
        s, _, _ = phase_cross_correlation(base, shifted, upsample_factor=self.upsample_factor)
        return s


def shift_fill(img, sh, fill=np.nan):
    """Shift and fill invalid positions"""
    aligned = shift(img, sh, mode='nearest')

    (u, v) = img.shape

    shifty = int(round(sh[0]))
    aligned[:max(0, shifty), :] = fill
    aligned[min(u, u+shifty):, :] = fill

    shiftx = int(round(sh[1]))
    aligned[:, :max(0, shiftx)] = fill
    aligned[:, min(v, v+shiftx):] = fill

    return aligned


def alignstack(raw, shiftfn, ref_frame_num=0, filterfn=lambda x: x):
    """Align to the first image"""
    shifts = calculate_stack_shifts(raw, shiftfn, ref_frame_num=ref_frame_num,
                                    filterfn=filterfn)
    aligned = alignstack_with_shifts(raw, shifts)

    return shifts, aligned

def calculate_stack_shifts(raw, shiftfn, ref_frame_num=0, filterfn=lambda x: x):
    """Calculate the shifts for each image in the stack"""
    base = filterfn(raw[ref_frame_num])
    shifts = []

    for i, image in enumerate(raw):
        if i != ref_frame_num:
            shifts.append(shiftfn(base, filterfn(image)))
        else:
            shifts.append((0, 0))
    shifts = np.array(shifts)

    return shifts

def alignstack_with_shifts(raw, shifts):
    """Aligns the stack using the provided shifts"""
    aligned = np.zeros((len(raw),) + raw[0].shape, dtype=raw[0].dtype)
    for k in range(len(raw)):
        aligned[k] = shift_fill(raw[k], shifts[k])

    return aligned

# -------------------------------------------------------------------------------------------------------
# Copied functions from orangecontrib.snom.preprocess.utils to extract images from datatable
# -------------------------------------------------------------------------------------------------------
class NoComputeValue:
    def __call__(self, data):
        return np.full(len(data), np.nan)
    
# Copied from orangecontrib.snom.preprocess.utils
def _prepare_domain_for_image(data, image_opts):
    at = data.domain[image_opts["attr_value"]].copy(compute_value=NoComputeValue())
    return domain_with_single_attribute_in_x(at, data.domain)

# Copied from orangecontrib.snom.preprocess.utils
def _prepare_table_for_image(data, image_opts):
    odata = data
    domain = _prepare_domain_for_image(data, image_opts)
    data = data.transform(domain)
    if len(data):
        with data.unlocked(data.X):
            data.X[:, 0] = odata.get_column(image_opts["attr_value"], copy=True)
    return data

# Copied from orangecontrib.snom.preprocess.utils
def _image_from_table(data, image_opts):
    hypercube, _, indices = get_ndim_hyperspec(
        data, (image_opts["attr_y"], image_opts["attr_x"])
    )
    return hypercube[:, :, 0], indices

# Copied from orangecontrib.snom.preprocess.utils
def domain_with_single_attribute_in_x(attribute, domain):
    """Create a domain with only the attribute in domain.attributes and ensure
    that the same attribute is removed from metas and class_vars if it was present
    there."""
    class_vars = [a for a in domain.class_vars if a.name != attribute.name]
    metas = [a for a in domain.metas if a.name != attribute.name]
    return Domain([attribute], class_vars, metas)

# Copied from orangecontrib.snom.preprocess.utils
def axes_to_ndim_linspace(coordinates):
    # modified to avoid domains as much as possible
    ls = []
    indices = []

    for i in range(coordinates.shape[1]):
        coor = coordinates[:, i]
        lsa = values_to_linspace(coor)
        if lsa is None:
            raise InvalidAxisException(i)
        ls.append(lsa)
        indices.append(index_values(coor, lsa))

    return ls, tuple(indices)
# Copied from orangecontrib.snom.preprocess.utils
def get_ndim_hyperspec(data, attrs):
    # mostly copied from orangecontrib.spectroscopy.utils,
    # but returns the indices too
    # also avoid Orange domains as much as possible
    coordinates = np.hstack([data.get_column(a).reshape(-1, 1) for a in attrs])

    ls, indices = axes_to_ndim_linspace(coordinates)

    # set data
    new_shape = tuple([lsa[2] for lsa in ls]) + (data.X.shape[1],)
    hyperspec = np.ones(new_shape) * np.nan

    hyperspec[indices] = data.X

    return hyperspec, ls, indices

# Need to modify to calculate shift and apply for the images
class RegisterDriftsToFeatureAttributes:

    def __init__(self, ref_frame_num=0, upsample_factor=1, filterfn=lambda x: x):
        self.shiftfn = RegisterTranslation(upsample_factor=upsample_factor)
        self.ref_frame_num = ref_frame_num
        self.filterfn = filterfn

    def __call__(self, data, image_opts, refdata=None):

        stackdata = data if refdata is None else refdata
        attrs_to_run = [v for v in stackdata.domain.attributes]
        attrnames_to_mark = [v.name for v in data.domain.attributes]
        
        # First get the template image
        image_opts["attr_value"] = attrs_to_run[self.ref_frame_num].name
        template_image_table = _prepare_table_for_image(stackdata, image_opts)
        template_image, _ = _image_from_table(template_image_table, image_opts)

        if bn.anynan(template_image):
            raise NanInsideHypercube(True)

        # Run through all the images, calculate shifts and asign then to the featureattributes
        for j, attr in enumerate(attrs_to_run):
            image_opts["attr_value"] = attr.name
            next_image_table = _prepare_table_for_image(stackdata, image_opts)
            next_image, _ = _image_from_table(next_image_table, image_opts)
            # Check for NaN values
            if bn.anynan(next_image):
                raise NanInsideHypercube(True)
            # Calculate the shift
            s = self.shiftfn(self.filterfn(template_image.T), self.filterfn(next_image.T))
            # Assign the shift to the attributes of the selected feature column
            if attr.name in attrnames_to_mark:
                idx = attrnames_to_mark.index(attr.name)
                data.domain.attributes[idx].attributes["shift"] = s
            else:
                raise NotMatchingFeatures(attr.name)

        return data
    
class ShiftRetriever:
    """Get the shifts assigned to the feature attributes if available"""
    def __call__(self, data):
        shifts = []
        wn = []
        attrs_with_shifts = [v for v in data.domain.attributes if "shift" in list(v.attributes.keys())]
        for j, attr in enumerate(attrs_with_shifts):
            s = attr.attributes["shift"]
            shifts.append(s)
            wn.append(float(attr.name))

        if shifts is not None and wn is not None:   
            shifts = np.array(shifts)
            wn = np.array(wn)

        return shifts, wn

class ShiftCorrector:
    """Correct the images for each feature using the shift values stored in the feature attributes and returns the aligned datatable"""
    def __call__(self, data, image_opts):
        # Only correct the feature image if the shift is available
        attrs_to_corect = [v for v in data.domain.attributes if "shift" in list(v.attributes.keys())]
        aligned_stack = None
        shifts = []

        for j, attr in enumerate(attrs_to_corect):
            image_opts["attr_value"] = attr.name
            image_table = _prepare_table_for_image(data, image_opts)
            image, _ = _image_from_table(image_table, image_opts)
            if aligned_stack is None:
                aligned_stack = np.zeros((len(attrs_to_corect),) + image.T.shape, dtype=image.dtype)
            # Correct each image and biuld a hyperstack
            shift = attr.attributes["shift"]
            shifts.append(shift)
            aligned_stack[j] = shift_fill(image.T, shift)

        if shifts is not None:   
            shifts = np.array(shifts)

        # Crop them and calculate new coordinates (copied from old code)
        _, (lsy, lsx), _ = get_ndim_hyperspec(data, (image_opts["attr_y"], image_opts["attr_x"]))

        xmin, ymin = shifts[:, 0].min(), shifts[:, 1].min()
        xmax, ymax = shifts[:, 0].max(), shifts[:, 1].max()
        xmin, xmax = int(round(xmin)), int(round(xmax))
        ymin, ymax = int(round(ymin)), int(round(ymax))

        shape = (image.shape[0],image.shape[1],len(attrs_to_corect))
        slicex = slice(max(xmax, 0), min(shape[1], shape[1]+xmin))
        slicey = slice(max(ymax, 0), min(shape[0], shape[0]+ymin))
        cropped = np.array(aligned_stack).T[slicey, slicex]

        # transform numpy array Orange.data.Table
        return shifts, build_spec_table(*_spectra_from_image(cropped,
                                                            getx(data),
                                                            np.linspace(*lsx)[slicex],
                                                            np.linspace(*lsy)[slicey]))

def process_stack(data, xat, yat, upsample_factor=100, use_sobel=False, ref_frame_num=0, refdata=None):

    calculate_shift = RegisterTranslation(upsample_factor=upsample_factor)
    filterfn = sobel if use_sobel else lambda x: x

    hypercube, lsx, lsy = get_hypercube(data, xat, yat)
    if bn.anynan(hypercube):
        raise NanInsideHypercube(True)
    
    if refdata is None:
        shifts, aligned_stack = alignstack(hypercube.T,
                                        shiftfn=calculate_shift,
                                        ref_frame_num=ref_frame_num,
                                        filterfn=filterfn)
    else:
        # check the attribute thing later
        hypercube_ref, lsx_ref, lsy_ref = get_hypercube(refdata, xat, yat)
        if bn.anynan(hypercube_ref):
            raise NanInsideHypercube(True)
        shifts = calculate_stack_shifts(hypercube_ref.T,
                                        shiftfn=calculate_shift, 
                                        ref_frame_num=ref_frame_num,
                                        filterfn=filterfn)
        aligned_stack = alignstack_with_shifts(hypercube.T, shifts)

    xmin, ymin = shifts[:, 0].min(), shifts[:, 1].min()
    xmax, ymax = shifts[:, 0].max(), shifts[:, 1].max()
    xmin, xmax = int(round(xmin)), int(round(xmax))
    ymin, ymax = int(round(ymin)), int(round(ymax))

    shape = hypercube.shape
    slicex = slice(max(xmax, 0), min(shape[1], shape[1]+xmin))
    slicey = slice(max(ymax, 0), min(shape[0], shape[0]+ymin))
    cropped = np.array(aligned_stack).T[slicey, slicex]

    # transform numpy array back to Orange.data.Table
    return shifts, build_spec_table(*_spectra_from_image(cropped,
                                                         getx(data),
                                                         np.linspace(*lsx)[slicex],
                                                         np.linspace(*lsy)[slicey]))

def process_datatable(data, image_opts, upsample_factor=100, use_sobel=False, ref_frame_num=0, refdata=None):

    filterfn = sobel if use_sobel else lambda x: x

    shift_calculator = RegisterDriftsToFeatureAttributes(ref_frame_num=ref_frame_num, upsample_factor=upsample_factor, filterfn=filterfn)
    data = shift_calculator(data, image_opts, refdata=refdata)
    shift_getter = ShiftRetriever()
    shifts, wn = shift_getter(data)
    shift_corrector = ShiftCorrector()
    _, new_stack = shift_corrector(data, image_opts)

    return new_stack, shifts, wn


class OWStackAlign(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Align Stack"

    # Short widget description
    description = (
        "Aligns and crops a stack of images using various methods.")

    icon = "icons/stackalign.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Stack of images", Table, default=True)
        refdata = Input("Reference images", Table, default=True)

    class Outputs:
        newstack = Output("Aligned image stack", Table, default=True)

    class Error(OWWidget.Error):
        nan_in_image = Msg("Unknown values within images: {} unknowns")
        invalid_axis = Msg("Invalid axis: {}")
        missing_features = Msg("Data and reference data have different features.")
        no_refdata = Msg("No reference data provided. Connect a reference table.")

    autocommit = settings.Setting(True)

    want_main_area = True
    want_control_area = True
    resizing_enabled = False

    settingsHandler = DomainContextHandler()
    use_refinput = settings.Setting(False)

    sobel_filter = settings.Setting(False)
    attr_x = ContextSetting(None, exclude_attributes=True)
    attr_y = ContextSetting(None, exclude_attributes=True)
    upscale_factor = settings.Setting(1)
    ref_frame_num = settings.Setting(1)

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.controlArea, "Axes")

        common_options = dict(labelWidth=50, orientation=Qt.Horizontal,
                              sendSelectedValue=True)
        self.xy_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                    valid_types=ContinuousVariable)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self._update_attr,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self._update_attr,
            model=self.xy_model, **common_options)

        self.contextAboutToBeOpened.connect(self._init_interface_data)

        refbox = gui.widgetBox(self.controlArea, "Tracking images")
        gui.checkBox(refbox, self, "use_refinput",
                     label="Use reference images",
                     callback=self._ref_frame_changed)
        
        box = gui.widgetBox(self.controlArea, "Parameters")

        gui.checkBox(box, self, "sobel_filter",
                     label="Use sobel filter",
                     callback=self._sobel_changed)
        gui.separator(box)
        hbox1 = gui.hBox(box)
        self.le_upscale = lineEditIntRange(box, self, "upscale_factor", bottom=1, default=1,
                                    callback=self._update_attr)
        hbox1.layout().addWidget(QLabel("Upscale factor:", self))
        hbox1.layout().addWidget(self.le_upscale)

        self.le1 = lineEditIntRange(box, self, "ref_frame_num", bottom=1, default=1,
                                    callback=self._ref_frame_changed)
        hbox2 = gui.hBox(box)
        hbox2.layout().addWidget(QLabel("Reference frame:", self))
        hbox2.layout().addWidget(self.le1)

        gui.rubber(self.controlArea)

        plot_box = gui.widgetBox(self.mainArea, "Shift curves")
        self.plotview = PlotWidget(viewBox=InteractiveViewBox(self))
        self.plotview.plotItem.buttonsHidden = True
        plot_box.layout().addWidget(self.plotview)
        # TODO:  resize widget to make it a bit smaller

        self.data = None
        self.refdata = None

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")

    def _sanitize_ref_frame(self):
        if self.refdata is None:
            if self.ref_frame_num > self.data.X.shape[1]:
                self.ref_frame_num = self.data.X.shape[1]
        else:
            if self.ref_frame_num > self.refdata.X.shape[1]:
                self.ref_frame_num = self.refdata.X.shape[1]

    def _ref_frame_changed(self):
        self._sanitize_ref_frame()
        self.commit.deferred()

    def _sobel_changed(self):
        self.commit.deferred()

    def _init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def image_opts(self):
        return {
            'attr_x': str(self.attr_x),
            'attr_y': str(self.attr_y),
            'attr_value': "",
        }
    
    def _init_interface_data(self, args):
        data = args[0]
        self._init_attr_values(data)

    def _update_attr(self):
        self.commit.deferred()

    @Inputs.data
    def set_data(self, dataset):
        self.closeContext()
        self.openContext(dataset)
        if dataset is not None:
            self.data = dataset
            self._sanitize_ref_frame()
        else:
            self.data = None
        self.Error.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.commit.now()

    @Inputs.refdata
    def set_reference(self, refdataset):
        self.closeContext()
        self.openContext(refdataset)
        if refdataset is not None:
            self.refdata = refdataset
            self._sanitize_ref_frame()
        else:
            self.refdata = None
        self.Error.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.no_refdata.clear()
        self.commit.now()

    @gui.deferred
    def commit(self):
        new_stack = None

        self.Error.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.Error.no_refdata.clear()
        self.plotview.plotItem.clear()

        if self.data and len(self.data.domain.attributes) and self.attr_x and self.attr_y:
            try:

                refdata = self.refdata if self.use_refinput else None
                if self.refdata is None and self.use_refinput is True:
                    raise MissingReference()

                new_stack, shifts, wn = process_datatable(self.data, self.image_opts(), upsample_factor=100, use_sobel=False, ref_frame_num=0, refdata=refdata)

            except NanInsideHypercube as e:
                self.Error.nan_in_image(e.args[0])
            except InvalidAxisException as e:
                self.Error.invalid_axis(e.args[0])
            except NotMatchingFeatures as e:
                self.Error.missing_features(e.args[0])
            except MissingReference:
                self.Error.no_refdata()
            else:
                frames = np.linspace(1, shifts.shape[0], shifts.shape[0])
                self.plotview.plotItem.plot(frames, shifts[:, 0],
                                            pen=pg.mkPen(color=(255, 40, 0), width=3),
                                            symbol='o', symbolBrush=(255, 40, 0), symbolPen='w',
                                            symbolSize=7)
                self.plotview.plotItem.plot(frames, shifts[:, 1],
                                            pen=pg.mkPen(color=(0, 139, 139), width=3),
                                            symbol='o', symbolBrush=(0, 139, 139), symbolPen='w',
                                            symbolSize=7)
                self.plotview.getPlotItem().setLabel('bottom', 'Feature name')
                self.plotview.getPlotItem().setLabel('left', 'Shift / pixel')
                self.plotview.getPlotItem().addLine(self.ref_frame_num,
                                                    pen=pg.mkPen(color=(150, 150, 150), width=3,
                                                                 style=Qt.DashDotDotLine))

        self.Outputs.newstack.send(new_stack)

    def send_report(self):
        self.report_items((
            ("Use sobel filter", str(self.sobel_filter)),
        ))


if __name__ == "__main__":  # pragma: no cover
    from orangecontrib.spectroscopy.tests.test_owalignstack import stxm_diamond
    rev = Orange.data.Table("TGQ1-stepscan-M1A-raw.xyz")
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWStackAlign).run(set_data=rev)
