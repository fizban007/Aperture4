import tables
import numpy
import os
import matplotlib
from matplotlib import ticker
from cycler import cycler
from scipy import special
np=numpy

from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D, CompositeAffine2D
import mpl_toolkits.mplot3d.art3d as art3d

C = 299792458e2  # cm / s !! (not m / s)
E = 4.8032068e-10
ME = 9.1093897e-28
H = 6.6260755e-27  # erg * s
EV_TO_ERG = 1.602177e-12
R0 = E * E / (ME * C * C)


# Function to import custom XML colormaps, of the kind you can download from
# https://sciviscolor.org/colormaps/divergent/
def cmap_from_xml(xml_cmap_file, cmap_name = "mycmap"):
    try:
        from bs4 import BeautifulSoup
    except:
        msg = "func cmap_from_xml requires missing module: "
        msg += "bs4.BeautifulSoup"
        raise ValueError(msg)

    try:
        from matplotlib.colors import LinearSegmentedColormap
    except:
        raise ValueError(msg)
        msg = "func cmap_from_xml requires missing module: "
        msg += "matplotlib.colors.LinearSegmentedColormap"

    try:
        from pathlib import Path
    except:
        msg = "func cmap_from_xml requires missing module: "
        msg = "pathlib.Path; are you running on Python 3?"
        raise ValueError(msg)

    path = Path(xml_cmap_file)
    txt = path.read_text()
    soup = BeautifulSoup(txt, "html.parser")
    points = soup.find_all("point")

    levels = np.array( [ float(points[i].get("x")) for i in range(len(points))] )
    colors = np.array( [[float(points[i].get("r")), float(points[i].get("g")),
                        float(points[i].get("b"))] for i in range(len(points))] )

    return LinearSegmentedColormap.from_list(cmap_name, list(zip(levels, colors)))


def stringify(num, digits = 1):
    if digits is None or digits == "auto":
        numstr = "%g" % num
    else:
        numstr = "%." + str(digits) + "g"
        numstr = numstr % num
    return numstr


# num is, e.g., 3.14159e10
# Converts num to a pretty string such that, typeset using matplotlib,
# you'll get something that, in LaTeX, would be more like
#     3.14 \times 10^10
# digits: the number of digits to include in the significant/mantissa (not the exponent)
def prettyNum(num, digits = 1):
    numstr = stringify(num, digits = digits)
    if "e" in numstr:
        exponent = int(np.floor(np.log10(num)))
        reduced = num / 10**exponent
        if digits == 0:
            numstr = "10^{%d}" % exponent
        else:
            numstr = stringify(reduced, digits = digits)
            numstr += " \\times "
            numstr += "10^{%d}" % exponent
    return numstr


def getArgsAndKwargs(allArgs): #{
    """
    Take a list of strings (e.g., command line arguments)
    and split them into args and kwargs -- a kwarg has an equals sign in
    it, e.g., "name=val".  Returns the args as they are, in a list, and
    return a dictionary of kwargs, converting the value through evaluation.
    """
    args = []
    kwargs = dict()
    for a in allArgs: #{
        if '=' in a: #{
            parts = a.split("=")
            k = parts[0]
            vStr = '='.join(parts[1:])
            #k,vStr = a.split("=")
            try: #{
                # don't let local variables (like k and a and vStr) get evaluated
                #v = eval(vStr)
                v = eval(vStr, dict(), dict())
            except: #}{
                v = vStr
                #}
            kwargs[k] = v
        else: #}{
            args.append(a)
        #}
    #}
    return (args, kwargs)
#}


# Find value(s) of x where y crosses 0
# x and y are 1D arrays of the same length
# mode - The direction of the zero-crossing.
#   One of ["both", "rising", "falling"]
# ret_index - Returns array indices of zero-crossings as a second return value
#   That is, return a list of indices i such that y(x) crosses zero at some x
#   such that x[i] <= x < x[i+1]
def getZeroLocations(x, y, mode = "both", ret_index = False):
    validModes = ["both", "rising", "falling"]
    if not (mode in validModes):
        raise ValueError("'mode' must be one of " + validModes)
    if len(x) != len(y):
        raise ValueError("x & y must have matching lengths")

    yright = y[1:]
    yleft  = y[:-1]
    xright = x[1:]
    xleft = x[:-1]
    if mode == "both":
        idx = np.where( ((yleft < 0) & (yright >= 0)) |
                        ((yleft > 0) & (yright <= 0)) )
    elif mode == "rising":
        idx = np.where( ((yleft < 0) & (yright >= 0)) )
    else:
        idx = np.where( ((yleft > 0) & (yright <= 0)) )

    dx = xright[idx] - xleft[idx]  # x[1] - x[0]
    xloc = x[idx] - yleft[idx] * dx / (yright[idx] - yleft[idx])

    if ret_index:
        return xloc, idx[0]
    else:
        return xloc


# These pruned locator classes implement matplotlib's MaxNLocator's
# prune keyword more robustly.
#     https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.MaxNLocator
#
# prune can be one of ['upper', 'lower', 'both', None(default)]
# and specifies, respectively, where the upper, lower, upper & lower, or neither
# tick labels returned by the parent locator class should be discarded.
# This is useful for stacked plots when the upper/lower labels can often
# overlap.
#
# Additionally, one can specify, if known, the min/max limits of this locator's
# axis as will be displayed on the final plot. Supplying these in addition to
# 'prune' can more firmly guarantee that the pruning will have the desired
# effect.
#
# Finally, if you supply the 'buffer' with the 'limits' and 'prune',
# you remove ticks on the left/right/both sides of the axes if they are
# within a fraction 'buffer' of the axes length of the border.
#
# Example: limits = [0, 2], buffer = 0.1, prune = None
#   Only ticks within 0+(2-0)*0.1 = 0.2 and 2-(2-0)*0.1=1.8 are retained
# Example: limits = None, buffer = 0.0, prune = "right"
#   The rightmost autogenerated tick is always removed
class PrunedLocator(ticker.AutoLocator):

    def __init__(self, *args, **kwargs):
        prune = None
        if "prune" in kwargs:
            prune = kwargs.pop("prune")
        lims = None
        if "limits" in kwargs:
            lims = kwargs.pop("limits")
        buf = 0.0
        if "buffer" in kwargs:
            buf = kwargs.pop("buffer")

        super().__init__(*args, **kwargs)
        
        prune_options = ["upper", "lower", "both", None]
        self.prune = prune
        if not (self.prune in prune_options):
            msg = "Option 'prune' to PrunedLocator must be one of "
            msg += "%s; got %s" % (pruned_options, self.prune)
            raise ValueError(msg)

        self.lims = lims
        self.buf  = buf

    def __call__(self, *args, **kwargs):
        locs = super().__call__(*args, **kwargs)
        prune = self.prune
        lims  = self.lims
        buf   = self.buf

        if lims is not None:
            minloc = lims[0]
            if prune == "lower" or prune == "both":
                minloc = lims[0] + (lims[1] - lims[0]) * buf
            maxloc = lims[1]
            if prune == "upper" or prune == "both":
                maxloc = lims[1] - (lims[1] - lims[0]) * buf
            locs = locs[(locs >= minloc) & (locs <= maxloc)]

        if prune is not None and buf == 0.0 and lims is None:
            # Force removal of the outermost limits only if buf == 0
            if prune == "upper":
                locs = locs[:-1]
            elif prune == "lower":
                locs = locs[1:]
            elif prune == "both":
                locs = locs[1:-1]
            else:
                msg = "PrunedLocator does not implement prune = %s" % prune
                raise NotImplementedError(msg)
        return locs


# Basically the same as above, but for log axis scales
class PrunedLogLocator(matplotlib.ticker.LogLocator):

    def __init__(self, *args, **kwargs):
        prune = None
        if "prune" in kwargs:
            prune = kwargs.pop("prune")
        lims = None
        if "limits" in kwargs:
            lims = kwargs.pop("limits")

        super().__init__(*args, **kwargs)
        
        prune_options = ["upper", "lower", "both", None]
        self.prune = prune
        if not (self.prune in prune_options):
            msg = "Option 'prune' to PrunedLocator must be one of "
            msg += "%s; got %s" % (pruned_options, self.prune)
            raise ValueError(msg)

        self.lims = lims

    def __call__(self, *args, **kwargs):
        locs = super().__call__(*args, **kwargs)
        prune = self.prune
        lims  = self.lims

        if lims is not None:
            locs = locs[(locs >= lims[0]) & (locs <= lims[-1])]

        if prune is not None:
            if prune == "upper":
                locs = locs[:-1]
            elif prune == "lower":
                locs = locs[1:]
            elif prune == "both":
                locs = locs[1:-1]
            else:
                msg = "PrunedLocator does not implement prune = %s" % prune
                raise NotImplementedError(msg)
        return locs


def readArrayFromHdf5(h5FileName, datasetName, slices = None,
  shapeOnly = False, verbose=False, alternateDatasetName = None): #{
  """
  Returns a (numpy float64) array of the field values
  from an hdf5 dataset

  If slices is set, this will retrieve the requested slice of a field;
  e.g., if the field is 4x5x3, then 
  slices = (slice(None), slice(2,5), slice(None)) will return a 
  4x3x3 subset.

  If shapeOnly, returns the field shape rather than the field, and 
    slices is ignored.
  """
  try:
    file = tables.open_file(h5FileName,'r')
    getNode = file.get_node
  except:
    try:
      file = tables.openFile(h5FileName,'r')
      getNode = file.getNode
    except:
      msg = "\n\nError: Failed to open hdf5 file: '" + h5FileName + "'"
      msg += " in " + os.getcwd() + " (is it an hdf5 file?)\n"
      raise ValueError(msg)

  groupName = '/' + datasetName

  if alternateDatasetName is not None and datasetName not in getNode("/"):
    if alternateDatasetName not in getNode("/"):
      msg = "Neither " + datasetName + " nor " + alternateDatasetName 
      msg += " names a datset in " + h5FileName
      raise ValueError(msg)
    groupName = '/' + alternateDatasetName

  try:
    if shapeOnly: #{
      res = getNode(groupName).shape
    else: #}{
      if (slices is None):
        res = numpy.array( getNode(groupName) )
      else:
        if verbose:
          print("Getting slices =", slices, "for", h5FileName)
        res = numpy.array( getNode(groupName)[tuple(slices)] )
        #print "Got slices =", slices, "->", res.shape
    #}
  except:
    file.close()
    raise
    msg = "\n\nError: Failed to open dataset '" + groupName + "'"
    msg += " in hdf5 file: '" + h5FileName + "'\n"
    raise ValueError(msg)
  file.close()

  return res
#}


# Smooth a field in 2d according to the filter matrix.
#     |1 2 1|
# M = |2 4 2|
#     |1 2 1|
# pdim1, pdim2 -- Whether the first, second dimensions of 'field' are periodic,
#   respectively.
def smoothField2d(field, pdim1 = True, pdim2 = True):
    expanded = np.zeros( (field.shape[0]+2, field.shape[1]+2) )
    normalization = 16.*np.ones(field.shape)

    expanded[1:-1,1:-1] = field
    if pdim1:
        expanded[0,1:-1] = field[-1,:]
        expanded[-1,1:-1] = field[0,:]
    else:
        normalization[0,:] = 12.
        normalization[-1,:] = 12.

    if pdim2:
        expanded[1:-1,0] = field[:,-1]
        expanded[1:-1,-1] = field[:,0]
    else:
        normalization[:,0] = 12.
        normalization[:,-1] = 12.

    if pdim1 and pdim2:
        # Fill in the corners according to PBCs
        expanded[0,0] = field[-1,-1]
        expanded[0,-1] = field[-1,0]
        expanded[-1,0] = field[0,-1]
        expanded[-1,-1] = field[0,0]

    # If only one of the dimensions is periodic, the corners are already
    # correctly set to 16 or 12. Only if BOTH dimensions are not periodic do we
    # set the normalization at the corners to 9.
    if not pdim1 and not pdim2:
        normalization[0,0] = 9.
        normalization[0,-1] = 9.
        normalization[-1,0] = 9.
        normalization[-1,-1] = 9.

    return 1./normalization * (1.0*expanded[:-2,:-2] + 2.0*expanded[1:-1,:-2]  + 1.0*expanded[2:,:-2]
                             +  2.0*expanded[:-2,1:-1]+ 4.0*expanded[1:-1,1:-1] + 2.0*expanded[2:,1:-1]
                             +  1.0*expanded[:-2,2:]  + 2.0*expanded[1:-1,2:]   + 1.0*expanded[2:,2:])


# Calculate Az from Bx and By
# Bx,By are both assumed to be 2D arrays
# x,y are assumed to be 1D arrays such that Bx[i,j] and By[i,j] are known
# at (x[i], y[j]).
def calc_Az(x,y,Bx,By):
    (nx, ny) = Bx.shape[:2]
    dx = numpy.diff(x).mean()
    dy = numpy.diff(y).mean()

    # Bx = d_y Az, By = -d_x Az

    Az = numpy.zeros((nx+1, ny+1), dtype = Bx.dtype)
    # First integrate from (0,0) to (0,y) to find Az(0,y) for all y
    Az[0,1:ny+1] = -Bx[0,:ny].cumsum() * dy 
    # Integrate from Az(0,y) to Az(x,y)
    Az[1:nx+1,:ny] = Az[numpy.newaxis,0,:ny] + By[:nx,:ny].cumsum(axis=0)*dx
    # Az[1:, ny] has not been set
    # Assuming rotationless field, the top row of By is
    ByTop = Bx[1:nx, ny-1] - Bx[0:nx-1, ny-1] + By[0:nx-1, ny-1]
    Az[1:nx,ny] = Az[0,ny] + ByTop[:nx-1].cumsum() * dx 
    # Now only Az[nx,ny] is unset -- we don't know how to find it
    # without invoking boundary conditions. Assume periodic in x
    Az[nx,ny] = Az[0,ny]

    # Revert Az back to same grid as Bx/By
    Az = 0.25 * (Az[1:,1:] + Az[:-1, :-1] + Az[1:, :-1] + Az[:-1, 1:])

    return Az


FKN_SMALL_ARG_THRESHOLD = 0.01
def fkn(q):
    # Magical logic that allows this function to work both on plain numbers
    # and arrays alike.
    q = np.asarray(q)
    scalar_input = False
    if q.ndim == 0:
        q = q[None]  # Makes q 1D
        scalar_input = True

    # Use the small-argument approximation to fkn where q is small.
    # This construction does not save computational time, but may eliminate
    # potential numerical issues that come in at small arguments.
    out = np.where(q < FKN_SMALL_ARG_THRESHOLD,
        1. - 63. * q / 40, _fkn_helper(q))
    
    if scalar_input:
        return np.squeeze(out)
    return out


# Helper function containing the complicated expression for fkn
# Assumes the argument is an array.
def _fkn_helper(q):
    t1 = (0.5 * q + 6. + 6. / q) * np.log(1. + q)
    t2 = -1. / (1. + q)**2 * (11. * q**3 / 12. + 6. * q**2 + 9. * q + 4.)
    t3 = -2. + 2. * special.spence(1. + q)
    out = 9. / q**3 * (t1 + t2 + t3)
    return np.where(q != np.inf, out, np.zeros(q.shape))


def fkn_asymp(q):
    return 9. / (2. * q**2) * (np.log(q) - 11. / 6)


def get_figsize_onecol(plt):
    return plt.rcParams["figure.figsize"]


# This preset is for the apj.mplstyle style sheet.
# If you are using that style sheet, the default figure width retrieved
# from this function will match \textwidth for apj journal articles.
# The aspect ratio from the onecol size is maintained to calculate the vertical
# size.
def get_figsize_twocol(plt):
    return tuple(2.119 * np.array(plt.rcParms["figure.figsize"]))


# Return unit vectors b1 = (b1x, b1y, b1z), b2 = (b2x, b2y, b2z) such that
# the vectors b1, b2, and nhat form an orthonormal triad. I.e.
#
# norm(b1) == norm(b2) == norm(nhat) == 1
# cross(b1, b2) == nhat
#
# If nhat = (0, 0, 1), then return (1, 0, 0), (0, 1, 0)
# Otherwise, return \hat{phi}, -\hat{theta} where \hat{phi} and \hat{theta} are
# the usual spherical coordinate unit vectors
def orthonormal_triad(nhatx, nhaty, nhatz):
    if nhatz > 0.99999:
        return (1., 0., 0.), (0., 1., 0.)

    theta = np.arccos(nhatz)
    phi = np.arctan2(nhaty, nhatx)

    thx = np.cos(theta) * np.cos(phi)
    thy = np.cos(theta) * np.sin(phi)
    thz = -np.sin(theta)

    phx = -np.sin(phi)
    phy = np.cos(phi)
    phz = 0.0

    return (phx, phy, phz), (-thx, -thy, -thz)


# Computes the spectral from the zpostprocess.zradcalc.Spectrum object, spec
# The cutoff is computed as d/dp (<\gamma^{p+1}> / <\gamma^p>) evaluated
# at some central p0.
# Spec must be 1D
def compute_spec_cut(spec, p0=5.0, eps=0.1):
    phi = p0 + eps
    plo = p0 - eps
    data = spec.get_spec()
    if data.ndim != 1:
        raise ValueError("Got arg. 'spec' with %d dims. Need 1D." % data.ndim)

    xval  = spec.get_axes()[0]
    xedge = spec.get_axes_edges()[0]
    data  = spec.get_spec()

    dx = np.diff(xedge)
    invnorm = 1.0 / np.sum(data * dx)
    mplo   = np.sum(data * xval**(plo    ) * dx) * invnorm
    mplop1 = np.sum(data * xval**(plo + 1) * dx) * invnorm
    mphi   = np.sum(data * xval**(phi    ) * dx) * invnorm
    mphip1 = np.sum(data * xval**(phi + 1) * dx) * invnorm
    
    ratphi = mphip1 / mphi
    ratplo = mplop1 / mplo

    deriv = (ratphi - ratplo) / (2 * eps)

    return deriv


# Stolen from
# https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    # d = np.cross((0, 0, 1), normal) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta


# Stolen from:
# https://matplotlib.org/stable/gallery/mplot3d/pathpatch3d.html#sphx-glr-gallery-mplot3d-pathpatch3d-py
def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    """
    Plots the string *s* on the axes *ax*, with position *xyz*, size *size*,
    and rotation angle *angle*. *zdir* gives the axis which is to be treated as
    the third dimension. *usetex* is a boolean indicating whether the string
    should be run through a LaTeX subprocess or not.  Any additional keyword
    arguments are forwarded to `.transform_path`.
    
    Note: zdir affects the interpretation of xyz.
    """
    x, y, z = xyz
    # if zdir == "y":
    #     xy1, z1 = (x, z), y
    # elif zdir == "x":
    #     xy1, z1 = (y, z), x
    # elif zdir == "z":
    #     xy1, z1 = (x, y), z
    if zdir == "x":
        zdir = (1, 0, 0)
    elif zdir == "-x":
        zdir = (-1, 0, 0)
    elif zdir == "y":
        zdir = (0, 1, 0)
    elif zdir == "-y":
        zdir = (0, -1, 0)
    elif zdir == "z":
        zdir = (0, 0, 1)
    elif zdir == "-z":
        zdir = (0, 0, -1)

    # Convert to a unit vector
    zdir = np.array(zdir)
    zdir = zdir / np.sqrt(np.dot(zdir, zdir))

    b1, b2 = orthonormal_triad(*zdir)
    b1 = np.array(b1)
    b2 = np.array(b2)
    xyz = np.array(xyz)
    xy1 = (np.dot(b1, xyz), np.dot(b2, xyz))
    z1 = np.dot(zdir, xyz)
    zdir = tuple(zdir)
    
    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    # trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])
    trans = Affine2D().rotate(angle).translate(0, 0)
    
    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    bbox = dict(facecolor='w', edgecolor='k', alpha=0.8, boxstyle="round")
    # p1 = ax.text2D(0, 0, s, fontsize=size, bbox=bbox).get_bbox_patch()
    # twindow = p1.get_window_extent()
    # p2 = matplotlib.patches.FancyBboxPatch(
    #     (0, 0), twindow.width, twindow.height,
    #     # transform = Affine2D().rotate(angle),
    #     zorder=4,
    #     facecolor='w', edgecolor='k', alpha=0.8, boxstyle='round')
    # t_start = ax.transData
    # t = Affine2D().rotate(angle).translate(0,0)
    # t_end = t_start + t
    # p2.set_transform(t_end)
    # # p2.set_transform(CompositeAffine2D(Affine2D().rotate(angle), p2.get_transform()))
    # # p2.set_transform(p2.get_transform())
    # # p2 = PathPatch(trans.transform_path(bbox), **kwargs)
    # ax.add_patch(p2)
    # # t_start = p2.get_transform()
    # # t = Affine2D().rotate(angle)
    # # t_end = t + t_start
    # # p2.set_transform(t_end)
    # t_start = p2.get_transform()
    # t = Affine2D().rotate(angle)
    # t_end = t + t_start
    # p2.set_transform(t_end)
    # pathpatch_2d_to_3d(p2, z=0, normal=zdir)
    # pathpatch_translate(p2, xyz)
    ax.add_patch(p1)
    # art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)
    pathpatch_2d_to_3d(p1, z=0, normal=zdir)
    pathpatch_translate(p1, xyz)

