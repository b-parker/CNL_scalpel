'''
Generates SVG with sulci from subject surfaces.


2017. James Gao and Alex Huth.
'''


import shutil
import numpy as np
import cortex

#RS edited because it wasn't importing!
#from cortex import svgoverlay, polyutils
from scipy import ndimage, sparse
from skimage import morphology
from skimage.measure import approximate_polygon

def main(subject, outfile):
    svgpath = cortex.db.get_paths(subject)['overlays']
    shutil.copyfile(svgpath, outfile)
    overlay = cortex.db.get_overlay(subject)
    overlay.svgfile = outfile
    overlay.add_layer("autosulci")

    flat = compute(subject)
    filt = ndimage.gaussian_filter(flat, 5)

    segments = []
    for p in trace(morphology.skeletonize(filt > 0.001)):
        ap = approximate_polygon(p/2., 1)
        if len(ap) > 3:
            segments.append(ap)

    write_svg(overlay, segments)

def compute(subject):
    depth = cortex.db.get_surfinfo(subject, 'sulcaldepth')
    data = []
    for hemi, sd in zip(['lh', 'rh'], [depth.left, depth.right]):
        pts, polys = cortex.db.get_surf(subject, 'fiducial', hemi)
        surf = polyutils.Surface(pts, polys)

        th = -0.1
        smd = surf.smooth(sd)
        gyri = np.nonzero(smd > th)

        # get laplace operator
        B,D,W,V = surf.laplace_operator
        Dinv = sparse.diags(D**-1)
        L = Dinv.dot(V - W)

        #lap_gydist = L.dot(gydist)
        lap_gydist = L.dot(np.clip(smd, -np.inf, th))

        # clean up edges by setting adjacent points to gyri to 0
        gyri_adj = (surf.adj[gyri].sum(0) > 0).A.squeeze()
        lap_gydist[gyri_adj] = 0
        lap_gydist[lap_gydist > 0] = 0
        lap_gydist = lap_gydist ** 2

        data.append(lap_gydist)
    vert = cortex.Vertex(np.hstack(data), subject)
    return cortex.quickflat.make_flatmap_image(vert, height=2048)[0]

def write_svg(overlay, segments):
    from lxml import etree
    from lxml.builder import E

    svgns = "http://www.w3.org/2000/svg"
    inkns = "http://www.inkscape.org/namespaces/inkscape"
    sodins = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"

    parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)
    svg = etree.parse(overlay.svgfile, parser=parser)
    dest = svgoverlay._find_layer(svgoverlay._find_layer(svg, "autosulci"), "shapes")
    path = etree.SubElement(dest, "{%s}path"%svgns)

    d = ""
    for segment in segments:
        d += " M %d,%d"%(tuple(segment[0][::-1]))
        for pt in segment[1:]:
            d += " %d,%d"%tuple(pt[::-1])
    path.attrib['d'] = d
    path.attrib['fill'] = 'none'
    path.attrib['stroke'] = 'black'

    with open(overlay.svgfile, "wb") as fp:
        fp.write(etree.tostring(svg, pretty_print=True)) # python2.X

def trace(skel):
    skel = skel.copy()
    coords = np.array(np.nonzero(skel)).T
    while len(coords) > 0:
        segment = [coords[0]]
        y, x = segment[-1]
        skel[y, x] = False
        block = skel[y-1:y+2, x-1:x+2]
        while block.sum() > 0:
            coord = [y,x] + np.array(np.nonzero(block)).T[0]-1
            segment.append(coord)
            y, x = coord
            skel[y, x] = False
            block = skel[y-1:y+2, x-1:x+2]

        #we need to go the other direction to make sure we grab a complete segment
        y, x = segment[0]
        block = skel[y-1:y+2, x-1:x+2]
        while block.sum() > 0:
            coord = [y,x] + np.array(np.nonzero(block)).T[0]-1
            segment.insert(0, coord)
            y, x = coord
            skel[y, x] = False
            block = skel[y-1:y+2, x-1:x+2]

        yield np.array(segment)
        coords = np.array(np.nonzero(skel)).T
