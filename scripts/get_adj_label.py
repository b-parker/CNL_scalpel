import os,sys
import os,sys
import nibabel as nib
from numpy import *
import gdist
#import surfdist as sd
import numpy as np


# NOTE: had trouble with numba format and jit in surfdist, so some functions are copied over with slight modifications below


def translate_src(src, cortex):
    """
    Convert source nodes to new surface (without medial wall).
    """
    src_new = np.array(np.where(np.in1d(cortex, src))[0], dtype=np.int32)

    return src_new


def triangles_keep_cortex(triangles, cortex):
    """
    Remove triangles with nodes not contained in the cortex label array
    """

    # for or each face/triangle keep only those that only contain nodes within the list of cortex nodes
    input_shape = triangles.shape
    triangle_is_in_cortex = np.all(np.reshape(np.in1d(triangles.ravel(), cortex), input_shape), axis=1)

    cortex_triangles_old = np.array(triangles[triangle_is_in_cortex], dtype=np.int32)

    # reassign node index before outputting triangles
    new_index = np.digitize(cortex_triangles_old.ravel(), cortex, right=True)
    cortex_triangles = np.array(np.arange(len(cortex))[new_index].reshape(cortex_triangles_old.shape), dtype=np.int32)

    return cortex_triangles


def surf_keep_cortex(surf, cortex):
    # split surface into vertices and triangles
    vertices, triangles = surf

    # keep only the vertices within the cortex label
    cortex_vertices = np.array(vertices[cortex], dtype='float64')

    # keep only the triangles within the cortex label
    cortex_triangles = triangles_keep_cortex(triangles, cortex)

    return cortex_vertices, cortex_triangles


def dist_calc_matrix(surf, cortex, label_inds_all):
    cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)
    
    n_labels = len(labels)
    dist_mat = zeros((n_labels,n_labels))
    for r1 in arange(n_labels):
        #print('r1',r1,label_inds_all[r1])
        for r2 in arange(n_labels):
            #print('r2',r2,label_inds_all[r2])
            #val1 = gdist.compute_gdist(cortex_vertices, cortex_triangles,
            #                                source_indices = array(label_inds_all[r1]))
            #print('val1',val1)

            val2 = gdist.compute_gdist(cortex_vertices, cortex_triangles,
                                            source_indices = array(label_inds_all[r1]),
                                            target_indices = array(label_inds_all[r2]))
            #print('val2',val2)
            dist_mat[r1,r2] = amin(val2)

    return dist_mat


def getLabelIndices(sub,hemi,labels,cortex):
    label_inds_all = []

    n_labels = len(labels)
    for l in arange(n_labels):
        if type(labels[l]) is list: # pick the first label in list that exits
            label_found = False
            for lab in labels[l]:
                labelfile = '%s/sub-%s/label/%s.%s.label'%(os.environ['SUBJECTS_DIR'],sub,hemi,lab)
                if os.path.exists(labelfile) and not label_found:
                    labelfile_use = labelfile
                    label_found = True
        else: # look for specific label
            labelfile_use = '%s/sub-%s/label/%s.%s.label'%(os.environ['SUBJECTS_DIR'],sub,hemi,labels[l])
        label_inds = nib.freesurfer.io.read_label(labelfile_use, read_scalars=False)
        label_inds_t = translate_src(label_inds,cortex) # exclude medial wall
        label_inds_all.append(label_inds_t)
    
    return label_inds_all


def getDistMatrix(subjects_dir=str, labels=str, sub=str, hemi=str, savedir=str, fmri_prep=False):
    """
    Outputs geodesic distances among all labels for a given sub/hemi
    """
    if fmri_prep == True:
        highres_surface = '%s/sub-%s/ses-%s/anat/sub-%s_ses-%s_hemi-%s_midthickness.surf.gii'%(subjects_dir,sub,sub[-1],sub,sub[-1],hemi[0].upper())
    if fmri_prep == False:
        highres_surface = f'{subjects_dir}/{sub}/surf/{hemi}.pial.surf.gii'
    
    
    giidata = nib.freesurfer.read_geometry(highres_surface)
    print(giidata)
    print(giidata.darrays[1].data)
    giidata2 = np.squeeze(np.asarray([x.data for x in giidata.darrays])) 
    surf = (giidata2[0],giidata2[1])  

    
    if fmri_prep == True:
        cort_file = '%s/sub-%s/label/%s.cortex.label'%(os.environ['SUBJECTS_DIR'],sub,hemi)
    if fmri_prep == False:
       cort_file = f'{subjects_dir}/{sub}/label/{hemi}.cortex.label'
    
    cortex = sort(nib.freesurfer.read_label(cort_file))

    label_inds_all = getLabelIndices(sub,hemi,labels,cortex)

    dist_matrix = dist_calc_matrix(surf,cortex,label_inds_all)
    print('dist_matrix',dist_matrix)

    savetxt('%s/adj-labels-%s.txt'%(savedir,hemi),dist_matrix)


if __name__ == '__main__':

    sub = sys.argv[1]
    hemi = sys.argv[2]
    subjects_dir = sys.argv[3]

    outdir='~/Desktop/cnl/misc'    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    os.environ['SUBJECTS_DIR'] = '~/Desktop/cnl/subjects'
    #fmriprep_derivatives_dir = '/home/weiner/Nora_PFCSulci/Projects/NORA_relmatch_funcNeuroAnat/data/bids/derivatives_v7'
    #lpfc_labels = ['ifs','painfs_any','pmfs_a','pmfs_i','pmfs_p','sfs_a','sfs_p'] + ['prts','lfms','aalf']
    #lpar_labels = ['slos1','sB','pips','mTOS','lTOS','IPS-PO','IPS','cSTS1','cSTS2','cSTS3','aipsJ']
    mpar_labels = [] #['1','2','3','MCGS','POS','prculs','prcus1','prcus2','prcus3','sbps','sps']
    lpfc_labels = ['ifs',['painfs_any','painfs_combined'],'pmfs_a','pmfs_i','pmfs_p','sfs_a','sfs_p'] + ['prts','lfms','aalf']
    lpar_labels = [['slos1','slocs-v','SLOS'],'sB','pips','mTOS',['iTOS','ITOS','lTOS'],'IPS-PO','IPS','cSTS1','cSTS2','cSTS3','aipsJ']
    labels = lpfc_labels + lpar_labels + mpar_labels

    getDistMatrix(subjects_dir,labels,sub,hemi,outdir, fmri_prep=False)

