
from pathlib import Path
from shutil import move
import time

import nibabel as nb
import numpy as np
from sklearn.cluster import KMeans

from src.surface_funcs import get_gyrus, cluster_label_KMeans, get_faces_from_vertices, separate_clusters, find_closest_clusters, find_label_boundary_vertices, find_shortest_path_in_mesh
from src.freesurfer_utils import read_label, write_label, create_freesurfer_ctab, freesurfer_label2annot



write_gyral_cluster_annotation = False
write_shortest_paths_label = False
num_clusters = 500
overwrite_labels = True


subjects = ['100206', '100307', '100408', '100610', '120111']
hemis = ['lh', 'rh']

for sub in subjects:
    for hemi in hemis:
        time_start = time.time()

        print(f'\n\nProcessing subject {sub} \n')
        print(f'Number of clusters: {num_clusters}')
        print('Reading inflated surface and labels \n \n')
        ## Read in inflated surface and get the indices of the vertices 
        inflated_surf = nb.freesurfer.read_geometry(f'/Users/benparker/Desktop/cnl/subjects/{sub}/surf/{hemi}.inflated')
        inflated_points, inflated_faces = inflated_surf[0], inflated_surf[1]
        inflated_ind = np.unique(inflated_faces)

        ## Read in the curvature data, label data, and get gyral vertices
        curv = nb.freesurfer.read_morph_data(f'/Users/benparker/Desktop/cnl/subjects/{sub}/surf/{hemi}.curv')
        label_1_name ='IPS'
        label_2_name ='IPS-PO'  
        subjects_dir = '/Users/benparker/Desktop/cnl/subjects'
        points, faces = nb.freesurfer.read_geometry(f'{subjects_dir}/{sub}/surf/{hemi}.pial')
        label_1_ind, label_1_RAS = read_label(f'{subjects_dir}/{sub}/label/{hemi}.{label_1_name}.label')
        label_2_ind, label_2_RAS = read_label(f'{subjects_dir}/{sub}/label/{hemi}.{label_2_name}.label')
        inflated_gyrus = get_gyrus(inflated_ind, inflated_points, curv)

        ## Cluster the gyral vertices
        print(f'Clustering {num_clusters} gyral vertices \n \n')
        cluster_kmeans = cluster_label_KMeans(inflated_gyrus[0], inflated_gyrus[1], inflated_points, inflated_faces, n_clusters=num_clusters)    

        if write_gyral_cluster_annotation:
            ## Write all clusters as labels
            print(f'Writing {num_clusters} gyral cluster labels and annotation')
            for i in range(num_clusters):
                label_i_ind, label_i_RAS = inflated_gyrus[0][cluster_kmeans == i], inflated_gyrus[1][cluster_kmeans == i]
                gyral_cluster_dir = Path(f"{subjects_dir}/{sub}/label/gyral_clusters")
                if not gyral_cluster_dir.exists():
                    gyral_cluster_dir.mkdir()

                cluster_file_labelname = f'INFLATED_{num_clusters}_gyral_cluster_{i}'
                
                write_label(label_i_ind, label_i_RAS, cluster_file_labelname, 'lh', subject_dir = f"/Users/benparker/Desktop/cnl/subjects/{sub}", overwrite=True)
                move(f"/Users/benparker/Desktop/cnl/subjects/{sub}/label/{hemi}.{cluster_file_labelname}_{i}.label", \
                    f"/Users/benparker/Desktop/cnl/subjects/{sub}/label/gyral_clusters/{hemi}.{cluster_file_labelname}.label")

            ## Write ctab for annotation
            gyral_cluster_label_list = [f'INFLATED_gyral_cluster_{i}' for i in range(500)]
            create_freesurfer_ctab(ctab_name='gyral_clusters', label_list=gyral_cluster_label_list, outdir=gyral_cluster_dir)

            ## Write annotation for gyral cluster annot
            subject_path = f'/Users/benparker/Desktop/cnl/subjects/{sub}'
            ctab_path = f'/Users/benparker/Desktop/cnl/subjects/{sub}/label/gyral_clusters/gyral_clusters.ctab'
            freesurfer_label2annot(subjects_dir=subjects_dir, subject_path=subject_path, label_list=gyral_cluster_label_list, hemi=hemi, ctab_path=ctab_path, annot_name='clustered_gyral')
                    
        
        ## For each label (label_1 and label_2), find boundary points
        label_1_faces = get_faces_from_vertices(faces, label_1_ind)
        label_2_faces = get_faces_from_vertices(faces, label_2_ind)

        ## Find closest clusters of each label

        ## concatenate label_1 and label2
        labels_12_ind = np.concatenate((label_1_ind, label_2_ind))
        labels_12_RAS = np.concatenate((label_1_RAS, label_2_RAS))

        print(f'Clustering {len(labels_12_ind)} vertices from {label_1_name} and {label_2_name}')

        cluster_12_kmeans = cluster_label_KMeans(labels_12_ind, labels_12_RAS, points, faces, n_clusters=15)

        label_1_clusters, label_2_clusters = separate_clusters(cluster_12_kmeans, label_1_ind, label_2_ind, labels_12_ind)
        
        closest_clusters = find_closest_clusters(label_1_RAS, label_1_ind, label_2_RAS, label_2_ind, label_1_clusters, label_2_clusters, sub, subjects_dir, hemi, num_clusters=1)

        ## Find the two closest clusters in each label
        closest_1 = labels_12_ind[np.where(cluster_12_kmeans == int(closest_clusters[0][0]))]
        closest_2 = labels_12_ind[np.where(cluster_12_kmeans == int(closest_clusters[0][1]))]       

            
        ## Find the closest vertices to the boundary of each closest cluster
        label_1_closest_faces = get_faces_from_vertices(faces, closest_1)
        label_2_closest_faces = get_faces_from_vertices(faces, closest_2)

        label_1_closest_boundary_ind = find_label_boundary_vertices(label_1_closest_faces)
        label_2_closest_boundary_ind = find_label_boundary_vertices(label_2_closest_faces)

        ## Get every 15th vertex to keep operation efficient
        label_1_closest_boundary_ind_pruned = label_1_closest_boundary_ind[::15]
        label_2_closest_boundary_ind_pruned = label_2_closest_boundary_ind[::15]

        print(f'Finding shortest paths between {label_1_name} and {label_2_name} boundary vertices')

        ## Shortest paths between clustered boundaries
        shortest_paths = np.array([])
        for i in range(len(label_1_closest_boundary_ind_pruned)):
            for j in range(len(label_2_closest_boundary_ind_pruned)):
                path = find_shortest_path_in_mesh(faces, label_1_closest_boundary_ind_pruned[i], label_2_closest_boundary_ind_pruned[j])
                shortest_paths = np.append(shortest_paths, path)

        shortest_paths = shortest_paths.astype(int)
        shortest_paths_ind, shortest_paths_RAS = shortest_paths, points[list(shortest_paths)]


        ## Write label containting all vertices in shortest paths
        if write_shortest_paths_label:
            print(f'Writing label for shortest paths between {label_1_name} and {label_2_name}')
            write_label(shortest_paths_ind, shortest_paths_RAS, f'INFLATED_shortest_paths_pruned_{label_1_name}_{label_2_name}', hemi, subject_dir = f"/Users/benparker/Desktop/cnl/subjects/{sub}", overwrite=overwrite_labels)
        
        
        ## Get gyral clusters which have a shortest path pass through them
        print(f'Finding gyral clusters intersected by shortest paths')
        shortest_path_clusters_unique = np.unique(cluster_kmeans[np.isin(inflated_gyrus[0], shortest_paths_ind)])
        shortest_path_clusters_unique_ind = inflated_gyrus[0][np.isin(cluster_kmeans, shortest_path_clusters_unique)]
        shortest_path_clusters_unique_RAS = inflated_surf[1][shortest_path_clusters_unique_ind]

        ## Write label for gyral clusters which have a shortest path pass through them
        write_label(shortest_path_clusters_unique_ind, shortest_path_clusters_unique_RAS, f'INFLATED_{num_clusters}_gyral_gap_{label_1_name}_{label_2_name}', hemi, subject_dir = f"/Users/benparker/Desktop/cnl/subjects/{sub}", overwrite=overwrite_labels)
        
        print(f'\n\nFinished processing {sub}')
        print(f'Elapsed time: {time.time() - time_start} \n')

