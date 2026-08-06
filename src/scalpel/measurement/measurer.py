from __future__ import annotations
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Optional
import numpy as np
from pathlib import Path
import csv


from scalpel.utils import surface_utils

if TYPE_CHECKING:
    from scalpel.subject import ScalpelSubject

class ScalpelMeasurer:
    """
    Class for measuring surface data.
    
    This class provides measurement functionality for a ScalpelSubject,
    including calculation of sulcal depth, surface area, cortical thickness,
    distances between labels, and overlap between labels.
    """
    
    def __init__(self, subject: 'ScalpelSubject'):
        """
        Initialize a ScalpelMeasurer.
        
        Parameters:
        -----------
        subject : ScalpelSubject
            The subject to measure
        """
        self._subject = subject
    
    @property
    def subject(self):
        """Get the associated ScalpelSubject."""
        return self._subject
    

    def calculate_sulcal_depth(self, label_name, depth_pct=8, n_deepest=100, use_n_deepest=True):
        """
        Sulcal depth following Madan (2019) calcSulc: the median distance from the
        deepest sulcal (pial) vertices to the nearest point on the
        ``pial-outer-smoothed`` gyral hull.

        Reference: Madan (2019), Brain Informatics 6:5.
        https://doi.org/10.1186/s40708-019-0098-1 -- https://github.com/cMadan/calcSulc

        Parameters:
        -----------
        label_name: str
            Name of the label corresponding to the sulcus
        depth_pct: float
            Percentage of deepest vertices to use (default: 8)
        n_deepest: int
            Number of deepest vertices to use (default: 100)
        use_n_deepest: bool
            If True, use n_deepest vertices; if False, use depth_pct percentage (default: True)
                
        Returns:
        --------
        float: The median depth of the sulcus in mm

        NOTE: Requires the pial and pial-outer-smoothed surfaces (recon-all -localGI).
        """
        try:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found")
                    
            label_vertices = self.subject.labels[label_name].vertex_indexes
                
            if not isinstance(label_vertices, np.ndarray):
                label_vertices = np.array(label_vertices, dtype=int)
                
            sulc_map = self.subject.sulc_vals
            
            label_sulc_values = sulc_map[label_vertices]
                
            sorted_indices = np.argsort(label_sulc_values)
            sorted_sulc = np.sort(label_sulc_values)
            
            
            num_vertices = len(sorted_indices)
            
            if use_n_deepest:
                num_fundus = min(n_deepest, num_vertices) 
            else:
                num_fundus = int(np.ceil(num_vertices * depth_pct / 100))
            
            
            fundus_indices = sorted_indices[-num_fundus:]
            fundus_vertices = label_vertices[fundus_indices]
                
            depths = []
            for vertex_idx in fundus_vertices:
                # min distance from this pial fundus vertex to the gyral hull
                v_xyz = self.subject.pial_v[vertex_idx]
                distances = np.sqrt(np.sum((self.subject.gyrif_v - v_xyz)**2, axis=1))
                    
                # Find minimum distance
                min_distance = np.min(distances)
                depths.append(min_distance)
            
            self.subject.labels[label_name].measurements['sulcal depth (mm)'] = np.median(depths)
                
            # Return median depth
            if len(depths) > 0:
                return np.median(depths)
            else:
                return np.nan
                
        except Exception as e:
            print(f"Error calculating sulcal depth for {label_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.nan

    def _get_vertex_neighbors(self) -> Dict[int, np.ndarray]:
        """Lazily build and cache a 1-ring vertex adjacency map for the whole mesh."""
        neighbors = getattr(self, '_vertex_neighbors', None)
        if neighbors is not None:
            return neighbors
        faces = np.asarray(self.subject.faces, dtype=np.int64)
        edges = np.concatenate([
            faces[:, [0, 1]], faces[:, [1, 0]],
            faces[:, [1, 2]], faces[:, [2, 1]],
            faces[:, [0, 2]], faces[:, [2, 0]],
        ], axis=0)
        order = np.argsort(edges[:, 0], kind='stable')
        edges = edges[order]
        src, counts = np.unique(edges[:, 0], return_counts=True)
        splits = np.cumsum(counts)[:-1]
        dst_groups = np.split(edges[:, 1], splits)
        neighbors = {int(s): np.unique(d) for s, d in zip(src, dst_groups)}
        self._vertex_neighbors = neighbors
        return neighbors

    def _expand_ring(self, seeds, iterations: int, neighbors: Dict[int, np.ndarray]) -> set:
        """
        Cumulative 1-ring mesh expansion from a set of seed vertices, repeated
        ``iterations`` times. Matches the 'walk' in Madan's calcSulc_width: at
        each iteration every vertex found so far also contributes its own
        1-ring neighbors, so the returned set grows monotonically.
        """
        visited = {int(s) for s in np.atleast_1d(seeds)}
        frontier = set(visited)
        for _ in range(iterations):
            new_frontier = set()
            for v in frontier:
                new_frontier.update(int(n) for n in neighbors.get(v, ()))
            frontier = new_frontier - visited
            if not frontier:
                break
            visited.update(frontier)
        return visited

    def _boundary_segments(self, component_vertices: np.ndarray, all_faces: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
        """
        Build ordered vertex segments around a (connected) set of label
        vertices: take mesh faces that straddle the label (exactly 2 of their
        3 vertices inside), keep the edges of those faces whose both endpoints
        are inside, and trace them into segments (Madan's calcSulc_getEdgeLoop
        does the equivalent, but as a single walk over the whole edge set --
        see below for why this diverges).

        A well-formed boundary has every vertex at degree exactly 2 and traces
        one simple closed loop. Two complications happen on real (especially
        hand-traced or algorithm-derived) labels:

        - The boundary can form multiple disjoint simple loops if the label
          encloses a small hole (an outer loop plus one or more inner hole
          loops). Only the largest is kept -- a hole's own tiny perimeter
          isn't the sulcus's bank-to-bank opening.
        - A vertex can have degree > 2 where the boundary touches or crosses
          itself (a pinch point). Madan's original code doesn't detect this:
          it breaks the tie by always walking to the lowest-indexed neighbor
          and only removes the one edge just used, so it can silently produce
          a self-touching "loop" that revisits the pinch vertex, determined by
          an arbitrary tie-break rather than geometry. Instead, any such
          vertex is removed from the graph entirely (severing exactly the
          edges causing the self-touch); what remains is simple closed loops
          and/or open chains ending where a removal cut them. All of them are
          returned rather than discarded, since a pinch point is usually a
          small fraction of the boundary and there's no reason to throw away
          the rest of it.

        Returns:
        --------
        List[Tuple[np.ndarray, bool]]: one (ordered vertex indices, is_closed)
        pair per retained segment. Empty if no usable boundary was found.
        """
        n_vertices = len(self.subject.surface_RAS)
        in_component = np.zeros(n_vertices, dtype=bool)
        in_component[component_vertices] = True

        membership = in_component[all_faces].sum(axis=1)
        straddling_faces = all_faces[membership == 2]
        if len(straddling_faces) == 0:
            return []

        edge_pairs = np.concatenate([
            straddling_faces[:, [0, 1]],
            straddling_faces[:, [1, 2]],
            straddling_faces[:, [0, 2]],
        ], axis=0)
        inside_mask = in_component[edge_pairs].all(axis=1)
        edges = edge_pairs[inside_mask]
        if len(edges) == 0:
            return []
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        from collections import defaultdict
        adjacency = defaultdict(list)
        for a, b in edges:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))

        # remove pinch/branch vertices (degree > 2); what's left has max degree 2
        hubs = {v for v, nb in adjacency.items() if len(nb) > 2}
        for hub in hubs:
            for nb in adjacency[hub]:
                if nb not in hubs:
                    adjacency[nb] = [v for v in adjacency[nb] if v != hub]
            del adjacency[hub]

        remaining = {v for v, nb in adjacency.items() if len(nb) > 0}
        segments: List[Tuple[np.ndarray, bool]] = []
        while remaining:
            degree1 = [v for v in remaining if len(adjacency[v]) == 1]
            start = degree1[0] if degree1 else next(iter(remaining))
            is_closed = not degree1

            path = [start]
            prev, current = None, start
            while True:
                candidates = [v for v in adjacency[current] if v != prev]
                if not candidates:
                    break  # reached the other end of an open chain
                nxt = candidates[0]
                if is_closed and nxt == start:
                    break
                path.append(nxt)
                prev, current = current, nxt
                if len(path) > len(adjacency) + 1:
                    break  # safety valve; shouldn't happen with max degree 2

            remaining -= set(path)
            if len(path) >= 3:
                segments.append((np.array(path, dtype=np.int64), is_closed))

        if not segments:
            return []

        # among purely closed loops, keep only the largest (see docstring: the
        # rest are holes, not the main boundary); open chains from pruning are
        # always genuine boundary and are all kept
        closed = [(pts, True) for pts, is_closed in segments if is_closed]
        open_chains = [(pts, False) for pts, is_closed in segments if not is_closed]
        if closed:
            closed = [max(closed, key=lambda seg: len(seg[0]))]
        return closed + open_chains

    def _loop_width_samples(self, loop: np.ndarray, pial_v: np.ndarray,
                            neighbors: Dict[int, np.ndarray], walk_iterations: int,
                            is_closed: bool = True) -> List[float]:
        """Per-boundary-point width estimates for one ordered segment (Madan's calcSulc_width)."""
        from scipy.spatial.distance import cdist

        n_points = len(loop)
        if n_points < 3:
            return []

        coords = pial_v[loop]
        dist_matrix = cdist(coords, coords)

        # exclude points within ~10% of the segment's length of each point,
        # matching the conv-based dilation in calcSulc_width. Closed loops wrap
        # around (cyclic distance); open chains (from pruning a pinch point)
        # don't -- their two ends aren't actually connected.
        half_window = max(1, round(n_points / 20))
        idx = np.arange(n_points)
        raw_diff = np.abs(idx[:, None] - idx[None, :])
        index_dist = np.minimum(raw_diff, n_points - raw_diff) if is_closed else raw_diff
        masked = np.where(index_dist <= half_window, np.inf, dist_matrix)

        samples = []
        for p in range(n_points):
            # same near-p exclusion zone used for the coarse match also applies
            # to the refinement step below, so a k-ring walk from the opposite
            # bank can't wrap back around a narrow sulcus and match p to itself
            excluded_vertices = set(int(v) for v in loop[index_dist[p] <= half_window])

            candidate_idx = int(np.argmin(masked[p]))
            d = masked[p, candidate_idx]
            if not np.isfinite(d):
                continue
            matched_vertex = int(loop[candidate_idx])

            # refine: does a closer point exist in a local k-ring neighborhood
            # around the coarse match?
            neighborhood = self._expand_ring([matched_vertex], walk_iterations, neighbors) - excluded_vertices
            if neighborhood:
                neighborhood_idx = np.array(sorted(neighborhood), dtype=np.int64)
                neighborhood_dist = np.linalg.norm(pial_v[neighborhood_idx] - coords[p], axis=1)
                best_local = int(np.argmin(neighborhood_dist))
                if neighborhood_dist[best_local] < d:
                    d = neighborhood_dist[best_local]

            samples.append(float(d))
        return samples

    def calculate_sulcal_width(self, label_name: str, walk_iterations: int = 4) -> float:
        """
        Sulcal width following Madan (2019) calcSulc_width: for each point on
        the sulcal label's boundary loop, the distance (on the pial surface) to
        the nearest non-adjacent point across the sulcal opening, excluding a
        ~10%-of-loop-length neighborhood around it, then refined by checking a
        local k-ring mesh neighborhood for an even closer point. Width is the
        median of these per-point distances.

        If the label has multiple disconnected components (e.g. a paracingulate
        sulcus traced as separate elements), each gets its own boundary
        segment(s) and every component's points are pooled into one median,
        since width is a local/pointwise measure rather than something
        additive like length. See _boundary_segments for how a single
        component's boundary is resolved into one or more segments, including
        three deliberate deviations from the original MATLAB:

        - A hole enclosed by the label produces an outer loop plus small inner
          hole loop(s); only the largest closed loop is kept.
        - A vertex where the boundary touches or crosses itself (degree > 2)
          is removed from the boundary graph rather than resolved by Madan's
          arbitrary lowest-index tie-break, which can silently produce a
          self-touching "loop". What's left after removal (simple loops and/or
          open chains) is all kept and pooled, rather than discarding the cut
          sections.
        - The k-ring refinement walk excludes the same
          near-point exclusion zone used for the coarse match. Without that
          guard, a sulcus narrower than the walk radius lets the walk from the
          opposite bank wrap back and match a point to itself (distance 0); the
          original code has no such guard.

        Reference: Madan (2019), Brain Informatics 6:5.
        https://doi.org/10.1186/s40708-019-0098-1 -- https://github.com/cMadan/calcSulc

        Parameters:
        -----------
        label_name: str
            Name of the label corresponding to the sulcus
        walk_iterations: int
            Number of 1-ring mesh expansions used to refine each coarse match
            (default 4, matching Madan's ``setWidthWalk`` default).

        Returns:
        --------
        float: The median sulcal width in mm, or NaN if no component yielded a
        valid boundary loop.
        """
        if label_name not in self.subject.labels:
            raise ValueError(f"Label '{label_name}' not found in subject")

        label = self.subject.labels[label_name]
        label_vertices = np.asarray(label.vertex_indexes, dtype=np.int64)
        all_faces = np.asarray(self.subject.faces, dtype=np.int64)
        pial_v = self.subject.pial_v

        label_faces = surface_utils.get_faces_from_vertices(all_faces, label_vertices)
        if len(label_faces) > 0:
            components = surface_utils.get_label_subsets(label_faces, all_faces)
            component_vertex_sets = [np.unique(c) for c in components]
        else:
            component_vertex_sets = [np.unique(label_vertices)]

        neighbors = self._get_vertex_neighbors()

        pooled_samples: List[float] = []
        for component_vertices in component_vertex_sets:
            for segment, is_closed in self._boundary_segments(component_vertices, all_faces):
                pooled_samples.extend(
                    self._loop_width_samples(segment, pial_v, neighbors, walk_iterations, is_closed=is_closed)
                )

        width = float(np.median(pooled_samples)) if pooled_samples else float('nan')
        label.measurements['sulcal width (mm)'] = width
        return width

    def _get_face_area(self, face_vertices):
        """
        Calculate the area of a triangular face using cross product.
        Replicates the FreeSurfer face area calculation.
        
        Parameters:
        -----------
        face_vertices: np.ndarray
            3x3 array of vertex coordinates for the face
            
        Returns:
        --------
        float
            Area of the face
        """
        v0, v1, v2 = face_vertices
        # Calculate cross product of two edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        # Area is half the magnitude of cross product
        return 0.5 * np.linalg.norm(cross)

    def _compute_vertex_areas(self, vertices, faces, label_vertices=None):
        """
        Compute area associated with each vertex (1/3 of adjacent face areas).
        Matches FreeSurfer's vertex area calculation.
        
        Parameters:
        -----------
        vertices: np.ndarray
            Vertex coordinates
        faces: np.ndarray
            Face connectivity
        label_vertices: np.ndarray, optional
            Specific vertices to compute areas for
            
        Returns:
        --------
        np.ndarray
            Area associated with each vertex
        """
        vertex_areas = np.zeros(len(vertices))
        
        for face in faces:
            # Get face vertices
            face_coords = vertices[face]
            face_area = self._get_face_area(face_coords)
            
            # Each vertex gets 1/3 of the face area (VERTICES_PER_FACE = 3)
            for vertex_idx in face:
                vertex_areas[vertex_idx] += face_area / 3.0
        
        if label_vertices is not None:
            return vertex_areas[label_vertices]
        
        return vertex_areas

    def calculate_surface_area(self, label_name: Optional[str] = None) -> float:
        """
        Calculate the surface area of a label or the entire cortical surface.
        Replicates FreeSurfer's surface area calculation from mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate area for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        float
            The surface area in mm²
        """
        # Use the original surface (white matter surface typically)
        vertices = self.subject.white_v  # or whatever surface coordinates are available
        faces = self.subject.faces
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices for the label
            label_vertices = self.subject.labels[label_name].vertex_indexes
            
            # Calculate vertex areas for the label
            vertex_areas = self._compute_vertex_areas(vertices, faces, label_vertices)
            total_area = np.sum(vertex_areas)
            
            # Store in measurements
            self.subject.labels[label_name].measurements['total surface area (mm^2)'] = total_area
        else:
            # Calculate total surface area
            vertex_areas = self._compute_vertex_areas(vertices, faces)
            total_area = np.sum(vertex_areas)
        
        return total_area

    def calculate_gray_matter_volume(self, label_name: Optional[str] = None) -> float:
        """
        Calculate gray matter volume between white and pial surfaces.
        Replicates FreeSurfer's volume calculation from mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate volume for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        float
            The gray matter volume in mm³
        """
        white = self.subject.white_v
        pial = self.subject.pial_v
        faces = self.subject.faces

        # Per-face volume (FreeSurfer MRISvolumeTH3): each white->pial prism split
        # into 3 tetrahedra, summed as |scalar triple product| / 6.
        Pa, Pb, Pc = pial[faces[:, 0]], pial[faces[:, 1]], pial[faces[:, 2]]
        Wa, Wb, Wc = white[faces[:, 0]], white[faces[:, 1]], white[faces[:, 2]]
        Bp, Cp = Pb - Pa, Pc - Pa
        Aw, Bw, Cw = Wa - Pa, Wb - Pa, Wc - Pa
        T1 = np.abs(np.einsum('ij,ij->i', Aw, np.cross(Bw, Cw)))
        T2 = np.abs(np.einsum('ij,ij->i', Bp, np.cross(Cp, Bw)))
        T3 = np.abs(np.einsum('ij,ij->i', Cp, np.cross(Cw, Bw)))
        face_volumes = (T1 + T2 + T3) / 6.0

        # 1/3 of each face's volume to each vertex (FreeSurfer "vertex volume").
        vertex_volumes = np.zeros(len(white))
        for col in range(3):
            np.add.at(vertex_volumes, faces[:, col], face_volumes / 3.0)

        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            target_vertices = self.subject.labels[label_name].vertex_indexes
        else:
            # FreeSurfer masks the whole-hemisphere volume to cortex (no medial wall).
            target_vertices = self.subject.cortex_vertices

        total_volume = float(np.sum(vertex_volumes[target_vertices]))

        if label_name is not None:
            self.subject.labels[label_name].measurements['gray matter volume (mm^3)'] = total_volume

        return total_volume

    def calculate_cortical_thickness(self, label_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation of cortical thickness.
        Replicates FreeSurfer's thickness calculation from mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate thickness for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        Tuple[float, float]
            Mean cortical thickness and standard deviation in mm
        """
        thickness_vals = self.subject.thickness  # or self.subject.analyzer.thickness
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices for the label
            label_vertices = self.subject.labels[label_name].vertex_indexes
            
            # Get thickness values for the label vertices
            label_thickness = thickness_vals[label_vertices]
            
            # Calculate mean and standard deviation
            mean_thickness = np.mean(label_thickness)
            std_thickness = np.std(label_thickness, ddof=0)  # Population std, like FreeSurfer
            
            # Store in measurements
            self.subject.labels[label_name].measurements['average cortical thickness (mm)'] = mean_thickness
            self.subject.labels[label_name].measurements['cortical thickness std (mm)'] = std_thickness
        else:
            # Calculate for entire cortical surface
            mean_thickness = np.mean(thickness_vals)
            std_thickness = np.std(thickness_vals, ddof=0)
        
        return mean_thickness, std_thickness

    def calculate_absolute_curvature(self, label_name: Optional[str] = None, curvature_type: str = 'mean') -> float:
        """
        Calculate integrated rectified (absolute) curvature.
        Replicates FreeSurfer's MRIScomputeAbsoluteCurvature function.
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate curvature for. If None, calculates for the entire cortex.
        curvature_type: str
            Type of curvature ('mean' or 'gaussian')
            
        Returns:
        --------
        float
            Integrated rectified curvature
        """
        if curvature_type == 'mean':
            curvature_vals = self.subject.mean_curvature  # or appropriate curvature data
        elif curvature_type == 'gaussian':
            curvature_vals = self.subject.gaussian_curvature
        else:
            raise ValueError("curvature_type must be 'mean' or 'gaussian'")
        
        vertices = self.subject.white_v  # Use white surface coordinates
        faces = self.subject.faces
        
        # Calculate vertex areas
        vertex_areas = self._compute_vertex_areas(vertices, faces)
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            label_vertices = self.subject.labels[label_name].vertex_indexes
            label_curvature = curvature_vals[label_vertices]
            label_areas = vertex_areas[label_vertices]
            
            # FreeSurfer's integrated rectified curvature: area-weighted |curvature|
            # summed over the label, divided by vertex count.
            integrated_curvature = (np.sum(np.abs(label_curvature) * label_areas) / len(label_vertices)
                                    if len(label_vertices) > 0 else 0.0)

            # Store in measurements
            if curvature_type == 'mean':
                self.subject.labels[label_name].measurements['integrated rectified mean curvature'] = integrated_curvature
            else:
                self.subject.labels[label_name].measurements['integrated rectified gaussian curvature'] = integrated_curvature
        else:
            # Calculate for entire surface
            integrated_curvature = np.sum(np.abs(curvature_vals) * vertex_areas) / len(curvature_vals)
        
        return integrated_curvature

    def calculate_curvature_indices(self, label_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate folding index and intrinsic curvature index.
        Replicates FreeSurfer's MRIScomputeCurvatureIndices function.
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate indices for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        Tuple[float, float]
            Folding index and intrinsic curvature index
        """
        k1 = self.subject.k1_curvature
        k2 = self.subject.k2_curvature
        vertices = self.subject.white_v
        faces = self.subject.faces

        # Calculate vertex areas
        vertex_areas = self._compute_vertex_areas(vertices, faces)

        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")

            label_vertices = self.subject.labels[label_name].vertex_indexes
            k1 = k1[label_vertices]
            k2 = k2[label_vertices]
            label_areas = vertex_areas[label_vertices]
        else:
            label_areas = vertex_areas

        abs_k1, abs_k2 = np.abs(k1), np.abs(k2)

        # Folding index: (1/4pi) * integral |k1|*(|k1|-|k2|) dA
        folding_index = np.sum(abs_k1 * (abs_k1 - abs_k2) * label_areas) / (4.0 * np.pi)

        # Intrinsic curvature index: (1/4pi) * integral of positive Gaussian (k1*k2) dA
        gaussian = k1 * k2
        intrinsic_curvature_index = np.sum(np.maximum(gaussian, 0.0) * label_areas) / (4.0 * np.pi)
        
        if label_name is not None:
            self.subject.labels[label_name].measurements['folding index'] = folding_index
            self.subject.labels[label_name].measurements['intrinsic curvature index'] = intrinsic_curvature_index
        
        return folding_index, intrinsic_curvature_index

    def calculate_local_gyrification_index(self, label_name: Optional[str] = None) -> float:
        """
        Mean local gyrification index (Schaer 2008) over a label, or the cortex if
        label_name is None. Computed from the pial and pial-outer-smoothed surfaces.

        Reference: Schaer et al. (2008), IEEE TMI 27(2):161-170.
        """
        lgi = self.subject.pial_lgi
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            vertices = self.subject.labels[label_name].vertex_indexes
            mean_lgi = float(np.mean(lgi[vertices]))
            self.subject.labels[label_name].measurements['local gyrification index'] = mean_lgi
            return mean_lgi
        return float(np.mean(lgi[self.subject.cortex_vertices]))

    def calculate_all_freesurfer_stats(self, label_name: str) -> Dict[str, float]:
        """
        Calculate all FreeSurfer anatomical statistics for a label.
        Replicates the complete output of mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: str
            Name of the label to calculate statistics for
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all anatomical measurements
        """
        if label_name not in self.subject.labels:
            raise ValueError(f"Label '{label_name}' not found in subject")
        
        results = {}
        
        # Number of vertices
        label_vertices = self.subject.labels[label_name].vertex_indexes
        results['num_vertices'] = len(label_vertices)
        
        # Surface area
        results['surface_area_mm2'] = self.calculate_surface_area(label_name)
        
        # Gray matter volume
        results['gray_volume_mm3'] = self.calculate_gray_matter_volume(label_name)
        
        # Cortical thickness
        mean_thick, std_thick = self.calculate_cortical_thickness(label_name)
        results['thickness_mean_mm'] = mean_thick
        results['thickness_std_mm'] = std_thick
        
        # Curvature measures
        results['mean_curvature'] = self.calculate_absolute_curvature(label_name, 'mean')
        results['gaussian_curvature'] = self.calculate_absolute_curvature(label_name, 'gaussian')
        
        # Curvature indices
        folding_idx, intrinsic_idx = self.calculate_curvature_indices(label_name)
        results['folding_index'] = folding_idx
        results['intrinsic_curvature_index'] = intrinsic_idx

        # The sub-methods above already cache their (descriptive) keys onto the
        # label; return the compact dict for programmatic use.
        return results
    
    def calculate_euclidean_distance(self, label1: str, label2: str, method: str = 'centroid') -> float:
        """
        Calculate the Euclidean distance between two labels in 3D space.
        
        Parameters:
        -----------
        label1: str
            Name of the first label
        label2: str
            Name of the second label
        method: str
            Method to use for calculating distance ('centroid', 'nearest', 'farthest')
            
        Returns:
        --------
        float
            The Euclidean distance in mm
        """
        # Check if labels exist
        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")
        
        # Get RAS coordinates for each label
        coords1 = self._subject.labels[label1].label_RAS
        coords2 = self._subject.labels[label2].label_RAS
        
        if method == 'centroid':
            # Area-weighted geometric centroid of each label (falls back to the
            # unweighted vertex mean for labels with no complete interior faces)
            centroid1 = self._label_centroid(label1)
            centroid2 = self._label_centroid(label2)

            # Calculate Euclidean distance between centroids
            distance = np.linalg.norm(centroid2 - centroid1)
        
        elif method == 'nearest':
            # Find pair of vertices from each label with minimum distance
            min_distance = float('inf')
            
            # Calculate pairwise distances between all vertices
            for c1 in coords1:
                for c2 in coords2:
                    dist = np.linalg.norm(c2 - c1)
                    if dist < min_distance:
                        min_distance = dist
            
            distance = min_distance
        
        elif method == 'farthest':
            # Find pair of vertices from each label with maximum distance
            max_distance = 0
            
            # Calculate pairwise distances between all vertices
            for c1 in coords1:
                for c2 in coords2:
                    dist = np.linalg.norm(c2 - c1)
                    if dist > max_distance:
                        max_distance = dist
            
            distance = max_distance
        
        else:
            raise ValueError("Invalid method. Choose 'centroid', 'nearest', or 'farthest'.")

        return distance

    def _label_centroid(self, label_name: str) -> np.ndarray:
        """Area-weighted geometric centroid of a label (vertex-mean fallback if no faces)."""
        label = self._subject.labels[label_name]
        label_faces = surface_utils.get_faces_from_vertices(
            self._subject.faces, label.vertex_indexes
        )
        if len(label_faces) == 0:
            return np.mean(label.label_RAS, axis=0)
        try:
            return surface_utils.calculate_geometric_centroid(
                self._subject.surface_RAS, label_faces
            )
        except ValueError:
            return np.mean(label.label_RAS, axis=0)

    def _get_geodesic_algorithm(self, cortex_only: bool = True, surface: str = 'subject'):
        """
        Lazily build and cache an exact geodesic solver over a surface.

        ``surface='subject'`` uses the subject's loaded surface (``surface_RAS``);
        ``surface='fiducial'`` uses the vertex-wise midpoint of white and pial
        (mid-thickness), independent of the ``surface_type`` the subject was
        constructed with.

        By default the mesh is restricted to cortex faces (?h.cortex.label), so the
        medial wall becomes isolated vertices that geodesics cannot cross; pass
        cortex_only=False for the whole surface. Vertex indices remain global
        either way.
        """
        attr = f"_geoalg_{surface}_{'cortex' if cortex_only else 'full'}"
        solver = getattr(self, attr, None)
        if solver is not None:
            return solver
        try:
            import pygeodesic.geodesic as geodesic
        except ImportError as exc:
            raise ImportError(
                "Geodesic distance requires the 'pygeodesic' package. "
                "Install it with `pip install pygeodesic`."
            ) from exc
        if surface == 'fiducial':
            vertices = np.asarray(self._subject.fiducial_v, dtype=np.float64)
        else:
            vertices = np.asarray(self._subject.surface_RAS, dtype=np.float64)
        faces = np.asarray(self._subject.faces, dtype=np.int32)
        if cortex_only:
            in_cortex = np.zeros(len(vertices), dtype=bool)
            in_cortex[self._subject.cortex_vertices] = True
            faces = faces[in_cortex[faces].all(axis=1)]     # keep only cortex faces
        solver = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        setattr(self, attr, solver)
        return solver

    def _nearest_label_vertex_to_centroid(self, label_name: str) -> int:
        """Return the label's vertex index that is closest to its centroid."""
        label = self._subject.labels[label_name]
        centroid = self._label_centroid(label_name)
        local_idx = np.argmin(np.linalg.norm(label.label_RAS - centroid, axis=1))
        return int(label.vertex_indexes[local_idx])

    def calculate_geodesic_distance(self, label1: str, label2: str, method: str = 'centroid',
                                    cortex_only: bool = True) -> float:
        """
        Exact geodesic (on-surface) distance between two labels, measured across
        the subject's loaded surface via the MMP algorithm (`pygeodesic`).

        Parameters:
        -----------
        label1: str
            Name of the first label
        label2: str
            Name of the second label
        method: str
            'centroid' - geodesic distance between each label's centroid vertex
            'nearest'  - minimum geodesic distance between the two label vertex sets
        cortex_only: bool
            Restrict the geodesic to the cortex mesh (default True); set False to
            allow paths across the medial wall.

        Returns:
        --------
        float
            The geodesic distance in mm
        """
        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")

        geoalg = self._get_geodesic_algorithm(cortex_only)

        if method == 'centroid':
            source = self._nearest_label_vertex_to_centroid(label1)
            target = self._nearest_label_vertex_to_centroid(label2)
            distance, _ = geoalg.geodesicDistance(source, target)
            return float(distance)

        elif method == 'nearest':
            # min geodesic distance between the two label vertex sets (one wavefront)
            sources = np.asarray(
                self._subject.labels[label1].vertex_indexes, dtype=np.int32
            )
            targets = np.asarray(
                self._subject.labels[label2].vertex_indexes, dtype=np.int32
            )
            distances, _ = geoalg.geodesicDistances(sources, targets)
            return float(np.min(distances))

        else:
            raise ValueError("Invalid method. Choose 'centroid' or 'nearest'.")

    def calculate_geodesic_path(self, label1: str, label2: str, method: str = 'centroid',
                                cortex_only: bool = True):
        """
        Ordered surface-vertex indices of the exact geodesic path between two
        labels, plus its length in mm.

        Parameters:
        -----------
        label1, label2: str
            Label names.
        method: str
            'centroid' - path between each label's centroid vertex.
            'nearest'  - path between the closest pair of label vertices.
        cortex_only: bool
            Restrict the path to the cortex mesh (default True); set False to
            allow it across the medial wall.

        Returns:
        --------
        Tuple[np.ndarray, float]
            Ordered path vertex indices and the geodesic length in mm.
        """
        from scipy.spatial import cKDTree

        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")

        geoalg = self._get_geodesic_algorithm(cortex_only)

        if method == 'centroid':
            source = self._nearest_label_vertex_to_centroid(label1)
            target = self._nearest_label_vertex_to_centroid(label2)
        elif method == 'nearest':
            sources = np.asarray(self._subject.labels[label1].vertex_indexes, dtype=np.int32)
            targets = np.asarray(self._subject.labels[label2].vertex_indexes, dtype=np.int32)
            distances, best_source = geoalg.geodesicDistances(sources, targets)
            j = int(np.argmin(distances))
            source, target = int(best_source[j]), int(targets[j])
        else:
            raise ValueError("Invalid method. Choose 'centroid' or 'nearest'.")

        distance, path = geoalg.geodesicDistance(source, target)
        # map the geodesic polyline back to ordered, de-duplicated surface vertices
        _, nearest = cKDTree(self._subject.surface_RAS).query(np.asarray(path))
        seen = set()
        path_vertices = [int(v) for v in nearest if not (v in seen or seen.add(v))]
        return np.array(path_vertices, dtype=int), float(distance)

    def _label_components(self, label_vertices: np.ndarray) -> List[np.ndarray]:
        """
        Split a label's vertices into geodesically-routable connected
        components, returning each component's own faces; a component with no
        complete interior face is returned as an empty face array (caller
        falls back to raw vertices).

        surface_utils.get_label_subsets only guarantees *vertex*-level
        connectivity (two faces sharing even a single vertex end up in the
        same group), which is too coarse here: two sub-patches touching at a
        single 'bowtie' vertex with no shared edge are vertex-connected but
        not edge-connected, and pygeodesic's exact solver -- correctly --
        can't route a geodesic path between them, silently returning inf for
        those pairs. So each vertex-based group is further split by strict
        face-edge adjacency (two faces are neighbors only if they share a full
        edge, i.e. two vertices), which is what geodesic routing actually
        requires.
        """
        label_faces = surface_utils.get_faces_from_vertices(self.subject.faces, label_vertices)
        if len(label_faces) == 0:
            return [label_faces]
        components = []
        for component_faces in surface_utils.get_label_subsets(label_faces, self.subject.faces):
            components.extend(self._split_by_edge_connectivity(component_faces))
        return components

    def _split_by_edge_connectivity(self, faces: np.ndarray) -> List[np.ndarray]:
        """Split a face array into maximal groups connected via shared edges
        (two full vertices), stricter than shared-vertex connectivity."""
        if len(faces) == 0:
            return [faces]

        from collections import defaultdict
        edge_to_faces = defaultdict(list)
        for i, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])):
                edge_to_faces[(min(int(a), int(b)), max(int(a), int(b)))].append(i)

        face_adjacency = defaultdict(set)
        for face_indices in edge_to_faces.values():
            for i in face_indices:
                face_adjacency[i].update(j for j in face_indices if j != i)

        visited = set()
        groups = []
        for start in range(len(faces)):
            if start in visited:
                continue
            group, stack = [], [start]
            visited.add(start)
            while stack:
                f = stack.pop()
                group.append(f)
                for neighbor in face_adjacency[f]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            groups.append(faces[group])
        return groups

    def _component_geodesic_algorithm(self, component_faces: np.ndarray, surface: str = 'fiducial'):
        """
        Build an exact geodesic solver restricted to a single label component's
        own faces, so a path between its boundary vertices can never leave the
        component -- matching the original Miller et al. (2020) implementation
        (pycortex ``Surface.create_subsurface`` over just the label's vertex
        mask), rather than allowing shortcuts through neighboring, non-label
        cortex the way a whole-cortex solver would. Built fresh per call, not
        cached like ``_get_geodesic_algorithm``, since it's specific to
        whichever component is being measured -- but a label-sized mesh is
        far smaller than the whole cortex, so this is also much faster.

        Unlike ``_get_geodesic_algorithm`` (which reuses the subject's full
        vertex array as-is), this remaps the component's global vertex IDs to
        a local, dense 0..k-1 numbering. pygeodesic's exact solver asserts
        ``faces.min() == 0 and faces.max() == len(vertices) - 1`` -- i.e. the
        face array must span the *entire* provided vertex array, not just
        avoid duplicates. The whole-cortex solver satisfies that by
        coincidence (cortex spans almost the full vertex range), but a single
        label component is a small, spatially localized subset that generally
        touches neither the array's first nor last index, so reusing the full
        vertex array here would fail that assertion.

        Returns:
        --------
        Tuple[PyGeodesicAlgorithmExact, np.ndarray]: the solver, and the
        sorted array of global vertex IDs such that local index ``i``
        corresponds to global vertex ``global_ids[i]`` -- use
        ``np.searchsorted(global_ids, global_vertex_ids)`` to convert global
        vertex IDs (e.g. boundary vertices) into the local indices this
        solver expects.
        """
        try:
            import pygeodesic.geodesic as geodesic
        except ImportError as exc:
            raise ImportError(
                "Geodesic distance requires the 'pygeodesic' package. "
                "Install it with `pip install pygeodesic`."
            ) from exc
        if surface == 'fiducial':
            all_vertices = self._subject.fiducial_v
        else:
            all_vertices = self._subject.surface_RAS

        global_ids = np.unique(component_faces)
        local_faces = np.searchsorted(global_ids, component_faces).astype(np.int32)
        local_vertices = np.asarray(all_vertices[global_ids], dtype=np.float64)
        solver = geodesic.PyGeodesicAlgorithmExact(local_vertices, local_faces)
        return solver, global_ids

    def _euclidean_boundary_diameter(self, boundary: np.ndarray, surface: str = 'fiducial') -> Tuple[float, Optional[int], Optional[int]]:
        """
        Straight-line fallback for a component with no complete face -- too
        small (e.g. 1-2 vertices wide) to have any triangle fully inside it,
        so there's no mesh to build a restricted geodesic solver from at all.
        """
        from scipy.spatial.distance import pdist, squareform
        if len(boundary) < 2:
            return 0.0, None, None
        coords = self._subject.fiducial_v if surface == 'fiducial' else self._subject.surface_RAS
        dmat = squareform(pdist(coords[boundary]))
        i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
        return float(dmat[i, j]), int(boundary[i]), int(boundary[j])

    def _component_geodesic_diameter(self, boundary: np.ndarray, geoalg) -> Tuple[float, Optional[int], Optional[int]]:
        """
        Longest geodesic distance between any pair of vertices in ``boundary``,
        plus the (source, target) vertex pair that achieves it. Shared by
        calculate_sulcal_length and calculate_sulcal_length_path so both are
        guaranteed to agree on exactly which pair defines a component's length.

        Unreachable pairs (inf, from a restricted solver where the boundary
        isn't fully connected) are ignored rather than allowed to win the max
        -- _label_components already guards against this for the common case
        (a bowtie vertex splitting a nominal component), but this is a
        defensive backstop against ever silently returning inf as a length.
        """
        diameter = 0.0
        best_source, best_target = None, None
        for i in range(len(boundary) - 1):
            source = boundary[i:i + 1]
            targets = boundary[i + 1:]
            distances, _ = geoalg.geodesicDistances(source, targets)
            distances = np.where(np.isfinite(distances), distances, -np.inf)
            j = int(np.argmax(distances))
            farthest = float(distances[j])
            if farthest > diameter:
                diameter = farthest
                best_source, best_target = int(source[0]), int(targets[j])
        return diameter, best_source, best_target

    def calculate_sulcal_length(self, label_name: str) -> float:
        """
        Sulcal length: the longest geodesic distance between any pair of vertices
        on the label's boundary, measured on the fiducial surface (vertex-wise
        midpoint of white and pial), with the geodesic search restricted to the
        label's own footprint. If the label is split into multiple disconnected
        components, length is the sum of each component's own longest
        boundary-to-boundary geodesic distance, each restricted to that
        component alone (annectant gyral bridges between components are not
        counted). A component too small to contain a single complete face (no
        mesh to restrict a geodesic solver to) falls back to a straight-line
        distance between its raw vertices.

        Reference: Miller et al. (2020), Scientific Reports 10:17132, "Sulcal
        length" (Materials and Methods). https://doi.org/10.1038/s41598-020-73213-x

        Parameters:
        -----------
        label_name: str
            Name of the label corresponding to the sulcus

        Returns:
        --------
        float: The sulcal length in mm
        """
        if label_name not in self.subject.labels:
            raise ValueError(f"Label '{label_name}' not found in subject")

        label = self.subject.labels[label_name]
        label_vertices = np.asarray(label.vertex_indexes, dtype=int)
        components = self._label_components(label_vertices)

        total_length = 0.0
        for component_faces in components:
            if len(component_faces) > 0:
                boundary = surface_utils.find_label_boundary(component_faces)
            else:
                boundary = np.unique(label_vertices)

            if len(boundary) < 2:
                continue  # an isolated vertex/component has no length

            boundary = np.asarray(boundary, dtype=np.int32)
            if len(component_faces) > 0:
                geoalg, global_ids = self._component_geodesic_algorithm(component_faces, surface='fiducial')
                local_boundary = np.searchsorted(global_ids, boundary).astype(np.int32)
                diameter, _, _ = self._component_geodesic_diameter(local_boundary, geoalg)
            else:
                diameter, _, _ = self._euclidean_boundary_diameter(boundary, surface='fiducial')
            total_length += diameter

        label.measurements['sulcal length (mm)'] = total_length
        return total_length

    def calculate_sulcal_length_path(self, label_name: str) -> List[Tuple[np.ndarray, float]]:
        """
        Per-component geodesic paths underlying calculate_sulcal_length: for each
        connected component of the label, the ordered fiducial-surface vertex
        path (and its length in mm) between the pair of boundary vertices that
        define that component's contribution to the sulcal length -- traced
        through a solver restricted to that component alone, so the path is
        guaranteed to stay inside the label (see calculate_sulcal_length). Sum
        of the returned lengths equals calculate_sulcal_length's result.
        Intended for visually auditing what a sulcal length measurement
        actually traced. A component with no complete face falls back to its
        two straight-line-farthest vertices as a degenerate 2-point "path".

        Parameters:
        -----------
        label_name: str
            Name of the label corresponding to the sulcus

        Returns:
        --------
        List[Tuple[np.ndarray, float]]: one (ordered path vertex indices, length
        in mm) per connected component that yielded a valid boundary.
        """
        from scipy.spatial import cKDTree

        if label_name not in self.subject.labels:
            raise ValueError(f"Label '{label_name}' not found in subject")

        label_vertices = np.asarray(self.subject.labels[label_name].vertex_indexes, dtype=int)
        components = self._label_components(label_vertices)
        fiducial_tree = cKDTree(self.subject.fiducial_v)

        results = []
        for component_faces in components:
            if len(component_faces) > 0:
                boundary = surface_utils.find_label_boundary(component_faces)
            else:
                boundary = np.unique(label_vertices)

            if len(boundary) < 2:
                continue

            boundary = np.asarray(boundary, dtype=np.int32)

            if len(component_faces) > 0:
                geoalg, global_ids = self._component_geodesic_algorithm(component_faces, surface='fiducial')
                local_boundary = np.searchsorted(global_ids, boundary).astype(np.int32)
                diameter, local_source, local_target = self._component_geodesic_diameter(local_boundary, geoalg)
                if local_source is None:
                    continue
                _, path = geoalg.geodesicDistance(local_source, local_target)
                _, nearest = fiducial_tree.query(np.asarray(path))
                seen = set()
                path_vertices = [int(v) for v in nearest if not (v in seen or seen.add(v))]
                results.append((np.array(path_vertices, dtype=int), diameter))
            else:
                diameter, source, target = self._euclidean_boundary_diameter(boundary, surface='fiducial')
                if source is None:
                    continue
                results.append((np.array([source, target], dtype=int), diameter))

        return results

    def calculate_label_overlap(self, label1: str, label2: str) -> Dict[str, float]:
        """
        Calculate the overlap between two labels using multiple metrics.
        
        Parameters:
        -----------
        label1: str
            Name of the first label
        label2: str
            Name of the second label
            
        Returns:
        --------
        Dict[str, float]
            A dictionary containing overlap metrics:
            - 'dice': Dice coefficient (2*|A∩B| / (|A|+|B|))
            - 'jaccard': Jaccard index (|A∩B| / |A∪B|)
            - 'overlap_coefficient': Overlap coefficient (|A∩B| / min(|A|,|B|))
            - 'intersection_size': Size of intersection (number of vertices)
            - 'union_size': Size of union (number of vertices)
        """
        # Check if labels exist
        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")
        
        # Get vertex indices for each label
        vertices1 = self._subject.labels[label1].vertex_indexes
        vertices2 = self._subject.labels[label2].vertex_indexes
        
        # Calculate intersection and union
        intersection = np.intersect1d(vertices1, vertices2)
        union = np.union1d(vertices1, vertices2)
        
        # Calculate sizes
        size1 = len(vertices1)
        size2 = len(vertices2)
        intersection_size = len(intersection)
        union_size = len(union)
        
        # Calculate overlap metrics
        dice = 2 * intersection_size / (size1 + size2) if (size1 + size2) > 0 else 0
        jaccard = intersection_size / union_size if union_size > 0 else 0
        overlap_coefficient = intersection_size / min(size1, size2) if min(size1, size2) > 0 else 0
        
        # Return metrics as a dictionary
        return {
            'dice': dice,
            'jaccard': jaccard,
            'overlap_coefficient': overlap_coefficient,
            'intersection_size': intersection_size,
            'union_size': union_size
        }
    
    def export_measurements(self, labels: List[str], measurements: List[str], 
                           output_file: str, delimiter: str = ',') -> bool:
        """
        Export measurements for multiple labels to a CSV file.
        
        Parameters:
        -----------
        labels: List[str]
            List of label names to measure
        measurements: List[str]
            List of measurements to calculate:
            - 'area': Surface area
            - 'thickness': Cortical thickness
            - 'depth': Sulcal depth
            - 'length': Sulcal length
            - 'width': Sulcal width
            - 'volume': Gray matter volume
            - 'curvature': Mean and Gaussian curvature
            - 'indices': Folding and intrinsic curvature indices
            - 'all_freesurfer': All FreeSurfer stats
        output_file: str
            Path to the output file
        delimiter: str
            Delimiter for the CSV file (default: ',')
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # Check if labels exist
        for label in labels:
            if label not in self._subject.labels:
                raise ValueError(f"Label '{label}' not found in subject")
        
        # Check measurement types
        valid_measurements = ['area', 'thickness', 'depth', 'length', 'width', 'volume', 'curvature', 'indices', 'all_freesurfer']
        for measurement in measurements:
            if measurement not in valid_measurements:
                raise ValueError(f"Invalid measurement '{measurement}'. Choose from {valid_measurements}")
        
        # Prepare header based on requested measurements
        header = ['label']
        for measurement in measurements:
            if measurement == 'area':
                header.append('surface_area_mm2')
            elif measurement == 'thickness':
                header.extend(['thickness_mean_mm', 'thickness_std_mm'])
            elif measurement == 'depth':
                header.append('sulcal_depth_mm')
            elif measurement == 'length':
                header.append('sulcal_length_mm')
            elif measurement == 'width':
                header.append('sulcal_width_mm')
            elif measurement == 'volume':
                header.append('gray_volume_mm3')
            elif measurement == 'curvature':
                header.extend(['mean_curvature', 'gaussian_curvature'])
            elif measurement == 'indices':
                header.extend(['folding_index', 'intrinsic_curvature_index'])
            elif measurement == 'all_freesurfer':
                header.extend(['num_vertices', 'surface_area_mm2', 'gray_volume_mm3', 
                              'thickness_mean_mm', 'thickness_std_mm', 'mean_curvature', 
                              'gaussian_curvature', 'folding_index', 'intrinsic_curvature_index'])
        
        # Prepare data rows
        rows = []
        for label in labels:
            row = [label]
            
            for measurement in measurements:
                if measurement == 'area':
                    value = self.calculate_surface_area(label)
                    row.append(value)
                elif measurement == 'thickness':
                    mean_thick, std_thick = self.calculate_cortical_thickness(label)
                    row.extend([mean_thick, std_thick])
                elif measurement == 'depth':
                    value = self.calculate_sulcal_depth(label)
                    row.append(value)
                elif measurement == 'length':
                    value = self.calculate_sulcal_length(label)
                    row.append(value)
                elif measurement == 'width':
                    value = self.calculate_sulcal_width(label)
                    row.append(value)
                elif measurement == 'volume':
                    value = self.calculate_gray_matter_volume(label)
                    row.append(value)
                elif measurement == 'curvature':
                    mean_curv = self.calculate_absolute_curvature(label, 'mean')
                    gauss_curv = self.calculate_absolute_curvature(label, 'gaussian')
                    row.extend([mean_curv, gauss_curv])
                elif measurement == 'indices':
                    fold_idx, intrinsic_idx = self.calculate_curvature_indices(label)
                    row.extend([fold_idx, intrinsic_idx])
                elif measurement == 'all_freesurfer':
                    stats = self.calculate_all_freesurfer_stats(label)
                    row.extend([stats['num_vertices'], stats['surface_area_mm2'], 
                               stats['gray_volume_mm3'], stats['thickness_mean_mm'], 
                               stats['thickness_std_mm'], stats['mean_curvature'], 
                               stats['gaussian_curvature'], stats['folding_index'], 
                               stats['intrinsic_curvature_index']])
            
            rows.append(row)
        
        # Write to CSV file
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(header)
                writer.writerows(rows)
            
            return True
        
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
            return False