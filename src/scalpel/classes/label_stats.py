from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List, Union, Optional
import re
from pathlib import Path

if TYPE_CHECKING:
    from scalpel.subject import ScalpelSubject
    from scalpel.classes.label import Label

class LabelStats:
    """
    Class to hold label statistics for a single label.
    """

    def __init__(self, label: 'Label', stats_filepath: Optional[Union[str, Path]] = None, verbose = False):
        """
        Initialize the LabelStats object.

        Parameters:
        - label (Label): The label object.
        - stats_filepath (Union[str, Path], optional): Path to the label stats file. If not provided, defaults to 
                                      the subject's label stats path in subject_fs_path / subject / label / label_stats 
        """
        self._label = label
        self._subject = label.subject
        self._stats_filepath = self._determine_stats_filepath(stats_filepath)
        self._measurements = {}
        
        # Parse the stats file if it exists
        if self._stats_filepath.exists():
            self._parse_stats_file()
        elif verbose:
            print(f"Warning: Stats file not found at {self._stats_filepath}")
    
    def _determine_stats_filepath(self, stats_filepath: Optional[Union[str, Path]]) -> Path:
        """
        Determine the appropriate stats filepath.
        
        Parameters:
        -----------
        stats_filepath : Optional[Union[str, Path]]
            User-provided stats filepath, if any
            
        Returns:
        --------
        Path
            The resolved stats filepath
        """
        if stats_filepath:
            return Path(stats_filepath)
        else:
            # Construct default path
            return self._subject.subject_fs_path / "label" / "label_stats" / f"{self._subject.hemi}.{self._label.label_name}.label.stats.txt"
    
    def _parse_stats_file(self) -> None:
        """
        Parse the FreeSurfer statistics file.
        """
        with open(self._stats_filepath, 'r') as f:
            content = f.read()
        
        # Initialize data structures for parsing
        metadata = {}
        column_names = []
        
        # Parse metadata
        metadata_patterns = {
            'Total face volume': r'Total face volume (\d+)',
            'Total vertex volume': r'Total vertex volume (\d+) \(mask=(\d+)\)'
        }
        
        for key, pattern in metadata_patterns.items():
            match = re.search(pattern, content)
            if match:
                if len(match.groups()) > 1:
                    metadata[key] = int(match.group(1))
                    metadata[f"{key} mask"] = int(match.group(2))
                else:
                    metadata[key] = int(match.group(1))
        
        # Parse column names
        column_pattern = r'table columns are:(.*?)(?=\n\s+\d+)'
        column_match = re.search(column_pattern, content, re.DOTALL)
        
        if column_match:
            column_text = column_match.group(1)
            # Extract column names from lines that start with spaces
            for line in column_text.split('\n'):
                line = line.strip()
                if line:
                    column_names.append(line)
        
        # Parse measurements
        measurement_pattern = r'^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(.+)$'
        measurement_lines = re.findall(measurement_pattern, content, re.MULTILINE)
        
        # Store the metadata
        self._measurements['metadata'] = metadata
        
        # Process the first measurement line (assuming there's only one relevant to this label)
        if measurement_lines and len(measurement_lines[0]) == 10:
            line = measurement_lines[0]
            
            # Store individual measurements
            self._measurements['number of vertices'] = int(line[0])
            self._measurements['total surface area (mm^2)'] = int(line[1])
            self._measurements['total gray matter volume (mm^3)'] = int(line[2])
            self._measurements['average cortical thickness (mm)'] = float(line[3])
            self._measurements['cortical thickness standard deviation (mm)'] = float(line[4])
            self._measurements['integrated rectified mean curvature'] = float(line[5])
            self._measurements['integrated rectified Gaussian curvature'] = float(line[6])
            self._measurements['folding index'] = int(line[7])
            self._measurements['intrinsic curvature index'] = float(line[8])
            self._measurements['structure name'] = line[9].strip()
    
    @property
    def measurements(self) -> Dict[str, Any]:
        """
        Get the label's measurements.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary of measurements for the label
        """
        return self._measurements
    
    def get_measurement(self, key: str) -> Any:
        """
        Get a specific measurement value.
        
        Parameters:
        -----------
        key : str
            The measurement key to retrieve
            
        Returns:
        --------
        Any
            The value of the measurement, or None if not found
        """
        return self._measurements.get(key)
    
    def has_measurements(self) -> bool:
        """
        Check if measurements have been loaded.
        
        Returns:
        --------
        bool
            True if measurements have been loaded, False otherwise
        """
        # Check if we have any measurements beyond metadata
        return len(self._measurements) > 1  # More than just metadata
