"""
Helper module to load and manage archetype fingerprint data for Dash app.

This module provides easy access to the exported GMM analysis data:
- Patient-level fingerprints with CASA metrics
- Per-track data with GMM posteriors
- Frame-by-frame trajectory coordinates
- Archetype configuration
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class ArchetypeDataLoader:
    """Load and manage archetype fingerprint data for visualization."""
    
    def __init__(self, data_dir: str = "dash_data"):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing exported data files
        """
        self.data_dir = Path(data_dir)
        
        # Load all datasets
        self.tracks_df = None
        self.frames_df = None
        self.patient_df = None
        self.config = None
        
        self._load_data()
    
    def _load_data(self):
        """Load all data files."""
        # Load parquet files
        tracks_path = self.data_dir / "tracks_data.parquet"
        frames_path = self.data_dir / "frames_data.parquet"
        patient_path = self.data_dir / "patient_data.parquet"
        config_path = self.data_dir / "archetype_config.json"
        
        if tracks_path.exists():
            self.tracks_df = pd.read_parquet(tracks_path)
            print(f"✓ Loaded {len(self.tracks_df):,} tracks")
        else:
            raise FileNotFoundError(f"Tracks data not found: {tracks_path}")
        
        if frames_path.exists():
            self.frames_df = pd.read_parquet(frames_path)
            print(f"✓ Loaded {len(self.frames_df):,} frames")
        else:
            raise FileNotFoundError(f"Frames data not found: {frames_path}")
        
        if patient_path.exists():
            self.patient_df = pd.read_parquet(patient_path)
            print(f"✓ Loaded {len(self.patient_df):,} patients")
        else:
            raise FileNotFoundError(f"Patient data not found: {patient_path}")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✓ Loaded config with {len(self.config.get('archetypes', {})):,} archetypes")
        else:
            print("⚠️  Config file not found, using defaults")
            self.config = {"archetypes": {}}
    
    def get_archetype_list(self) -> List[str]:
        """Get list of available archetype names."""
        return list(self.config.get('archetypes', {}).keys())
    
    def get_archetype_info(self, archetype_name: str) -> Dict:
        """
        Get information about a specific archetype.
        
        Returns dict with keys: participant_id, title, description
        """
        return self.config.get('archetypes', {}).get(archetype_name, {})
    
    def get_patient_tracks(self, participant_id: str) -> pd.DataFrame:
        """
        Get all tracks for a specific participant.
        
        Parameters
        ----------
        participant_id : str
            Participant ID
            
        Returns
        -------
        pd.DataFrame
            Tracks for this participant with posteriors and computed axes
        """
        return self.tracks_df[self.tracks_df['participant_id'] == participant_id].copy()
    
    def get_patient_frames(self, participant_id: str) -> pd.DataFrame:
        """
        Get all frame data for a specific participant.
        
        Parameters
        ----------
        participant_id : str
            Participant ID
            
        Returns
        -------
        pd.DataFrame
            Frame-by-frame trajectory data for all tracks
        """
        return self.frames_df[self.frames_df['participant_id'] == participant_id].copy()
    
    def get_patient_summary(self, participant_id: str) -> pd.Series:
        """
        Get patient-level summary data.
        
        Parameters
        ----------
        participant_id : str
            Participant ID
            
        Returns
        -------
        pd.Series
            Patient fingerprint with CASA metrics
        """
        patient_data = self.patient_df[self.patient_df['participant_id'] == participant_id]
        if patient_data.empty:
            raise ValueError(f"Patient {participant_id} not found in patient data")
        return patient_data.iloc[0]
    
    def get_archetype_data(self, archetype_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict]:
        """
        Get all data needed to plot an archetype fingerprint.
        
        Parameters
        ----------
        archetype_name : str
            Archetype name (e.g., 'A', 'B', 'C', 'D')
            
        Returns
        -------
        tracks_df : pd.DataFrame
            Tracks for this participant
        frames_df : pd.DataFrame
            Frame data for trajectories
        patient_summary : pd.Series
            Patient-level fingerprint
        archetype_info : Dict
            Metadata (title, description)
        """
        archetype_info = self.get_archetype_info(archetype_name)
        if not archetype_info:
            raise ValueError(f"Archetype '{archetype_name}' not found")
        
        pid = archetype_info['participant_id']
        
        tracks = self.get_patient_tracks(pid)
        frames = self.get_patient_frames(pid)
        patient = self.get_patient_summary(pid)
        
        return tracks, frames, patient, archetype_info
    
    def pick_diverse_tracks(self, tracks_df: pd.DataFrame, n_total: int = 120, 
                           rng_seed: int = 42) -> List[str]:
        """
        Pick diverse set of tracks for visualization.
        
        Stratifies by cluster/subtype to show representative diversity.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Tracks dataframe (already filtered to one participant)
        n_total : int
            Total number of tracks to sample
        rng_seed : int
            Random seed for reproducibility
            
        Returns
        -------
        List[str]
            List of selected track IDs
        """
        rng = np.random.default_rng(rng_seed)
        
        # Try to stratify by subtype_label if available
        if 'subtype_label' in tracks_df.columns:
            group_col = 'subtype_label'
        elif 'cluster_id' in tracks_df.columns:
            group_col = 'cluster_id'
        else:
            # No grouping, just random sample
            tids = tracks_df['track_id'].unique()
            n_sample = min(n_total, len(tids))
            return rng.choice(tids, size=n_sample, replace=False).tolist()
        
        # Stratified sampling
        groups = tracks_df.groupby(group_col)
        group_sizes = groups.size().sort_values(ascending=False)
        
        # Allocate samples proportionally
        total_tracks = len(tracks_df)
        samples_per_group = {}
        for group_name, group_size in group_sizes.items():
            proportion = group_size / total_tracks
            n_sample = max(1, int(np.round(proportion * n_total)))
            samples_per_group[group_name] = n_sample
        
        # Adjust to hit exactly n_total
        total_allocated = sum(samples_per_group.values())
        if total_allocated > n_total:
            # Reduce largest groups
            for group_name in group_sizes.index:
                if samples_per_group[group_name] > 1:
                    samples_per_group[group_name] -= 1
                    total_allocated -= 1
                    if total_allocated <= n_total:
                        break
        
        # Sample from each group
        selected_tids = []
        for group_name, n_sample in samples_per_group.items():
            group_tracks = tracks_df[tracks_df[group_col] == group_name]
            tids = group_tracks['track_id'].unique()
            n_sample = min(n_sample, len(tids))
            if n_sample > 0:
                sampled = rng.choice(tids, size=n_sample, replace=False)
                selected_tids.extend(sampled.tolist())
        
        return selected_tids
    
    def get_casa_scales(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute CASA metric scales (5th to 95th percentile) for normalization.
        
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Mapping from short CASA label to (min, max) tuple
            e.g., {'ALH': (0.5, 3.2), 'VCL': (20, 150), ...}
        """
        casa_cols = [c for c in self.patient_df.columns 
                     if c.startswith('CASA_') and c.endswith('_mean')]
        
        scales = {}
        for col in casa_cols:
            vals = pd.to_numeric(self.patient_df[col], errors='coerce').to_numpy(float)
            lo, hi = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
            
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = np.nanmin(vals), np.nanmax(vals)
            
            # Create short label
            short_label = col.replace('CASA_', '').replace('_mean', '')
            scales[short_label] = (lo, hi)
        
        return scales


# Example usage
if __name__ == "__main__":
    # Load data
    loader = ArchetypeDataLoader("dash_data")
    
    # List archetypes
    print("\nAvailable archetypes:")
    for name in loader.get_archetype_list():
        info = loader.get_archetype_info(name)
        print(f"  {name}: {info.get('title', 'No title')}")
    
    # Get data for an archetype
    if loader.get_archetype_list():
        archetype = loader.get_archetype_list()[0]
        tracks, frames, patient, info = loader.get_archetype_data(archetype)
        
        print(f"\nData for archetype '{archetype}':")
        print(f"  Title: {info.get('title')}")
        print(f"  Participant: {info['participant_id']}")
        print(f"  Tracks: {len(tracks)}")
        print(f"  Frames: {len(frames)}")
        print(f"  Patient metrics: {len(patient)} columns")
        
        # Pick diverse tracks
        selected_tids = loader.pick_diverse_tracks(tracks, n_total=120)
        print(f"  Selected {len(selected_tids)} diverse tracks for visualization")
