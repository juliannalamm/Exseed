# Side-by-Side Comparison Feature

## Overview

The new comparison section provides a beautiful side-by-side visualization comparing:
- **Felipe Data** (reference dataset) 
- **Participant 158d356b** (individual archetype)

## Features

### 1. **Radar Charts**
- Fixed radar chart rings to properly span the entire chart
- Dynamic range adjustment based on data values
- GMM composition visualization for both datasets
- Beautiful color-coded styling:
  - 🔵 Blue for Felipe reference data
  - 🔴 Red for Participant 158d356b

### 2. **Animated Trajectories**
- Interactive animations showing 120 representative trajectories
- Synchronized playback controls (Play/Pause)
- Frame-by-frame slider for detailed inspection
- Centered trajectories starting at origin
- Equal aspect ratio for accurate shape representation

### 3. **Responsive Layout**
- Two-column grid layout
- Matching styling and borders
- Consistent sizing across all components

## Components Created

### `comparison_component.py`
- `create_comparison_section()` - Main comparison layout
- `create_animated_trajectories()` - Trajectory animation generator
- `register_comparison_callbacks()` - Dash callbacks for interactivity
- `load_felipe_data()` - Felipe data loader

### Updated Components
- `radar_component.py` - Fixed radar range calculation
- `dash_app.py` - Integrated comparison section
- `__init__.py` - Exported new components

## Usage

The comparison section automatically loads below the "Generating Continuous Motility Scores" section.

### Data Requirements
- **Felipe data**: `felipe_data/fid_level_data.csv` and `felipe_data/trajectory.csv`
- **Participant data**: Exported archetype data in `dash_data/`
- Participant 158d356b must be archetype 'A' in the exported config

## Visualization Details

### Radar Charts
- **Range**: Dynamically adjusted to `max(1.0, max_value * 1.15)`
- **Labels**: GMM clusters (Progressive, Rapid Progressive, etc.)
- **Values**: Mean posterior probabilities across all tracks

### Trajectory Animations
- **Track count**: 120 randomly sampled tracks per dataset
- **Frame rate**: 50ms per frame
- **Coordinates**: Centered at origin (Δx, Δy in μm)
- **Y-axis**: Inverted for conventional microscopy orientation

## Styling

### Color Scheme
- Felipe data: `rgba(99, 110, 250, ...)` (Blue)
- Participant: `rgba(239, 85, 59, ...)` (Red)
- Background: `rgba(26, 26, 26, 0.5)` (Semi-transparent dark)
- Borders: Color-matched with 30% opacity

### Layout
- Grid: `1fr 1fr` (equal columns)
- Gap: 30px
- Padding: 20px per card
- Border radius: 12px
- Border width: 2px

## Technical Notes

### Performance
- Trajectories are pre-computed (not real-time)
- Animation uses Plotly's built-in frame system
- Data is loaded once on component mount

### Browser Compatibility
- Tested on modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- Animations use CSS3 and HTML5 features

## Future Enhancements

Possible improvements:
- [ ] Add speed control for animations
- [ ] Export animation as video
- [ ] Compare multiple participants
- [ ] Add statistical comparison metrics
- [ ] Overlay trajectory heatmaps
