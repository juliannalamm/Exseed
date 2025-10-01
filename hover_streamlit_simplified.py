import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import subprocess
import json
import plotly.graph_objects as go
from full_pipeline import (
    json_to_df,
    predict_sperm_motility,
    overlay_trajectories_on_video,
    MODEL_PATH
)
from participant_metrics import calculate_median_participant_metrics, calculate_typical_patient_feature_profile


def convert_to_h264(input_path: str, output_path: str):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


# UI Setup
st.set_page_config(page_title="Sperm Motility Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center;'>CASA Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sample data directory
SAMPLE_DATA_DIR = "sample_data_for_streamlit"

# Available sample participants
SAMPLE_PARTICIPANTS = {
    "Patient 1": "767ef2ec_v2_tracks_with_mot_params.json",
    "Patient 2": "bcb2b5c7_v1_tracks_with_mot_params.json", 
    "Patient 3": "ef5f3e74_v1_tracks_with_mot_params.json"
}

# Clear results button in sidebar
if st.sidebar.button("🗑️ Clear Results"):
    for key in ['preds', 'frame_df', 'h264_path', 'preds_csv', 'participant_id']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# =====================
# 🔹 Main Analysis Interface
# =====================


# Participant selection
col1, col2 = st.columns([2, 1])
with col1:
    selected_participant = st.selectbox(
        "Select Participant:",
        list(SAMPLE_PARTICIPANTS.keys()),
        index=0,  # Default to first participant
        help="Choose from pre-loaded sample data"
    )
with col2:
    st.info("📁 Using sample data")

# Load selected participant data
if selected_participant:
    json_filename = SAMPLE_PARTICIPANTS[selected_participant]
    json_path = os.path.join(SAMPLE_DATA_DIR, json_filename)
    video_filename = json_filename.replace("_tracks_with_mot_params.json", "_raw_video.mp4")
    video_path = os.path.join(SAMPLE_DATA_DIR, video_filename)
    
    # Check if files exist
    if os.path.exists(json_path) and os.path.exists(video_path):
        st.success(f"✅ Loaded: {selected_participant}")
        
        # Load pre-calculated data for selected participant
        # Extract actual participant ID from filename
        actual_participant_id = SAMPLE_PARTICIPANTS[selected_participant].replace('_tracks_with_mot_params.json', '')
        
        if 'participant_id' not in st.session_state or st.session_state['participant_id'] != actual_participant_id:
            # Load pre-calculated data
            pregen_data_path = os.path.join(SAMPLE_DATA_DIR, "pregenerated", f"{actual_participant_id}_complete_data.json")
            
            if os.path.exists(pregen_data_path):
                with open(pregen_data_path, 'r') as f:
                    patient_data = json.load(f)
                
                # Convert back to DataFrames
                preds = pd.DataFrame(patient_data['raw_data']['preds'])
                track_df = pd.DataFrame(patient_data['raw_data']['track_df'])
                frame_df = pd.DataFrame(patient_data['raw_data']['frame_df'])
                
                # Store in session state
                st.session_state['preds'] = preds
                st.session_state['frame_df'] = frame_df
                st.session_state['h264_path'] = patient_data['video_path']
                st.session_state['participant_id'] = actual_participant_id
                st.session_state['patient_data'] = patient_data
                st.session_state['preds_csv'] = None  # No CSV for pre-calculated data
                
                st.success(f"🚀 Loaded {selected_participant} data instantly!")
                st.rerun()
            else:
                st.error(f"❌ Pre-calculated data not found for {selected_participant}")
                st.info("💡 Run `python pregenerate_all_data.py` to create pre-calculated data")
                st.stop()
    else:
        st.error(f"❌ Sample data not found for {selected_participant}")
        st.stop()

# Optional: Upload your own data
with st.expander("📁 Upload Your Own Data", expanded=False):
    st.info("You can also upload your own JSON and video files for analysis")
    col1, col2 = st.columns(2)
    with col1:
        json_file = st.file_uploader("Upload JSON", type=["json"])
    with col2:
        video_file = st.file_uploader("Upload Video", type=["mp4"])
    
    run_btn = st.button("▶️ Run Analysis", type="primary")
    
    # Process uploaded files
    if run_btn and json_file and video_file:
        with st.spinner("Processing your data..."):
            # Save uploaded files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as jtf:
                jtf.write(json_file.read())
                json_path = jtf.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vtf:
                vtf.write(video_file.read())
                video_path = vtf.name

            participant_id = os.path.splitext(os.path.basename(json_file.name))[0]
            out_dir = os.path.join(tempfile.gettempdir(), f"{participant_id}_outputs")
            os.makedirs(out_dir, exist_ok=True)

            # STEP 1: Parse JSON
            track_df, frame_df = json_to_df(json_path, participant_id)

            # STEP 2: Predict Subtypes (no UMAP)
            preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=False)
            preds_csv = os.path.join(out_dir, f"{participant_id}_predictions.csv")
            preds.to_csv(preds_csv, index=False)

            # STEP 3: Overlay all trajectories on video
            raw_overlay_path = os.path.join(out_dir, f"{participant_id}_raw_overlay.mp4")
            h264_path = os.path.join(out_dir, f"{participant_id}_overlay_h264.mp4")

            overlay_trajectories_on_video(
                frame_df=frame_df,
                track_df=preds,
                video_path=video_path,
                output_path=raw_overlay_path
            )
            convert_to_h264(raw_overlay_path, h264_path)
            
            # Store data in session state to persist between interactions
            st.session_state['preds'] = preds
            st.session_state['frame_df'] = frame_df
            st.session_state['h264_path'] = h264_path
            st.session_state['preds_csv'] = preds_csv
            st.session_state['participant_id'] = participant_id

# Display Results (if data exists in session state)
if 'preds' in st.session_state and 'frame_df' in st.session_state:
    preds = st.session_state['preds']
    frame_df = st.session_state['frame_df']
    h264_path = st.session_state['h264_path']
    participant_id = st.session_state['participant_id']
    
    # Handle preds_csv (may not exist for pre-calculated data)
    preds_csv = st.session_state.get('preds_csv', None)
    
    # Create side-by-side layout: Video + Patient Overview
    st.subheader("👤 Patient Sample Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if os.path.exists(h264_path):
            with open(h264_path, 'rb') as video_file:
                st.video(video_file.read(), start_time=0)
        else:
            st.error("Video overlay not found.")
    
    with col2:
        
        # Calculate comparison data for overview
        try:
            # Calculate median metrics for comparison
            median_metrics, metrics_df = calculate_median_participant_metrics()
            
            # Fix: Calculate percentages from median counts to ensure they add up to 100%
            subtype_percentages = {
                'progressive': median_metrics['progressive'],
                'vigorous': median_metrics['vigorous'], 
                'immotile': median_metrics['immotile'],
                'nonprogressive': median_metrics['nonprogressive']
            }
            
            # Normalize to ensure they sum to 100%
            total_percentage = sum(subtype_percentages.values())
            if total_percentage > 0:
                for subtype in subtype_percentages:
                    subtype_percentages[subtype] = (subtype_percentages[subtype] / total_percentage) * 100
            
            # Calculate current participant metrics
            current_total_tracks = len(preds)
            current_subtype_counts = preds['subtype_label'].value_counts()
            current_percentages = {}
            
            for subtype in ['progressive', 'vigorous', 'immotile', 'nonprogressive']:
                count = current_subtype_counts.get(subtype, 0)
                percentage = (count / current_total_tracks) * 100
                current_percentages[subtype] = percentage
            
            # Define colors and inline Lucide SVG icons for each subtype (matching Dash component colors)
            def lucide_svg(icon_path: str, color_hex: str, size: int = 16) -> str:
                return f"""
                <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color_hex}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    {icon_path}
                </svg>
                """.strip()
            LUCIDE_PATHS = {
                'fast-forward': '<polygon points="13 19 22 12 13 5 13 19"/><polygon points="2 19 11 12 2 5 2 19"/>',
                'zap': '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
                'move': '<polyline points="5 9 2 12 5 15"/><polyline points="9 5 12 2 15 5"/><polyline points="15 19 12 22 9 19"/><polyline points="19 9 22 12 19 15"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/>',
                'pause-circle': '<circle cx="12" cy="12" r="10"/><line x1="10" y1="15" x2="10" y2="9"/><line x1="14" y1="15" x2="14" y2="9"/>',
                'turtle': '<path d="m12 10 2 4v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3a8 8 0 1 0-16 0v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3l2-4h4Z"/><path d="M4.82 7.9 8 10"/><path d="M15.18 7.9 12 10"/><path d="M16.93 10H20a2 2 0 0 1 0 4H2"/>',
                'smile': '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
                'meh': '<circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
                'frown': '<circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>'
            }

            subtype_info = {
                'progressive': {'color': '#636EFA', 'icon_svg': lucide_svg(LUCIDE_PATHS['fast-forward'], '#636EFA', 24), 'name': 'Progressive'},
                'vigorous': {'color': '#EF553B', 'icon_svg': lucide_svg(LUCIDE_PATHS['zap'], '#EF553B', 24), 'name': 'Vigorous'},
                'nonprogressive': {'color': '#00CC96', 'icon_svg': lucide_svg(LUCIDE_PATHS['turtle'], '#00CC96', 24), 'name': 'Nonprogressive'},
                'immotile': {'color': '#FFA15A', 'icon_svg': lucide_svg(LUCIDE_PATHS['pause-circle'], '#FFA15A', 24), 'name': 'Immotile'}
            }
            
            # Create overview cards (stacked vertically)
            for subtype, info in subtype_info.items():
                current_pct = current_percentages[subtype]
                typical_pct = subtype_percentages[subtype]
                difference = current_pct - typical_pct
                
                # Determine if this is good or bad based on subtype
                if subtype in ['progressive', 'vigorous']:
                    # Higher is better
                    if difference > 0:
                        status = "Above Typical"
                        status_color = "#2ca02c"
                    else:
                        status = "Below Typical"
                        status_color = "#d62728"
                else:
                    # Lower is better
                    if difference < 0:
                        status = "Better Than Typical"
                        status_color = "#2ca02c"
                    else:
                        status = "Above Typical"
                        status_color = "#d62728"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {info['color']}20, {info['color']}10);
                    border: 2px solid {info['color']};
                    border-radius: 10px;
                    padding: 8px;
                    margin: 4px 0;
                ">
                    <div style="display: flex; align-items: center; gap: 8px; justify-content: center;">
                        <div>{info['icon_svg']}</div>
                        <div style="color: {info['color']}; font-weight: 600; font-size: 14px;">{info['name']}</div>
                    </div>
                    <div style="display: flex; align-items: baseline; justify-content: center; gap: 6px; margin-top: 6px;">
                        <div style="color: {info['color']}; font-size: 22px; font-weight: 700;">{current_pct:.1f}%</div>
                        <div style="color: #666; font-size: 11px;">vs {typical_pct:.1f}% typical</div>
                    </div>
                    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 6px;">
                        <div style="color: {status_color}; font-size: 12px; font-weight: 600;">{status}</div>
                        <div style="color: {info['color']}; font-size: 12px; font-weight: 600;">{difference:+.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            
        except Exception as e:
            st.warning(f"⚠️ Could not load patient overview: {str(e)}")
    
    # Track Analysis and Overall Assessment (below video)
    st.markdown("---")
    
    # Calculate data for track analysis and overall assessment
    try:
        # Calculate median metrics for comparison
        median_metrics, metrics_df = calculate_median_participant_metrics()
        
        # Fix: Calculate percentages from median counts to ensure they add up to 100%
        subtype_percentages = {
            'progressive': median_metrics['progressive'],
            'vigorous': median_metrics['vigorous'], 
            'immotile': median_metrics['immotile'],
            'nonprogressive': median_metrics['nonprogressive']
        }
        
        # Normalize to ensure they sum to 100%
        total_percentage = sum(subtype_percentages.values())
        if total_percentage > 0:
            for subtype in subtype_percentages:
                subtype_percentages[subtype] = (subtype_percentages[subtype] / total_percentage) * 100
        
        # Calculate current participant metrics
        current_total_tracks = len(preds)
        current_subtype_counts = preds['subtype_label'].value_counts()
        current_percentages = {}
        
        for subtype in ['progressive', 'vigorous', 'immotile', 'nonprogressive']:
            count = current_subtype_counts.get(subtype, 0)
            percentage = (count / current_total_tracks) * 100
            current_percentages[subtype] = percentage
        
        # Overall Assessment (full width above track analysis)
        st.markdown("#### 🎯 Overall Assessment")
        
        # Find dominant motility type
        dominant_type = max(current_percentages, key=current_percentages.get)
        dominant_percentage = current_percentages[dominant_type]
        
        # Define dominant type messages
        dominant_messages = {
            'immotile': "Most of your sperm are not moving",
            'nonprogressive': "Most of your sperm are slow movers - this might lead to poor fertilization outcomes",
            'vigorous': "Your sperm move very quickly, this could lead to positive fertilization outcomes",
            'progressive': "Your sperm move quickly and in a straight line, this helps the sperm find the egg and can help to positive fertilization outcomes"
        }
        
        # Calculate overall assessment
        good_differences = 0
        total_differences = 0
        
        for subtype in ['progressive', 'vigorous']:
            if current_percentages[subtype] > subtype_percentages[subtype]:
                good_differences += 1
        for subtype in ['immotile', 'nonprogressive']:
            if current_percentages[subtype] < subtype_percentages[subtype]:
                good_differences += 1
                
        # Determine status, color, and icon
        if good_differences >= 3:
            overall_status = "Excellent - Above average motility profile"
            status_color = "#2ca02c"
            status_icon_svg = lucide_svg(LUCIDE_PATHS['smile'], status_color, 20)
        elif good_differences >= 2:
            overall_status = "Good - Mixed motility profile"
            status_color = "#ff7f0e"
            status_icon_svg = lucide_svg(LUCIDE_PATHS['meh'], status_color, 20)
        else:
            overall_status = "Below Average - Lower motility profile"
            status_color = "#d62728"
            status_icon_svg = lucide_svg(LUCIDE_PATHS['frown'], status_color, 20)
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {status_color}20, {status_color}10);
            border: 1px solid {status_color};
            border-radius: 8px;
            padding: 8px;
            text-align: center;
            margin: 4px 0;
        ">
            <h4 style="margin: 0; color: {status_color}; font-size: 22px; display:flex; align-items:center; justify-content:center; gap:8px;">
                <span>Overall Assessment</span>
            </h4>
            <p style="margin: 4px 0; color: {status_color}; font-size: 18px; display:flex; align-items:center; justify-content:center; gap:8px;">
                <span>{status_icon_svg}</span>
                <span>{overall_status}</span>
            </p>
            <div style="
                background: linear-gradient(135deg, {status_color}22, {status_color}10);
                border-radius: 6px;
                padding: 6px;
                margin: 6px 0;
                border: 1px solid {status_color}55;
            ">
                <p style="margin: 0; color: #0f172a; font-size: 16px; font-weight: 600;">
                    {dominant_type.title()} ({dominant_percentage:.1f}%)
                </p>
                <p style="margin: 2px 0; color: #111827; font-size: 14px;">
                    {dominant_messages[dominant_type]}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Track Analysis (full width below overall assessment)
        st.markdown("---")
        st.markdown("#### 📊 Track Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Patient Tracks",
                value=f"{current_total_tracks:,}",
                delta=f"{current_total_tracks - 100:+.0f}",
                help="Number of sperm tracks analyzed"
            )
        
        with col2:
            st.metric(
                label="Typical Patient Tracks",
                value="100",
                delta=None,
                help="Average number of tracks per participant"
            )
        
        with col3:
            if current_total_tracks > 100:
                track_status = "📈 Above Average"
                track_color = "#2ca02c"
            elif current_total_tracks < 50:
                track_status = "📉 Low Sample Size"
                track_color = "#d62728"
            else:
                track_status = "✅ Good Sample Size"
                track_color = "#2ca02c"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {track_color}20, {track_color}10);
                border: 2px solid {track_color};
                border-radius: 8px;
                padding: 12px;
                text-align: center;
                margin: 6px 0;
            ">
                <p style="margin: 0; color: {track_color}; font-size: 18px; font-weight: bold;">
                    {track_status}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"⚠️ Could not load track analysis and assessment: {str(e)}")
    
    # Percent Breakdown of Motility Types
    st.markdown("---")
    st.subheader("📊 Percent Breakdown of Motility Types")
    try:
        # Calculate median metrics
        median_metrics, metrics_df = calculate_median_participant_metrics()
        
        # Fix: Calculate percentages from median counts to ensure they add up to 100%
        # Get the median total tracks
        median_total_tracks = median_metrics['total_tracks']
        
        # Calculate what the median counts would be for each subtype
        # We need to estimate the median counts, not use the median percentages
        # For now, let's use the median percentages but normalize them to sum to 100%
        subtype_percentages = {
            'progressive': median_metrics['progressive'],
            'vigorous': median_metrics['vigorous'], 
            'immotile': median_metrics['immotile'],
            'nonprogressive': median_metrics['nonprogressive']
        }
        
        # Normalize to ensure they sum to 100%
        total_percentage = sum(subtype_percentages.values())
        if total_percentage > 0:
            for subtype in subtype_percentages:
                subtype_percentages[subtype] = (subtype_percentages[subtype] / total_percentage) * 100
        
        # Calculate current participant metrics
        current_total_tracks = len(preds)
        current_subtype_counts = preds['subtype_label'].value_counts()
        current_percentages = {}
        
        for subtype in ['progressive', 'vigorous', 'immotile', 'nonprogressive']:
            count = current_subtype_counts.get(subtype, 0)
            percentage = (count / current_total_tracks) * 100
            current_percentages[subtype] = percentage
        
        # Create stacked horizontal bar chart
        fig = go.Figure()
        
        # Map metric names to colors (matching Dash component colors)
        metric_colors = {
            'progressive': '#636EFA',      # blue
            'vigorous': '#EF553B',         # red
            'nonprogressive': '#00CC96',   # teal/green
            'immotile': '#FFA15A'          # orange
        }
        
        # Add bars for current participant (stacked)
        metrics_order = ['progressive', 'vigorous', 'nonprogressive', 'immotile']
        metric_names = ['Progressive', 'Vigorous', 'Nonprogressive', 'Immotile']
        
        for i, (metric, name) in enumerate(zip(metrics_order, metric_names)):
            fig.add_trace(go.Bar(
                y=['Your Patient'],
                x=[current_percentages[metric]],
                orientation='h',
                name=name,
                marker_color=metric_colors[metric],
                text=f'{current_percentages[metric]:.1f}%',
                textposition='inside',
                textfont=dict(color='white', size=10),
                showlegend=True
            ))
        
        # Add bars for median values (separate row)
        for i, (metric, name) in enumerate(zip(metrics_order, metric_names)):
            fig.add_trace(go.Bar(
                y=['Typical Patient'],
                x=[subtype_percentages[metric]],
                orientation='h',
                name=name,
                marker_color=metric_colors[metric],
                text=f'{subtype_percentages[metric]:.1f}%',
                textposition='inside',
                textfont=dict(color='white', size=10),
                showlegend=False,
                opacity=0.7
            ))
        
        fig.update_layout(
            title="% Breakdown of Motility Types vs. Typical (median)",
            xaxis_title="Percentage (%)",
            barmode='stack',
            height=300,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="stacked_bar")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load metrics comparison: {str(e)}")
    
    # Feature Profile - Radar Charts
    st.subheader("📈 Feature Profile: Current vs Typical Patient")
    try:
        # Calculate typical patient feature profile (overall average)
        typical_profile, features_df = calculate_typical_patient_feature_profile()
        
        # Calculate current participant's overall average feature values (across all tracks)
        all_features = ['ALH', 'BCF', 'LIN', 'VCL', 'VSL', 'WOB', 'MAD', 'STR', 'VAP']
        current_values = {}
        typical_values = {}
        
        for feature in all_features:
            if feature in preds.columns:
                current_values[feature] = preds[feature].mean()
            else:
                current_values[feature] = 0
            
            # Get typical value (overall average across all clusters)
            if feature in features_df.columns:
                typical_values[feature] = features_df[feature].mean()
            else:
                typical_values[feature] = 0
        
        # Create single combined radar chart with all features
        all_features = ['VCL', 'VSL', 'VAP', 'LIN', 'STR', 'WOB', 'ALH', 'BCF', 'MAD']
        current_all = []
        typical_all = []
        
        # Normalize all features to 0-1 scale using the same method as comparison component
        for feature in all_features:
            current_val = current_values[feature]
            typical_val = typical_values[feature]
            
            # Apply same normalization as comparison component
            if feature in ['VCL', 'VAP', 'VSL']:  # Velocity metrics
                current_normalized = min(current_val / 200.0, 1.0)  # Assume max ~200
                typical_normalized = min(typical_val / 200.0, 1.0)
            elif feature == 'ALH':
                current_normalized = min(current_val / 10.0, 1.0)  # Assume max ~10
                typical_normalized = min(typical_val / 10.0, 1.0)
            elif feature == 'BCF':
                current_normalized = min(current_val / 20.0, 1.0)  # Assume max ~20
                typical_normalized = min(typical_val / 20.0, 1.0)
            elif feature == 'MAD':
                current_normalized = min(current_val / 5.0, 1.0)  # Assume max ~5
                typical_normalized = min(typical_val / 5.0, 1.0)
            else:  # LIN, STR, WOB are already 0-1
                current_normalized = min(current_val, 1.0)
                typical_normalized = min(typical_val, 1.0)
            
            current_all.append(current_normalized)
            typical_all.append(typical_normalized)
        
        # Create single radar chart
        fig = go.Figure()
        
        # Add typical patient (grey, behind)
        fig.add_trace(go.Scatterpolar(
            r=typical_all + [typical_all[0]],  # Close the loop
            theta=all_features + [all_features[0]],
            fill='toself',
            name='Typical Patient',
            fillcolor='rgba(150, 150, 150, 0.15)',
            line=dict(color='rgba(150, 150, 150, 0.6)', width=2, dash='dash'),
            hovertemplate='<b>Typical Patient</b><br>%{theta}: %{r:.1%}<extra></extra>',
        ))
        
        # Add current patient (blue, on top)
        fig.add_trace(go.Scatterpolar(
            r=current_all + [current_all[0]],  # Close the loop
            theta=all_features + [all_features[0]],
            fill='toself',
            name='Current Patient',
            fillcolor='rgba(99, 110, 250, 0.3)',
            line=dict(color='#636EFA', width=2.5),
            hovertemplate='<b>Current Patient</b><br>%{theta}: %{r:.1%}<extra></extra>',
        ))
        
        # Determine range
        max_val = max(max(current_all), max(typical_all)) if current_all and typical_all else 1.0
        range_max = max(1.0, max_val * 1.1)
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=False,  # Hide radial axis completely
                    range=[0, range_max],
                ),
                angularaxis=dict(
                    tickfont=dict(size=11),
                ),
            ),
            title="Overall Feature Profile Comparison",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Radar chart with hoverable info
        col1, col2 = st.columns([1, 20])
        with col1:
            st.markdown("📊")
        with col2:
            st.plotly_chart(fig, use_container_width=True, key="combined_radar")
        
        # Hoverable info about radar chart
        with st.expander("ℹ️ How to interpret this radar chart", expanded=False):
            st.markdown("""
            **📊 Radar Chart Interpretation:**
            - **Percentages (0-100%)** represent normalized feature values for fair comparison
            - **Velocity features** (VCL, VSL, VAP): Normalized by dividing by 200 μm/s
            - **ALH**: Normalized by dividing by 10 μm  
            - **BCF**: Normalized by dividing by 20 Hz
            - **MAD**: Normalized by dividing by 5 μm
            - **Linearity features** (LIN, STR, WOB): Already 0-1 scale
            - **Blue area**: Current patient's feature profile
            - **Grey dashed area**: Typical patient's feature profile
            """)
        
        # Show raw values in a table
        with st.expander("📊 All Raw Feature Values", expanded=False):
            comparison_data = []
            for feature in all_features:
                comparison_data.append({
                    'Feature': feature,
                    'Current Patient': f"{current_values[feature]:.3f}",
                    'Typical Patient': f"{typical_values[feature]:.3f}",
                    'Difference': f"{current_values[feature] - typical_values[feature]:+.3f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        
    except Exception as e:
        st.warning(f"⚠️ Could not load feature comparison: {str(e)}")
    
    
    # Download results
    st.subheader("📥 Download Results")
    if preds_csv and os.path.exists(preds_csv):
        with open(preds_csv, 'r') as f:
            st.download_button(
                label="📊 Download Predictions CSV",
                data=f.read(),
                file_name=f"{participant_id}_predictions.csv",
                mime="text/csv"
            )

    elif preds_csv is None:
        # For pre-calculated data, create CSV on demand
        csv_data = preds.to_csv(index=False)
        st.download_button(
            label="📊 Download Predictions CSV",
            data=csv_data,
            file_name=f"{participant_id}_predictions.csv",
            mime="text/csv"
        )
    elif run_btn and (not json_file or not video_file):
        st.error("❌ Please upload both JSON and video files to run analysis.")
