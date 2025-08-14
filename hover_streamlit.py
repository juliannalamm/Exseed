import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import subprocess
import plotly.graph_objects as go
from full_pipeline import (
    json_to_df,
    predict_sperm_motility,
    plot_umap_with_predictions,
    overlay_trajectories_on_video,
    MODEL_PATH
)
from participant_metrics import calculate_median_participant_metrics, get_metric_status, calculate_typical_patient_feature_profile
from global_umap_utils import ( 
    load_or_create_training_umap_data,  
    get_feature_analysis,
    create_single_feature_plot,
    get_global_umap_comparison, cluster_colors
    )


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
st.title("Motility Classification Results (400 Patients)")

# Clear results button in sidebar
if st.sidebar.button("🗑️ Clear Results"):
    for key in ['preds', 'frame_df', 'h264_path', 'preds_csv', 'participant_id']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Global variables for upload state
json_file = None
video_file = None
run_btn = False

# =====================
# 🔹 Tabbed Analysis Interface
# =====================
tab1, tab2 = st.tabs(["🌍 Pooled Data Explorer", "📊 Individual vs. Population Analysis"])

with tab1:
    st.subheader("🌍 Gaussian Mixture Model Clustering of Pooled Data ")
    
    # Load training data to get available participants
    training_data = load_or_create_training_umap_data()
    
    if training_data is not None:
        # Create global UMAP plot showing only training data
        with st.spinner("Creating global UMAP comparison..."):
            global_fig, comparison_stats = get_global_umap_comparison(None, participant_id=None)
        
        if comparison_stats:
            st.plotly_chart(global_fig, use_container_width=True)
   
            
            # Calculate percentages
            total_tracks = comparison_stats['training_total']
            dist_data = comparison_stats['training_distribution']
            
           
            
            # Get feature analysis for cutoffs
            feature_stats, cutoffs = get_feature_analysis(training_data)
            
            # Display percentages in columns with expandable criteria under each
            cols = st.columns(4)
            
            for i, (subtype, count) in enumerate(dist_data.items()):
                percentage = (count / total_tracks) * 100
                
                with cols[i]:
                    st.markdown(f"### {subtype.title()}")
                    st.markdown(f"**{count:,} tracks**")
                    st.markdown(f"## **{percentage:.1f}%**")
                    
                    # Create expandable section for this cluster's criteria
                    if cutoffs:
                        with st.expander(f"🎯 **{subtype.title()} Cut-offs**", expanded=False):
                     
                            
                            # Get actual cutoff values from the calculated cutoffs
                            key_features = ['VCL', 'ALH', 'VSL', 'LIN', 'BCF', 'WOB', 'MAD', 'STR', 'VAP']
                            cutoff_data = []
                            
                            for feature in key_features:
                                if feature in cutoffs and subtype in cutoffs[feature]:
                                    cutoff_value = cutoffs[feature][subtype]
                                    cutoff_data.append({
                                        'Feature': feature,
                                        'Cutoff': cutoff_value
                                    })
                                else:
                                    cutoff_data.append({
                                        'Feature': feature,
                                        'Cutoff': 'N/A'
                                    })
                            
                            # Create DataFrame for table
                            cutoff_df = pd.DataFrame(cutoff_data)
                            
                            # Display as a compact table
                            st.dataframe(cutoff_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("*No cutoff data*")
            
            # Show total tracks info
            st.markdown(f"**Total Tracks:** {total_tracks:,} | **Data Source:** train_track_df.csv")
            
        else:
            st.plotly_chart(global_fig, use_container_width=True)
            st.warning("⚠️ Could not load training data.")
        
        # Feature Analysis Sectio
        
        # Feature Statistics Bar Chart with Tabs
        st.subheader("📊 Feature Distribution Analysis")
        
        with st.spinner("Analyzing feature distributions..."):
            feature_stats, cutoffs = get_feature_analysis(training_data)
            
            if feature_stats:
                # Get all features and clusters
                features = ["ALH", "BCF", "LIN", "VCL", "VSL", "WOB", "MAD", "STR", "VAP"]
                clusters = list(feature_stats.keys())
                
                # Create data for the bar chart
                chart_data = []
                for cluster in clusters:
                    for feature in features:
                        if feature in feature_stats[cluster]:
                            stat = feature_stats[cluster][feature]
                            chart_data.append({
                                'Cluster': cluster,
                                'Feature': feature,
                                'Mean': stat['mean']
                            })
                
                # Create DataFrame for plotting
                chart_df = pd.DataFrame(chart_data)
                
                # Create tabs for raw vs normalized values
                raw_tab, norm_tab = st.tabs(["Raw Values", "Normalized Values"])
                
                with raw_tab:
                    # Create columns for the raw value charts
                    col1, col2 = st.columns(2)
                    
                    cluster_name_colors = {
                        'vigorous': '#1f77b4',      # blue
                        'progressive': '#2ca02c',   # green
                        'nonprogressive': '#ff7f0e', # red
                        'immotile': '#d62728'       # blue
                    }
                    
                    with col1:
                        # Main chart with VCL, VSL, MAD, VAP, BCF
                        main_features = ["VCL", "VSL", "MAD", "VAP", "BCF"]
                        chart_df_main = chart_df[chart_df['Feature'].isin(main_features)]
                        
                        fig_raw_main = px.bar(
                            chart_df_main,
                            x='Feature',
                            y='Mean',
                            color='Cluster',
                            title='Raw Mean Feature Values',
                            barmode='group',
                            color_discrete_map=cluster_name_colors
                        )
                        
                        fig_raw_main.update_layout(
                            xaxis_title="Feature",
                            yaxis_title="Mean Value",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_raw_main, use_container_width=True, key="raw_main")
                    
                    with col2:
                        # Secondary chart with WOB, STR, LIN, ALH
                        secondary_features = ["WOB", "STR", "LIN", "ALH"]
                        chart_df_secondary = chart_df[chart_df['Feature'].isin(secondary_features)]
                        
                        fig_raw_secondary = px.bar(
                            chart_df_secondary,
                            x='Feature',
                            y='Mean',
                            color='Cluster',
                            title='Raw Mean Feature Values',
                            barmode='group',
                            color_discrete_map=cluster_name_colors
                        )
                        
                        fig_raw_secondary.update_layout(
                            xaxis_title="Feature",
                            yaxis_title="Mean Value",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_raw_secondary, use_container_width=True, key="raw_secondary")
                
                with norm_tab:
                    # Create columns for the normalized value charts
                    col1, col2 = st.columns(2)
                    
                    # Create normalized data
                    chart_df_norm = chart_df.copy()
                    
                    # Normalize the values by feature (z-score normalization)
                    for feature in features:
                        feature_data = chart_df_norm[chart_df_norm['Feature'] == feature]['Mean']
                        if len(feature_data) > 0:
                            mean_val = feature_data.mean()
                            std_val = feature_data.std()
                            if std_val > 0:
                                chart_df_norm.loc[chart_df_norm['Feature'] == feature, 'Mean'] = (feature_data - mean_val) / std_val
                    
                    # Define colors explicitly by cluster names
                    cluster_name_colors = {
                        'vigorous': '#1f77b4',      # blue
                        'progressive': '#2ca02c',   # green
                        'nonprogressive': '#ff7f0e', # red
                        'immotile': '#d62728'       # blue
                    }
                    
                    with col1:
                        # Main chart with VCL, VSL, MAD, VAP, BCF
                        main_features = ["VCL", "VSL", "MAD", "VAP", "BCF"]
                        chart_df_norm_main = chart_df_norm[chart_df_norm['Feature'].isin(main_features)]
                        
                        fig_norm_main = px.bar(
                            chart_df_norm_main,
                            x='Feature',
                            y='Mean',
                            color='Cluster',
                            title='Normalized Mean Feature Values',
                            barmode='group',
                            color_discrete_map=cluster_name_colors,
                        )
                        
                        fig_norm_main.update_layout(
                            xaxis_title="Feature",
                            yaxis_title="Normalized Mean Value (Z-score)",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_norm_main, use_container_width=True, key="norm_main")
                    
                    with col2:
                        # Secondary chart with WOB, STR, LIN, ALH
                        secondary_features = ["WOB", "STR", "LIN", "ALH"]
                        chart_df_norm_secondary = chart_df_norm[chart_df_norm['Feature'].isin(secondary_features)]
                        
                        fig_norm_secondary = px.bar(
                            chart_df_norm_secondary,
                            x='Feature',
                            y='Mean',
                            color='Cluster',
                            title='Normalized Mean Feature Values',
                            barmode='group',
                            color_discrete_map=cluster_name_colors
                        )
                        
                        fig_norm_secondary.update_layout(
                            xaxis_title="Feature",
                            yaxis_title="Normalized Mean Value (Z-score)",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_norm_secondary, use_container_width=True, key="norm_secondary")
        
        
        # Create tabs for different features
        feature_tabs = st.tabs(["ALH", "BCF", "LIN", "VCL", "VSL", "WOB", "MAD", "STR", "VAP"])
        
        for i, feature in enumerate(["ALH", "BCF", "LIN", "VCL", "VSL", "WOB", "MAD", "STR", "VAP"]):
            with feature_tabs[i]:
                # Create and display the individual feature plot
                feature_fig = create_single_feature_plot(training_data, feature)
                st.plotly_chart(feature_fig, use_container_width=True)
            

                       
    else:
        st.error("❌ Training data not found. Please ensure train_track_df.csv is available.")

with tab2:
    st.subheader("📊 Individual vs. Population Analysis")
    
    # Upload section

    col1, col2 = st.columns(2)
    with col1:
        json_file = st.file_uploader("Upload JSON", type=["json"])
    with col2:
        video_file = st.file_uploader("Upload Video", type=["mp4"])
    
    run_btn = st.button("▶️ Run Analysis", type="primary")
    
    # Process Inputs
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

            # STEP 2: Predict Subtypes + UMAP
            preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=True)
            preds_csv = os.path.join(out_dir, f"{participant_id}_predictions.csv")
            preds.to_csv(preds_csv, index=False)

            # STEP 3: Static PNG (optional fallback)
            umap_png = os.path.join(out_dir, f"{participant_id}_umap.png")
            plot_umap_with_predictions(preds, umap_png)

            # STEP 4: Overlay all trajectories on video
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
        preds_csv = st.session_state['preds_csv']
        participant_id = st.session_state['participant_id']
        
        # UMAP Visualization
        
        if 'umap_1' in preds.columns and 'umap_2' in preds.columns:
            # Create a combined plot showing training data + new data
            
            # Load training data for comparison
            training_data = load_or_create_training_umap_data()
            if training_data is not None:
                # Create a combined plot showing both training and new data
              
                
                # Create the plot
                fig = go.Figure()
                
                # Add training data points (smaller, more transparent)
                for cluster_id in sorted(training_data['cluster_id'].unique()):
                    cluster_data = training_data[training_data['cluster_id'] == cluster_id]
                    subtype = cluster_data['subtype_label'].iloc[0]
                    
                    fig.add_trace(go.Scatter(
                        x=cluster_data['umap_1'],
                        y=cluster_data['umap_2'],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=cluster_colors.get(cluster_id, '#888'),
                            opacity=0.3
                        ),
                        name=f'Pooled data: {subtype}',
                        showlegend=True,
                        hovertemplate='<b>Training Data</b><br>' +
                                     f'Subtype: {subtype}<br>' +
                                     'Cluster: %{customdata}<br>' +
                                     '<extra></extra>',
                        customdata=cluster_data['cluster_id']
                    ))
                
                # Add new participant data (larger, more prominent)
                for cluster_id in sorted(preds['cluster_id'].unique()):
                    cluster_data = preds[preds['cluster_id'] == cluster_id]
                    subtype = cluster_data['subtype_label'].iloc[0]
                    
                    fig.add_trace(go.Scatter(
                        x=cluster_data['umap_1'],
                        y=cluster_data['umap_2'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=cluster_colors.get(cluster_id, '#444'),
                            opacity=1.0,
                            line=dict(color='white', width=1),
                        ),
                        name=f'Patient data: {subtype}',
                        showlegend=True,
                        hovertemplate='<b>Your Participant</b><br>' +
                                     'Track: %{text}<br>' +
                                     f'Subtype: {subtype}<br>' +
                                     'Cluster: %{customdata}<br>' +
                                     '<extra></extra>',
                        text=cluster_data['track_id'],
                        customdata=cluster_data['cluster_id']
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{participant_id}  vs. Global Distribution",
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True, key="global_umap")
                st.info("💡 **Global UMAP**: Participant sperm are highlighted as larger white-outlined circles. Double-click on legend points to isolate and view data")
            else:
                # Fallback to individual UMAP if training data not available
                fig = px.scatter(
                    preds,
                    x='umap_1',
                    y='umap_2',
                    color='subtype_label' if 'subtype_label' in preds else 'cluster_id',
                    hover_data=['track_id'],
                    title="Individual UMAP (Training data not available)",
                    width=400,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key="individual_umap")
                st.warning("⚠️ Training data not available, showing individual UMAP only.")
        else:
            st.warning("UMAP coordinates not available.")
        
        # Video display below UMAP

        total_tracks = len(preds['track_id'].unique())
        st.markdown(f"**Total Tracks Analyzed:** {total_tracks}")
        
        if os.path.exists(h264_path):
            with open(h264_path, 'rb') as video_file:
                st.video(video_file.read(), start_time=0)
        else:
            st.error("Video overlay not found.")
        

        
        # Stacked Bar Chart Comparison
        try:
            
            # Calculate median metrics
            median_metrics, metrics_df = calculate_median_participant_metrics()
            
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
            
            # Map metric names to cluster IDs for color consistency
            metric_to_cluster = {
                'progressive': 2,  # green cluster
                'vigorous': 0,     # orange cluster  
                'immotile': 1,     # blue cluster
                'nonprogressive': 3 # red cluster
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
                    marker_color=cluster_colors[metric_to_cluster[metric]],
                    text=f'{current_percentages[metric]:.1f}%',
                    textposition='inside',
                    textfont=dict(color='white', size=10),
                    showlegend=True
                ))
            
            # Add bars for median values (separate row)
            for i, (metric, name) in enumerate(zip(metrics_order, metric_names)):
                fig.add_trace(go.Bar(
                    y=['Typical Patient'],
                    x=[median_metrics[metric]],
                    orientation='h',
                    name=name,
                    marker_color=cluster_colors[metric_to_cluster[metric]],
                    text=f'{median_metrics[metric]:.1f}%',
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
            
        except Exception as e:
            st.warning(f"⚠️ Could not load metrics comparison: {str(e)}")
        
        # Cluster Distribution Summary
        with st.expander("📊 Cluster Distribution Summary", expanded=False):
            try:
                # Calculate median metrics for comparison
                median_metrics, metrics_df = calculate_median_participant_metrics()
                
                if "subtype_label" in preds.columns:
                    subtype_counts = preds["subtype_label"].value_counts()
                    total_tracks = len(preds)
                    
                    # Calculate comparison data
                    comparison_data = []
                    for subtype in subtype_counts.index:
                        count = subtype_counts[subtype]
                        percentage = (count / total_tracks * 100).round(1)
                        
                        # Get median value for comparison
                        median_value = median_metrics.get(subtype, 0)
                        if median_value > 0:
                            percent_diff = percentage - median_value
                            if percent_diff > 0:
                                diff_text = f"+{percent_diff:.1f}%"
                            else:
                                diff_text = f"{percent_diff:.1f}%"
                        else:
                            diff_text = "N/A"
                        
                        # Determine arrow color
                        if subtype in ['progressive', 'vigorous']:
                            if percentage > median_value:
                                arrow = "🟢"
                            else:
                                arrow = "🔴"
                        else:
                            if percentage < median_value:
                                arrow = "🟢"
                            else:
                                arrow = "🔴"
                        
                        comparison_data.append({
                            'Subtype': subtype,
                            'Count': count,
                            'Percentage': percentage,
                            'vs Typical': f"{arrow} {diff_text}"
                        })
                    
                    subtype_df = pd.DataFrame(comparison_data)
                    st.dataframe(subtype_df, use_container_width=True, hide_index=True)
                else:
                    cluster_counts = preds["cluster_id"].value_counts()
                    total_tracks = len(preds)
                    cluster_df = pd.DataFrame({
                        'Cluster': cluster_counts.index,
                        'Count': cluster_counts.values,
                        'Percentage': (cluster_counts.values / total_tracks * 100).round(1)
                    })
                    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"⚠️ Could not load cluster distribution: {str(e)}")
        
        # Feature Comparison by Cluster
        st.markdown("**Average Feature Values by Cluster vs. Typical (Median) Feature Values by Cluster**")
        try:
            # Calculate typical patient feature profile
            typical_profile, features_df = calculate_typical_patient_feature_profile()
            
            # Calculate current participant's average feature values per cluster
            current_participant_features = {}
            features = ['ALH', 'BCF', 'LIN', 'VCL', 'VSL', 'WOB', 'MAD', 'STR', 'VAP']
            
            for cluster in preds['subtype_label'].unique():
                cluster_data = preds[preds['subtype_label'] == cluster]
                current_participant_features[cluster] = {}
                
                for feature in features:
                    if feature in cluster_data.columns:
                        current_participant_features[cluster][feature] = cluster_data[feature].mean()
                    else:
                        current_participant_features[cluster][feature] = 0
            
            # Create tabs for each feature
            feature_tabs = st.tabs(features)
            
            for i, feature in enumerate(features):
                with feature_tabs[i]:
                    # Create data for this feature
                    feature_data = []
                    for cluster in ['vigorous', 'progressive', 'nonprogressive', 'immotile']:
                        if cluster in typical_profile and cluster in current_participant_features:
                            typical_value = typical_profile[cluster][feature]
                            current_value = current_participant_features[cluster][feature]
                            
                            feature_data.append({
                                'Cluster': cluster,
                                'Typical': typical_value,
                                'Current': current_value
                            })
                    
                    if feature_data:
                        # Create DataFrame for this feature
                        feature_df = pd.DataFrame(feature_data)
                        
                        # Create horizontal stacked bar chart similar to motility distribution
                        fig = go.Figure()
                        
                        # Define cluster colors
                        cluster_colors = {
                            'vigorous': '#1f77b4',      # blue
                            'progressive': '#2ca02c',   # green
                            'nonprogressive': '#ff7f0e', # orange
                            'immotile': '#d62728'       # red
                        }
                        
                        # Create data for stacked bars
                        current_data = []
                        typical_data = []
                        
                        for row in feature_data:
                            cluster = row['Cluster']
                            current_data.append({
                                'cluster': cluster,
                                'value': row['Current'],
                                'color': cluster_colors[cluster]
                            })
                            typical_data.append({
                                'cluster': cluster,
                                'value': row['Typical'],
                                'color': cluster_colors[cluster]
                            })
                        
                        # Add stacked bars for current participant
                        for item in current_data:
                            fig.add_trace(go.Bar(
                                y=['Your Patient'],
                                x=[item['value']],
                                orientation='h',
                                name=f"{item['cluster']}",
                                text=f'{item["value"]:.2f}',
                                textposition='inside',
                                textfont=dict(color='white', size=10),
                                marker_color=item['color'],
                                showlegend=True
                            ))
                        
                        # Add stacked bars for typical patient
                        for item in typical_data:
                            fig.add_trace(go.Bar(
                                y=['Typical Patient'],
                                x=[item['value']],
                                orientation='h',
                                name=f"Typical {item['cluster']}",
                                text=f'{item["value"]:.2f}',
                                textposition='inside',
                                textfont=dict(color='white', size=10),
                                marker_color=item['color'],
                                showlegend=False,
                                opacity = 0.7
                            ))
                        
                        fig.update_layout(
                            title=f"{feature} Feature Values by Cluster",
                            xaxis_title=f"{feature} Value",
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
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"feature_{feature}")
                        
                        st.info("💡 Interpretation: This patient’s cluster average (x µm/s) vs. the median value for that cluster. ")
                        
                       
                    else:
                        st.info(f"No data available for {feature}")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load feature comparison: {str(e)}")
        
        # Track Trajectory Viewer
        st.subheader("**Trajectory Viewer**")
        # Create a dropdown for track selection
        available_tracks = sorted(preds['track_id'].unique())
        selected_track = st.selectbox(
            "Select a track:",
            available_tracks,
            format_func=lambda x: f"{x} ({preds[preds['track_id']==x]['subtype_label'].iloc[0]})"
        )
        
        if selected_track:
            
            traj_df = frame_df[frame_df['track_id'] == selected_track].sort_values("frame_num")
            if not traj_df.empty:
                # Get track statistics for subtype and confidence
                track_stats = preds[preds['track_id'] == selected_track].iloc[0]
                subtype = track_stats['subtype_label'] if 'subtype_label' in track_stats else f"Cluster {track_stats['cluster_id']}"
                
                # Get confidence (probability of the predicted subtype)
                if 'subtype_label' in track_stats:
                    # Find the probability column for the predicted subtype
                    prob_column = f"P_{subtype}"
                    confidence = track_stats.get(prob_column, 0) * 100
                else:
                    confidence = 0
                
                fig_traj = px.line(
                    traj_df,
                    x='x',
                    y='y',
                    title=f'Trajectory Subtype: {subtype} | Confidence: {confidence:.1f}%',
                    markers=True,
                    width=400,
                    height=400
                )
                
                # Remove axes and axes titles
                fig_traj.update_layout(
                    xaxis=dict(showgrid=False, showticklabels=False, title=""),
                    yaxis=dict(showgrid=False, showticklabels=False, title=""),
                   
                )
                st.plotly_chart(fig_traj, use_container_width=True, key=f"trajectory_{selected_track}")
                
                # Show track statistics in collapsible dropdowns
                
                # Get specific cluster probabilities
                specific_probs = ['P_nonprogressive', 'P_vigorous', 'P_immotile', 'P_progressive']
                cluster_probs = {col: track_stats[col] for col in specific_probs if col in track_stats}
                
                # Get feature values
                features = ['ALH', 'BCF', 'LIN', 'MAD', 'STR', 'VAP', 'VCL', 'VSL', 'WOB']
                feature_values = {f: track_stats[f] for f in features if f in track_stats}
                
                # Create three columns for the collapsible sections
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.expander("📊 Feature Values", expanded=False):
                        if feature_values:
                            feature_df = pd.DataFrame(list(feature_values.items()), columns=['Feature', 'Value'])
                            feature_df['Value'] = feature_df['Value'].apply(lambda x: f"{x:.3f}")
                            st.dataframe(feature_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Feature values not available for this track.")
                
                with col2:
                    with st.expander("🎯 Cluster Probabilities", expanded=False):
                        if cluster_probs:
                            prob_df = pd.DataFrame(list(cluster_probs.items()), columns=['Cluster', 'Probability'])
                            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.3f}")
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Cluster probabilities not available for this track.")
                
                with col3:
                    with st.expander("📈 Track Summary", expanded=False):
                        st.markdown(f"""
                        **Track ID:** {selected_track}  
                        **Subtype:** {track_stats['subtype_label'] if 'subtype_label' in track_stats else f"Cluster {track_stats['cluster_id']}"}
                        """)
            else:
                st.warning("No trajectory data found.")
        

        

        
        # Download results
        st.markdown("**📥 Download Results**")
        if os.path.exists(preds_csv):
            with open(preds_csv, 'r') as f:
                st.download_button(
                    label="📊 Download Predictions CSV",
                    data=f.read(),
                    file_name=f"{participant_id}_predictions.csv",
                    mime="text/csv"
                )
    elif run_btn and (not json_file or not video_file):
        st.error("❌ Please upload both JSON and video files to run analysis.")
