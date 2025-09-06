# UMAP scatter plot component
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import sys
import os

# Handle imports for both local development and container
try:
    from ..datastore import POINTS
except ImportError:
    # For local development
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datastore import POINTS

def create_umap_figure():
    """Create the UMAP scatter plot figure"""
    # Create discrete color mapping
    unique_subtypes = POINTS["subtype_label"].unique()
    
    fig = go.Figure()
    
    for i, subtype in enumerate(unique_subtypes):
        mask = POINTS["subtype_label"] == subtype
        fig.add_trace(
            go.Scattergl(
                x=POINTS.loc[mask, "umap_1"], 
                y=POINTS.loc[mask, "umap_2"],
                mode="markers",
                marker=dict(size=4, opacity=0.75),
                name=str(subtype),  # Legend label
                customdata=POINTS.loc[mask, ["track_id","participant_id","subtype_label",
                                             "P_rapid_progressive","P_immotile","P_nonprogressive","P_progressive","P_cluster_4"]].values,
                hovertemplate="<b>Track:</b> %{customdata[0]}<br>" +
                             "<b>Class:</b> %{customdata[2]}<br>" +
                             "<b>Probabilities:</b><br>" +
                             "• Rapid Progressive: %{customdata[3]:.3f}<br>" +
                             "• Immotile: %{customdata[4]:.3f}<br>" +
                             "• Nonprogressive: %{customdata[5]:.3f}<br>" +
                             "• Progressive: %{customdata[6]:.3f}<br>" +
                             "• Star-spin: %{customdata[7]:.3f}<extra></extra>",
            )
        )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        uirevision="umap-static",
        showlegend=True,  # Show legend for discrete colors
        # Chart background customization
        paper_bgcolor="black",  # Outer chart background
        plot_bgcolor="#000000",  # Inner plot area background
        xaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
            title_font_color="white",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
            title_font_color="white",
        ),
    )
    return fig

def create_umap_component():
    """Create the UMAP component with graph"""
    return dcc.Graph(
        id="umap",
        figure=create_umap_figure(),
        style={"height": "640px"},
        config={"responsive": False},
        clear_on_unhover=False,
    )