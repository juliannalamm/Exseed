# t-SNE scatter plot component
from dash import dcc, html
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

# Color mapping for each motility type (matches cluster metrics colors)
SUBTYPE_COLORS = {
    "progressive": "#636EFA",      # blue (Plotly default color 0)
    "rapid_progressive": "#EF553B", # red (Plotly default color 1)
    "non_progressive": "#00CC96",   # teal/green (Plotly default color 2)
    "erratic": "#AB63FA",           # purple (Plotly default color 3)
    "immotile": "#FFA15A"           # orange (Plotly default color 4)
}

def create_tsne_figure():
    """Create the t-SNE scatter plot figure"""
    # Create discrete color mapping
    unique_subtypes = POINTS["subtype_label"].unique()
    
    fig = go.Figure()
    
    # First, add all regular points grouped by cluster
    for i, subtype in enumerate(unique_subtypes):
        mask = POINTS["subtype_label"] == subtype
        color = SUBTYPE_COLORS.get(subtype, "#636EFA")  # Default to blue if not found
        
        # Regular points (non-hyperactive)
        regular_mask = mask & (POINTS["is_hyperactive_mouse"] == 0)
        if regular_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=POINTS.loc[regular_mask, "tsne_1"], 
                    y=POINTS.loc[regular_mask, "tsne_2"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.75, color=color),
                    name=str(subtype).replace('_', ' ').title(),  # Legend label
                    customdata=POINTS.loc[regular_mask, ["track_id","participant_id","subtype_label","is_hyperactive_mouse"]].values,
                    hovertemplate="<b>Class:</b> %{customdata[2]}<br><extra></extra>")
            )
    
    # Add hyperactive points as regular points (no special glow) - they'll be shown in P/E plot
    hyperactive_mask = POINTS["is_hyperactive_mouse"] == 1
    if hyperactive_mask.any():
        for i, subtype in enumerate(unique_subtypes):
            subtype_hyperactive_mask = hyperactive_mask & (POINTS["subtype_label"] == subtype)
            if subtype_hyperactive_mask.any():
                color = SUBTYPE_COLORS.get(subtype, "#636EFA")
                fig.add_trace(
                    go.Scattergl(
                        x=POINTS.loc[subtype_hyperactive_mask, "tsne_1"], 
                        y=POINTS.loc[subtype_hyperactive_mask, "tsne_2"],
                        mode="markers",
                        marker=dict(size=4, opacity=0.75, color=color),
                        name="",  # No separate legend entry - they're just regular cluster points
                        showlegend=False,
                        customdata=POINTS.loc[subtype_hyperactive_mask, ["track_id","participant_id","subtype_label","is_hyperactive_mouse"]].values,
                        hovertemplate="<b>Class:</b> %{customdata[2]}<br><b>Hyperactive:</b> Yes<br><extra></extra>")
                )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="t-SNE-1", yaxis_title="t-SNE-2",
        uirevision="tsne-static",
        showlegend=True,  # Show legend for discrete colors
        # Chart background customization
        paper_bgcolor="#1a1a1a",  # Outer chart background
        plot_bgcolor="#1a1a1a",  # Inner plot area background
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="white",
            title_font_color="white",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            color="white",
            title_font_color="white",
        ),
    )
    return fig

def create_tsne_component():
    """Create the t-SNE component with graph"""
    return html.Div([
        html.Div([
            html.Div("Sperm Motility Clustered by Movement Type", 
                    style={"color": "white", "marginBottom": "4px", "fontSize": "20px", "fontWeight": "600", "textAlign": "center"}),
            html.Div("Each point is an individual cell, hover over a point to view it's trajectory!", 
                   style={"color": "#cccccc", "marginBottom": "16px", "fontSize": "14px", "textAlign": "center"}),
        ]),
        html.Div(
            dcc.Graph(
                id="tsne",
                figure=create_tsne_figure(),
                style={"height": "500px"},
                config={"responsive": False},
                clear_on_unhover=False,
            ),
            style={
                "borderRadius": "12px",
                "overflow": "hidden",
                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
            }
        )
    ])

