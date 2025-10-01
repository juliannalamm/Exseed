# P/E axis scatter plot component
import dash
from dash import dcc, html, Input, Output, callback_context
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

# Color mapping for each motility type (matches t-SNE plot colors)
SUBTYPE_COLORS = {
    "progressive": "#636EFA",      # blue (Plotly default color 0)
    "rapid_progressive": "#EF553B", # red (Plotly default color 1)
    "non_progressive": "#00CC96",   # teal/green (Plotly default color 2)
    "erratic": "#AB63FA",           # purple (Plotly default color 3)
    "immotile": "#FFA15A"           # orange (Plotly default color 4)
}

def create_pe_axis_figure():
    """Create the P/E axis scatter plot figure"""
    # Create discrete color mapping
    unique_subtypes = POINTS["subtype_label"].unique()
    
    fig = go.Figure()
    
    for i, subtype in enumerate(unique_subtypes):
        mask = POINTS["subtype_label"] == subtype
        color = SUBTYPE_COLORS.get(subtype, "#636EFA")
        
        # Regular points (non-hyperactive)
        regular_mask = mask & (POINTS["is_hyperactive_mouse"] == 0)
        if regular_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=POINTS.loc[regular_mask, "P_axis_byls"], 
                    y=POINTS.loc[regular_mask, "E_axis_byls"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.75, color=color),
                    name=str(subtype).replace('_', ' ').title(),  # Legend label
                    customdata=POINTS.loc[regular_mask, ["track_id","participant_id","subtype_label","entropy","is_hyperactive_mouse"]].values,
                    hovertemplate="<b>Class:</b> %{customdata[2]}<br><b>Cluster Uncertainty:</b> %{customdata[3]:.3f}<br><extra></extra>")
            )
        
        # Hyperactive points with glow effect
        hyperactive_mask = mask & (POINTS["is_hyperactive_mouse"] == 1)
        if hyperactive_mask.any():
            # Add glow layer (smaller, more radiant)
            fig.add_trace(
                go.Scattergl(
                    x=POINTS.loc[hyperactive_mask, "P_axis_byls"], 
                    y=POINTS.loc[hyperactive_mask, "E_axis_byls"],
                    mode="markers",
                    marker=dict(
                        size=6,  # Smaller size
                        opacity=0.4,  # More opaque for radiance
                        color="#fff8dc",  # Softer, more radiant golden color
                        line=dict(width=0)
                    ),
                    name="",  # No legend entry for glow
                    showlegend=False,
                    hoverinfo="skip"  # Skip hover for glow layer
                )
            )
            # Add main hyperactive points (keep original cluster colors)
            fig.add_trace(
                go.Scattergl(
                    x=POINTS.loc[hyperactive_mask, "P_axis_byls"], 
                    y=POINTS.loc[hyperactive_mask, "E_axis_byls"],
                    mode="markers",
                    marker=dict(
                        size=4, 
                        opacity=0.9,
                        color=color,  # Keep original cluster color
                        line=dict(width=1, color="#ffd700")  # Thinner golden border for hyperactive
                    ),
                    name="",  # No legend entry for individual hyperactive traces
                    showlegend=False,
                    customdata=POINTS.loc[hyperactive_mask, ["track_id","participant_id","subtype_label","entropy","is_hyperactive_mouse"]].values,
                    hovertemplate="<b>Class:</b> %{customdata[2]}<br><b>Cluster Uncertainty:</b> %{customdata[3]:.3f}<br><b>Hyperactive:</b> Yes<br><extra></extra>")
            )
    
    # Add a single "Hyperactive" legend entry if there are any hyperactive points
    if (POINTS["is_hyperactive_mouse"] == 1).any():
        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],  # Invisible trace
                mode="markers",
                marker=dict(
                    size=4,
                    color="#636EFA",  # Use a default color for the legend
                    line=dict(width=1, color="#ffd700")  # Thinner golden border
                ),
                name="Hyperactive",
                showlegend=True,
                hoverinfo="skip"
            )
        )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Progressivity (P-axis)", 
        yaxis_title="Erraticity (E-axis)",
        uirevision="pe-axis-static",
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

def create_pe_axis_component():
    """Create the P/E axis component with graph"""
    return html.Div([
        html.Div([
            html.Div("Sperm Motility Plotted by Progressive vs Erratic Movement Type", 
                    style={"color": "white", "marginBottom": "4px", "fontSize": "20px", "fontWeight": "600", "textAlign": "center"}),
            html.Div("This chart maps sperm along a continuous progressivity–erraticity spectrum, highlighting subtle shifts into hyperactivated movement—the vigorous swimming style needed to penetrate the egg.", 
                   style={"color": "#cccccc", "marginBottom": "8px", "fontSize": "14px", "textAlign": "center", "maxWidth": "600px", "margin": "0 auto 8px auto"}),
        ]),
        html.Div(
            dcc.Graph(
                id="pe-axis",
                figure=create_pe_axis_figure(),
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
