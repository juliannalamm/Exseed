# t-SNE scatter plot component colored by drug ID
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

def create_tsne_drug_figure():
    """Create the t-SNE scatter plot figure colored by drug ID"""
    if "experiment_media" not in POINTS.columns:
        # Fallback to regular t-SNE if drug data not available
        from .tsne_component import create_tsne_figure
        return create_tsne_figure()
    
    # Get unique drug IDs
    unique_drugs = sorted(POINTS["experiment_media"].unique())
    
    # Define colors for drugs (using Plotly's default color sequence)
    drug_colors = [
        "#636EFA",  # blue
        "#EF553B",  # red  
        "#00CC96",  # teal/green
        "#AB63FA",  # purple
        "#FFA15A",  # orange
        "#19D3F3",  # cyan
        "#FF6692",  # pink
        "#B6E880",  # light green
        "#FF97FF",  # magenta
        "#FECB52"   # yellow
    ]
    
    fig = go.Figure()
    
    for i, drug_id in enumerate(unique_drugs):
        mask = POINTS["experiment_media"] == drug_id
        color = drug_colors[i % len(drug_colors)]
        
        # Regular points (non-hyperactive)
        regular_mask = mask & (POINTS["is_hyperactive_mouse"] == 0)
        if regular_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=POINTS.loc[regular_mask, "tsne_1"], 
                    y=POINTS.loc[regular_mask, "tsne_2"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.75, color=color),
                    name=f"Drug {drug_id}",
                    customdata=(POINTS.loc[regular_mask, ["track_id","participant_id","subtype_label","is_hyperactive_mouse","experiment_media"]].values 
                               if "experiment_media" in POINTS.columns 
                               else POINTS.loc[regular_mask, ["track_id","participant_id","subtype_label","is_hyperactive_mouse"]].values),
                    hovertemplate=("<b>Drug:</b> %{customdata[4]}<br>" + 
                                 "<b>Class:</b> %{customdata[2]}<br>" + 
                                 "<extra></extra>")
                )
            )
        
        # Hyperactive points
        hyperactive_mask = mask & (POINTS["is_hyperactive_mouse"] == 1)
        if hyperactive_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=POINTS.loc[hyperactive_mask, "tsne_1"], 
                    y=POINTS.loc[hyperactive_mask, "tsne_2"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.75, color=color),
                    name="",  # No separate legend entry
                    showlegend=False,
                    customdata=(POINTS.loc[hyperactive_mask, ["track_id","participant_id","subtype_label","is_hyperactive_mouse","experiment_media"]].values 
                               if "experiment_media" in POINTS.columns 
                               else POINTS.loc[hyperactive_mask, ["track_id","participant_id","subtype_label","is_hyperactive_mouse"]].values),
                    hovertemplate=("<b>Drug:</b> %{customdata[4]}<br>" + 
                                 "<b>Class:</b> %{customdata[2]}<br>" + 
                                 "<b>Hyperactive:</b> Yes<br>" + 
                                 "<extra></extra>")
                )
            )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="t-SNE-1", yaxis_title="t-SNE-2",
        uirevision="tsne-drug-static",
        showlegend=True,
        # Chart background customization
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
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

def create_tsne_drug_component():
    """Create the drug-colored t-SNE component with graph"""
    return html.Div([
        html.Div([
            html.Div("Sperm Motility Colored by Drug Treatment", 
                    style={"color": "white", "marginBottom": "4px", "fontSize": "20px", "fontWeight": "600", "textAlign": "center"}),
            html.Div("Each point is an individual cell colored by drug ID, hover over a point to view it's trajectory!", 
                   style={"color": "#cccccc", "marginBottom": "16px", "fontSize": "14px", "textAlign": "center"}),
        ]),
        html.Div(
            dcc.Graph(
                id="tsne-drug",
                figure=create_tsne_drug_figure(),
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
