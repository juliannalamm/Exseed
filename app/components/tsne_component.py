# t-SNE scatter plot component
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

def create_tsne_figure():
    """Create the t-SNE scatter plot figure"""
    # Create discrete color mapping
    unique_subtypes = POINTS["subtype_label"].unique()
    
    fig = go.Figure()
    
    for i, subtype in enumerate(unique_subtypes):
        mask = POINTS["subtype_label"] == subtype
        fig.add_trace(
            go.Scattergl(
                x=POINTS.loc[mask, "tsne_1"], 
                y=POINTS.loc[mask, "tsne_2"],
                mode="markers",
                marker=dict(size=4, opacity=0.75),
                name=str(subtype),  # Legend label
                customdata=POINTS.loc[mask, ["track_id","participant_id","subtype_label"]].values,
                hovertemplate="<b>Track:</b> %{customdata[0]}<br>" +
                             "<b>Class:</b> %{customdata[2]}<br>")
        )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="t-SNE-1", yaxis_title="t-SNE-2",
        uirevision="tsne-static",
        showlegend=True,  # Show legend for discrete colors
        # Chart background customization
        paper_bgcolor="black",  # Outer chart background
        plot_bgcolor="#000000",  # Inner plot area background
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
            html.H2("Sperm Motility Analysis", 
                    style={"color": "white", "marginBottom": "8px", "fontSize": "24px", "fontWeight": "600", "textAlign": "center"}),
            html.P("Interactive t-SNE visualization of sperm motility patterns", 
                   style={"color": "#cccccc", "marginBottom": "16px", "fontSize": "14px", "textAlign": "center"}),
        ]),
        dcc.Graph(
            id="tsne",
            figure=create_tsne_figure(),
            style={"height": "640px"},
            config={"responsive": False},
            clear_on_unhover=False,
        )
    ])

