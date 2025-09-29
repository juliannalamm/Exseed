from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import sys
import os

# Handle imports for both local development and container
try:
    from ..datastore import POINTS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datastore import POINTS

KINEMATIC_FEATURES = [
    "ALH", "BCF", "LIN", "MAD", "STR", "VAP", "VCL", "VSL", "WOB"
]


def _cluster_means(subtype: str) -> pd.DataFrame:
    df = POINTS
    if df.empty or subtype is None or "subtype_label" not in df.columns:
        return pd.DataFrame({"Feature": KINEMATIC_FEATURES, "Mean": [0] * len(KINEMATIC_FEATURES)})
    mask = df["subtype_label"] == subtype
    if not mask.any():
        return pd.DataFrame({"Feature": KINEMATIC_FEATURES, "Mean": [0] * len(KINEMATIC_FEATURES)})
    cols = [c for c in KINEMATIC_FEATURES if c in df.columns]
    means = df.loc[mask, cols].mean(numeric_only=True)
    return pd.DataFrame({"Feature": means.index, "Mean": means.values})


def _minmax(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mn = float(df["Mean"].min())
    mx = float(df["Mean"].max())
    rng = (mx - mn) or 1.0
    out = df.copy()
    out["Norm"] = (out["Mean"] - mn) / rng
    return out


def create_cluster_metrics_component():
    options = [
        ("erratic", "Erratic"),
        ("rapid_progressive", "Rapid Progressive"),
        ("non-progressive", "Non-progressive"),
        ("immotile", "Immotile"),
    ]
    return html.Div(
        children=[
            dcc.Tabs(
                id="metrics-tabs",
                value=options[0][0],
                children=[
                    dcc.Tab(label=label, value=val) for val, label in options
                ],
                style={
                    "color": "black",
                },
            ),
            dcc.Graph(id="cluster-metrics", style={"height": "300px"}),
        ]
    )


def register_cluster_metrics_callbacks(app):
    @app.callback(
        Output("cluster-metrics", "figure"),
        Input("metrics-tabs", "value"),
        prevent_initial_call=False,
    )
    def update_metrics(active_subtype):
        stats = _cluster_means(active_subtype)
        stats = _minmax(stats)
        fig = px.bar(
            stats,
            x="Feature",
            y="Norm",
            title=f"{active_subtype}: normalized (0-1) kinematic means",
            color_discrete_sequence=["#7aa2f7"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, title="0-1 scale"),
        )
        return fig


