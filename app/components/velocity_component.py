# Velocity meter component (capsule-style blocks)
import dash
from dash import html, Input, Output, callback_context
import typing
import numpy as np
import sys
import os

# Handle imports for both local development and container
try:
    from ..datastore import POINTS
except ImportError:
    # For local development
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datastore import POINTS


METRICS = [
    ("VCL", "Curvilinear velocity"),
    ("VSL", "Straight-line velocity"),
    ("VAP", "Average path velocity"),
]


def _percentile_caps(values):
    """
    Compute robust max (p95) per metric for normalization to 10 capsules.
    Returns a dict: metric -> p95 (fallback to finite max or 1.0).
    """
    caps = {}
    for metric in values:
        if metric in POINTS.columns:
            col = POINTS[metric].to_numpy()
            col = col[np.isfinite(col)]
            if col.size:
                p95 = float(np.quantile(col, 0.95))
                caps[metric] = p95 if p95 > 0 else (float(np.max(col)) if np.max(col) > 0 else 1.0)
            else:
                caps[metric] = 1.0
        else:
            caps[metric] = 1.0
    return caps


P95_CAPS = _percentile_caps([m for m, _ in METRICS])


def _capsule_row(metric_key: str, label_text: str, value: typing.Optional[float]) -> html.Div:
    """
    Render a single metric row with up to 10 filled capsules based on normalized value.
    More compact design for integrated display.
    """
    cap = P95_CAPS.get(metric_key, 1.0)
    norm = 0.0 if value is None or not np.isfinite(value) else max(0.0, min(1.0, float(value) / cap))
    filled = int(round(norm * 10))

    cells = []
    for i in range(10):
        is_on = i < filled
        cells.append(html.Div(
            style={
                "width": "8px",
                "height": "12px",
                "borderRadius": "4px",
                "marginRight": "3px",
                "background": "#636EFA" if is_on else "#2a2a2a",
                "boxShadow": "0 1px 2px rgba(0,0,0,0.25)" if is_on else "none",
                "transition": "background-color .2s ease"
            }
        ))

    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "padding": "2px 0",
            "marginBottom": "4px"
        },
        children=[
            html.Div(f"{label_text}", style={"color": "#e6eaf2", "fontSize": "11px", "minWidth": "35px"}),
            html.Div(
                style={"display": "flex", "alignItems": "center", "flex": "1"},
                children=cells
            ),
            html.Div(
                f"{0 if value is None or not np.isfinite(value) else round(float(value), 1)}",
                style={"color": "#8b93a7", "fontSize": "10px", "textAlign": "right", "minWidth": "35px"}
            )
        ]
    )


def _empty_rows():
    return [
        _capsule_row(key, label, None) for key, label in METRICS
    ]


def create_velocity_component(component_id="velocity-meters"):
    """Create a velocity component that integrates seamlessly within the trajectory card."""
    return html.Div(
        id=f"velocity-card-{component_id}",
        style={
            "padding": "12px 16px",
            "backgroundColor": "rgba(26,26,26,0.5)",  # Match card background exactly
            "borderRadius": "0",  # Let the parent card control rounding
            "marginTop": "0"  # No margin to connect with trajectory
        },
        children=[
            html.Div(
                "Velocity Metrics",
                style={"fontWeight": "600", "fontSize": "12px", "color": "#e6eaf2", "marginBottom": "8px", "textAlign": "center"}
            ),
            html.Div(id=component_id, children=_empty_rows()),
            
        ]
    )


def register_velocity_callbacks(app):
    @app.callback(
        [Output("tsne-velocity-meters", "children"),
         Output("pe-velocity-meters", "children")],
        Input("tsne", "hoverData"),
        Input("tsne", "clickData"),
        Input("pe-axis", "hoverData"),
        Input("pe-axis", "clickData"),
        prevent_initial_call=False,
    )
    def update_velocity_rows(tsne_hover, tsne_click, pe_hover, pe_click):
        ctx = callback_context
        
        # Determine which event triggered the callback
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"]
            if prop_id.startswith("tsne.clickData"):
                ev = tsne_click
            elif prop_id.startswith("tsne.hoverData"):
                ev = tsne_hover
            elif prop_id.startswith("pe-axis.clickData"):
                ev = pe_click
            elif prop_id.startswith("pe-axis.hoverData"):
                ev = pe_hover
            else:
                ev = None
        else:
            ev = None
            
        if not ev or "points" not in ev:
            return _empty_rows(), _empty_rows()

        p = ev["points"][0]
        customdata = p.get("customdata")
        if customdata is None or len(customdata) < 2:
            return _empty_rows(), _empty_rows()

        track_id = str(customdata[0])
        # POINTS uses track_id as string
        row = POINTS[POINTS["track_id"].astype(str) == track_id]
        if row.empty:
            return _empty_rows(), _empty_rows()
        r0 = row.iloc[0]

        rows = []
        for key, label in METRICS:
            val = r0[key] if key in row.columns else None
            try:
                val = float(val) if val is not None else None
            except Exception:
                val = None
            rows.append(_capsule_row(key, key if label is None else key, val))

        return rows, rows


