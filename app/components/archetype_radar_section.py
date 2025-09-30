"""
Archetype radar section component that displays beautiful radar charts
for patient archetypes below the continuous motility scores.
"""

from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Handle imports for both local development and container
try:
    from components.radar_component import create_dual_radar, create_archetype_radar_grid
    from archetype_data_loader import ArchetypeDataLoader
except ImportError:
    from app.components.radar_component import create_dual_radar, create_archetype_radar_grid
    from app.archetype_data_loader import ArchetypeDataLoader


def create_archetype_radar_section():
    """
    Create the archetype radar section with patient fingerprints.
    
    Returns
    -------
    html.Div
        Dash HTML component with radar charts
    """
    
    # Try to load the data
    try:
        # Try multiple possible paths for dash_data
        data_paths = ["dash_data", "app/dash_data", "./app/dash_data"]
        loader = None
        for data_path in data_paths:
            try:
                print(f"Trying to load archetype data from: {data_path}")
                loader = ArchetypeDataLoader(data_path)
                print(f"Successfully loaded archetype data from: {data_path}")
                break
            except Exception as path_error:
                print(f"Failed to load from {data_path}: {path_error}")
                continue
        
        if loader is None:
            raise Exception("Could not find dash_data in any expected location")
            
        archetypes = loader.get_archetype_list()
        has_data = len(archetypes) > 0
    except Exception as e:
        print(f"Could not load archetype data: {e}")
        has_data = False
        loader = None
        archetypes = []
    
    if not has_data:
        return html.Div(
            style={
                "backgroundColor": "rgba(26, 26, 26, 0.5)",
                "borderRadius": "16px",
                "padding": "40px",
                "textAlign": "center",
                "border": "1px solid rgba(255, 255, 255, 0.1)",
            },
            children=[
                html.H3(
                    "Patient Archetype Fingerprints",
                    style={"color": "white", "marginBottom": "16px"}
                ),
                html.P(
                    "Archetype data not available. Please export data from the notebook first.",
                    style={"color": "#e0e0e0", "fontSize": "14px"}
                ),
                html.P(
                    "Run the export cell in your notebook to generate the radar chart data.",
                    style={"color": "#999", "fontSize": "12px", "marginTop": "8px"}
                )
            ]
        )
    
    # Create archetype dropdown options
    archetype_options = []
    for arch_name in archetypes:
        info = loader.get_archetype_info(arch_name)
        label = f"{arch_name}: {info.get('title', 'Unknown')}"
        archetype_options.append({'label': label, 'value': arch_name})
    
    return html.Div(
        style={
            "backgroundColor": "rgba(26, 26, 26, 0.5)",
            "borderRadius": "16px",
            "padding": "30px",
            "border": "1px solid rgba(255, 255, 255, 0.1)",
            "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.3)",
        },
        children=[
            # Section header
            html.H2(
                "Patient Archetype Fingerprints",
                style={
                    "color": "white",
                    "textAlign": "center",
                    "marginBottom": "12px",
                    "fontSize": "26px",
                    "fontWeight": "700",
                    "background": "linear-gradient(135deg, #636EFA 0%, #EF553B 100%)",
                    "WebkitBackgroundClip": "text",
                    "WebkitTextFillColor": "transparent",
                    "backgroundClip": "text",
                }
            ),
            
            # Description
            html.P(
                "Each patient exhibits a unique 'fingerprint' of motility patterns. "
                "The radar charts below visualize both the GMM-derived cluster composition "
                "and traditional CASA metrics for representative archetypes.",
                style={
                    "color": "#e0e0e0",
                    "textAlign": "center",
                    "maxWidth": "800px",
                    "margin": "0 auto 30px auto",
                    "fontSize": "15px",
                    "lineHeight": "1.6",
                }
            ),
            
            # Controls row
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "gap": "20px",
                    "marginBottom": "30px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                        children=[
                            html.Label(
                                "View:",
                                style={
                                    "color": "white",
                                    "fontWeight": "600",
                                    "fontSize": "14px",
                                }
                            ),
                            dcc.RadioItems(
                                id='radar-view-mode',
                                options=[
                                    {'label': ' All Archetypes', 'value': 'grid'},
                                    {'label': ' Single Detail', 'value': 'single'},
                                ],
                                value='grid',
                                inline=True,
                                style={"color": "white"},
                                labelStyle={
                                    "marginRight": "15px",
                                    "color": "#e0e0e0",
                                    "fontSize": "14px",
                                },
                                inputStyle={"marginRight": "5px"}
                            ),
                        ]
                    ),
                    
                    html.Div(
                        id='archetype-selector-container',
                        style={"display": "none"},  # Hidden by default
                        children=[
                            html.Label(
                                "Select Archetype:",
                                style={
                                    "color": "white",
                                    "fontWeight": "600",
                                    "marginRight": "10px",
                                    "fontSize": "14px",
                                }
                            ),
                            dcc.Dropdown(
                                id='single-archetype-selector',
                                options=archetype_options,
                                value=archetypes[0] if archetypes else None,
                                style={
                                    "width": "300px",
                                    "backgroundColor": "#2a2a2a",
                                    "color": "white",
                                },
                                clearable=False,
                            ),
                        ]
                    ),
                ]
            ),
            
            # Radar chart display
            html.Div(
                id='radar-charts-container',
                style={
                    "borderRadius": "12px",
                    "overflow": "hidden",
                },
                children=[
                    dcc.Graph(
                        id='archetype-radars',
                        style={"width": "100%"},
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d'],
                        }
                    )
                ]
            ),
            
            # Legend/interpretation guide
            html.Div(
                style={
                    "marginTop": "25px",
                    "padding": "20px",
                    "backgroundColor": "rgba(99, 110, 250, 0.1)",
                    "borderRadius": "12px",
                    "border": "1px solid rgba(99, 110, 250, 0.3)",
                },
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                            "gap": "15px",
                        },
                        children=[
                            html.Div([
                                html.Span(
                                    "🔵 GMM Composition",
                                    style={
                                        "color": "#636EFA",
                                        "fontWeight": "600",
                                        "fontSize": "13px",
                                    }
                                ),
                                html.P(
                                    "Probabilistic cluster membership showing motility pattern distribution",
                                    style={
                                        "color": "#e0e0e0",
                                        "fontSize": "12px",
                                        "margin": "5px 0 0 0",
                                        "lineHeight": "1.4",
                                    }
                                ),
                            ]),
                            html.Div([
                                html.Span(
                                    "🔴 CASA Metrics",
                                    style={
                                        "color": "#EF553B",
                                        "fontWeight": "600",
                                        "fontSize": "13px",
                                    }
                                ),
                                html.P(
                                    "Traditional kinematic parameters (ALH, VCL, LIN, etc.) normalized to population",
                                    style={
                                        "color": "#e0e0e0",
                                        "fontSize": "12px",
                                        "margin": "5px 0 0 0",
                                        "lineHeight": "1.4",
                                    }
                                ),
                            ]),
                        ]
                    ),
                ]
            ),
        ]
    )


def register_archetype_radar_callbacks(app):
    """
    Register callbacks for the archetype radar section.
    
    Parameters
    ----------
    app : dash.Dash
        Dash app instance
    """
    
    # Try to load data
    try:
        # Try multiple possible paths for dash_data
        data_paths = ["dash_data", "app/dash_data", "./app/dash_data"]
        loader = None
        for data_path in data_paths:
            try:
                print(f"Trying to load archetype data from: {data_path}")
                loader = ArchetypeDataLoader(data_path)
                print(f"Successfully loaded archetype data from: {data_path}")
                break
            except Exception as path_error:
                print(f"Failed to load from {data_path}: {path_error}")
                continue
        
        if loader is None:
            raise Exception("Could not find dash_data in any expected location")
            
        has_data = True
    except Exception as e:
        print(f"Could not load archetype data for callbacks: {e}")
        has_data = False
        loader = None
    
    if not has_data:
        # Return dummy callback if no data
        @app.callback(
            Output('archetype-radars', 'figure'),
            Input('radar-view-mode', 'value'),
        )
        def dummy_callback(view_mode):
            return go.Figure().update_layout(
                title="No data available",
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )
        return
    
    # Callback to show/hide archetype selector
    @app.callback(
        Output('archetype-selector-container', 'style'),
        Input('radar-view-mode', 'value'),
    )
    def toggle_archetype_selector(view_mode):
        if view_mode == 'single':
            return {"display": "flex", "alignItems": "center", "gap": "10px"}
        return {"display": "none"}
    
    # Callback to update radar charts
    @app.callback(
        Output('archetype-radars', 'figure'),
        Input('radar-view-mode', 'value'),
        Input('single-archetype-selector', 'value'),
    )
    def update_radar_charts(view_mode, selected_archetype):
        """Update radar charts based on view mode and selection."""
        
        if view_mode == 'grid':
            # Show all archetypes in a grid
            archetypes = loader.get_archetype_list()
            archetypes_data = []
            
            for arch_name in archetypes:
                tracks, frames, patient, info = loader.get_archetype_data(arch_name)
                
                # Get GMM composition
                post_cols = ['P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 
                            'P_immotile', 'P_erratic']
                available_posts = [c for c in post_cols if c in tracks.columns]
                
                if available_posts:
                    gmm_values = tracks[available_posts].mean(axis=0).values.tolist()
                    gmm_labels = [c.replace('P_', '').replace('_', ' ').title() 
                                 for c in available_posts]
                else:
                    gmm_values = []
                    gmm_labels = []
                
                # Get CASA metrics (we'll just use the labels for now)
                casa_cols = [c for c in patient.index if c.startswith('CASA_') and c.endswith('_mean')]
                casa_labels = [c.replace('CASA_', '').replace('_mean', '') for c in casa_cols[:5]]
                casa_values = [0.5] * len(casa_labels)  # Placeholder
                
                archetypes_data.append({
                    'title': info.get('title', f'Archetype {arch_name}'),
                    'participant_id': info['participant_id'],
                    'gmm_values': gmm_values,
                    'gmm_labels': gmm_labels,
                    'casa_values': casa_values,
                    'casa_labels': casa_labels,
                })
            
            return create_archetype_radar_grid(archetypes_data, cols=2)
        
        else:
            # Show single archetype in detail
            if not selected_archetype:
                return go.Figure()
            
            tracks, frames, patient, info = loader.get_archetype_data(selected_archetype)
            
            # Get GMM composition
            post_cols = ['P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 
                        'P_immotile', 'P_erratic']
            available_posts = [c for c in post_cols if c in tracks.columns]
            
            gmm_values = tracks[available_posts].mean(axis=0).values.tolist()
            gmm_labels = [c.replace('P_', '').replace('_', ' ').title() 
                         for c in available_posts]
            
            # Get CASA metrics
            casa_cols_full = sorted([c for c in patient.index if c.startswith('CASA_') and c.endswith('_mean')])
            casa_values_raw = [float(patient[c]) if c in patient.index else np.nan 
                              for c in casa_cols_full]
            casa_labels = [c.replace('CASA_', '').replace('_mean', '') for c in casa_cols_full]
            
            # Normalize CASA values
            casa_scales = loader.get_casa_scales()
            casa_values = []
            for label, val in zip(casa_labels, casa_values_raw):
                if label in casa_scales and np.isfinite(val):
                    lo, hi = casa_scales[label]
                    normalized = (val - lo) / (hi - lo + 1e-12)
                    normalized = np.clip(normalized, 0, 1)
                else:
                    normalized = 0
                casa_values.append(normalized)
            
            return create_dual_radar(
                gmm_values=gmm_values,
                gmm_labels=gmm_labels,
                casa_values=casa_values,
                casa_labels=casa_labels,
                archetype_title=info.get('title', 'Patient Archetype'),
                participant_id=info['participant_id']
            )
