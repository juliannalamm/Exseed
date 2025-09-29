from dash import html
import dash_bootstrap_components as dbc


def create_header_component():
    return dbc.Navbar(
        html.Div(
            [
                dbc.NavbarBrand(
                    html.Div(
                        [
                            html.Span(
                                "🧬",
                                style={"fontSize": "24px", "marginRight": "12px"}
                            ),
                            html.Span(
                                "SP",
                                className="badge bg-gradient rounded-pill me-2",
                                style={"fontSize": "12px", "padding": "6px 12px", "background": "linear-gradient(45deg, #667eea 0%, #764ba2 100%)"}
                            ),
                            html.Span(
                                "Sperm Phenotype Explorer",
                                style={"fontSize": "20px", "fontWeight": "600", "color": "white"}
                            ),
                        ],
                        className="d-flex align-items-center",
                    ),
                    class_name="fw-semibold",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.Button(
                                "📚 Docs", 
                                color="outline-light", 
                                outline=True, 
                                size="sm", 
                                class_name="me-2",
                                style={"borderRadius": "20px", "padding": "6px 16px"}
                            ),
                            dbc.Button(
                                "💬 Feedback", 
                                color="outline-light", 
                                outline=True, 
                                size="sm",
                                style={"borderRadius": "20px", "padding": "6px 16px"}
                            ),
                        ],
                        class_name="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "width": "100%",
                "padding": "0 20px",
            }
        ),
        color="dark",
        dark=True,
        class_name="shadow-lg",
        style={
            "background": "linear-gradient(135deg, #7aa2f7 0%, #5a7fd4 100%)",
            "borderBottom": "2px solid rgba(255, 255, 255, 0.1)",
            "backdropFilter": "blur(10px)",
            "padding": "0 !important",
            "margin": "0 !important",
            "width": "100% !important",
            "maxWidth": "none !important",
            "left": "0",
            "right": "0"
        }
    )


