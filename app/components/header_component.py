from dash import html
import dash_bootstrap_components as dbc


def create_header_component():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    html.Div(
                        [
                            html.Span(
                                "SP",
                                className="badge bg-primary rounded-pill me-2",
                            ),
                            html.Span("Sperm Phenotype Explorer"),
                        ],
                        className="d-flex align-items-center",
                    ),
                    class_name="fw-semibold",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.Button("Docs", color="secondary", outline=True, size="sm", class_name="me-2"),
                            dbc.Button("Feedback", color="secondary", outline=True, size="sm"),
                        ],
                        class_name="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]
        ),
        color="dark",
        dark=True,
        class_name="shadow-sm",
    )


