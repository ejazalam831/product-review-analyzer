# app.py

import os
from flask import Flask, jsonify
import pandas as pd
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from analysis_pipeline import analyze_product_reviews
import plotly.express as px

# Initialize Flask and Dash
app = Flask(__name__)
dash_app = Dash(
    __name__, 
    server=app, 
    url_base_pathname='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Load the preprocessed dataset
filtered_reviews = pd.read_csv('data/cleaned_reviews.csv.gz', compression='gzip')

def create_product_info(product_id):
    """Create product info section"""
    return dbc.Card([
        dbc.CardHeader(
            html.H4("Product Information", className="font-bold m-0"),
            style={'backgroundColor': '#f5f5f5'}  
        ),
        dbc.CardBody([
            html.Div([
                html.Span("Product ID: ", className="text-2xl font-bold text-gray-600"),
                html.Span(product_id, className="text-2xl font-bold text-gray-900")
            ])
        ])
    ], className="mb-4")
    
def create_summary_metrics_cards(metrics):
    """Create summary metrics cards layout"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Total Reviews: ", className="font-bold text-gray-600"), 
                        html.Span(f"{metrics['total_reviews']}", className="font-bold text-gray-900")  
                    ])  
                ])
            ], className="h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Average Rating: ", className="font-bold text-gray-600"),
                        html.Span(f"{metrics['average_rating']:.1f}", className="font-bold text-gray-900")
                    ])
                ])
            ], className="h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Sentiment Score: ", className="font-bold text-gray-600"),
                        html.Span(f"{metrics['sentiment_score']:.0f}%", className="font-bold text-gray-900")
                    ])
                ])
            ], className="h-100")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Helpful Votes: ", className="font-bold text-gray-600"),
                        html.Span(f"{metrics['total_helpful_votes']}", className="font-bold text-gray-900")
                    ])
                ])
            ], className="h-100")
        ], width=3)
    ], className="g-4")

def create_sentiment_chart(sentiment_data):
    """Create sentiment distribution chart"""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[d['sentiment'] for d in sentiment_data],
                y=[d['percentage'] for d in sentiment_data],
                marker_color=['#10b981', '#6b7280', '#ef4444']
            )
        ],
        layout={
            'height': 400,
            'margin': {'l': 40, 'r': 40, 't': 20, 'b': 40},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'weight': 'bold'}
        }
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_features_chart(features_data):
    """Create key features chart"""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[f['Mentions'] for f in features_data],
                y=[f['Features'] for f in features_data],
                orientation='h',
                marker_color='#10b981'
            )
        ],
        layout={
            'height': 400,
            'margin': {'l': 150, 'r': 40, 't': 20, 'b': 40},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'weight': 'bold'}
        }
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_feature_details_table(features_data):
    """Create feature details table"""
    # Sort features by frequency (mentions) in descending order
    sorted_features = sorted(features_data, key=lambda x: x['frequency'], reverse=True)
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("FEATURE", className="font-bold"),
                html.Th("MENTIONS", className="font-bold text-center"),
                html.Th("SENTIMENT", className="font-bold text-center")
            ], style={'backgroundColor': '#f3f4f6'})
        ]),
        html.Tbody([
            html.Tr([
                html.Td(feature['feature']),
                html.Td(feature['frequency'], className="text-center"),
                html.Td(
                    html.Span(
                        feature['sentiment_label'],
                        className="px-2 py-1 rounded text-sm",
                        style={
                            'backgroundColor': {
                                'Positive': '#e6f4ea',
                                'Neutral': '#f1f3f4',
                                'Negative': '#fce8e6'
                            }[feature['sentiment_label']],
                            'color': {
                                'Positive': '#137333',
                                'Neutral': '#5f6368',
                                'Negative': '#c5221f'
                            }[feature['sentiment_label']]
                        }
                    ),
                    className="text-center"
                )
            ]) for feature in sorted_features
        ])
    ], bordered=True, hover=True, className="w-100")
    
# Define the Dash layout
dash_app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Product Review Analysis Dashboard", 
                       className="text-3xl font-bold text-gray-900 mb-1 text-center"),  
                html.P("Analyze product reviews and sentiment in real-time",
                      className="text-gray-600 mb-4 text-center"),  
                html.Div([  
                    dbc.Button(
                        "Analyze Random Product",
                        id="analyze-button",
                        style={
                            'backgroundColor': 'black',
                            'color': 'white',
                            'border': 'none',
                            'marginBottom': '20px'
                        },
                        className="px-4 py-2 rounded hover:bg-gray-800"
                    )
                ], className="d-flex justify-content-center")  
            ])
        ])
    ], className="mb-4"),

    # Loading spinner for content
    dcc.Loading(
        id="loading-content",
        children=[
            # Product ID Section
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Current Product", className="font-bold m-0"),
                    style={'backgroundColor': '#f5f5f5'}
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span("Product ID: ", className="text-2xl font-bold text-gray-600"),
                        html.Span(id="product-id")
                    ], className="text-2xl font-bold text-gray-900")
                ])
            ], className="mb-4"),
            
            # Metrics Section
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Key Metrics", className="font-bold m-0"),
                    style={'backgroundColor': '#f5f5f5'}
                ),
                dbc.CardBody(
                    html.Div(id="metrics-section")
                )
            ], className="mb-4"),

            # Summary Section
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Analysis Summary", className="font-bold m-0"),
                    style={'backgroundColor': '#f5f5f5'}
                ),
                dbc.CardBody([
                    html.P(id="analysis-summary", className="m-0")
                ])
            ], className="mb-4"),

            # Charts Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H4("Sentiment Distribution", className="font-bold m-0"),
                            style={'backgroundColor': '#f5f5f5'}
                        ),
                        dbc.CardBody(id="sentiment-chart")
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H4("Key Features Analysis", className="font-bold m-0"),
                            style={'backgroundColor': '#f5f5f5'}
                        ),
                        dbc.CardBody(id="features-chart")
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Feature Details Section
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Feature Details", className="font-bold m-0"),
                    style={'backgroundColor': '#f5f5f5'}
                ),
                dbc.CardBody([
                    html.Div(id="feature-details-table")
                ])
            ], className="mb-4"),
        ]
    )
], fluid=True, style={'backgroundColor': '#f3f4f6', 'padding': '2rem'})

@dash_app.callback(
    [
        Output("product-id", "children"),
        Output("metrics-section", "children"),
        Output("sentiment-chart", "children"),
        Output("features-chart", "children"),
        Output("analysis-summary", "children"),
        Output("feature-details-table", "children")
    ],
    Input("analyze-button", "n_clicks")
)

def update_dashboard(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        # Select random product and analyze
        random_product = filtered_reviews.sample(n=1).iloc[0]
        product_id = random_product['product_id']
        results = analyze_product_reviews(filtered_reviews[filtered_reviews['product_id'] == product_id])

        # Create components
        product_info = create_product_info(product_id)
        metrics_cards = create_summary_metrics_cards(results['metrics'])
        sentiment_chart = create_sentiment_chart(
            results['metrics']['sentiment_distribution'].to_dict('records')
        )
        features_chart = create_features_chart(
            results['feature_analysis']['top_positive'][:5]
        )
        feature_table = create_feature_details_table(results['features'])
        
        return (
            product_id,
            metrics_cards, 
            sentiment_chart, 
            features_chart, 
            results['summary'],
            feature_table
        )
    except Exception as e:
        print(f"Error updating dashboard: {str(e)}")
        return "Error", "Error", "Error", "Error", "Error", "Error"

# Add custom CSS
dash_app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .card {
                border-radius: 8px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1),
                          0 1px 2px 0 rgba(0, 0, 0, 0.06);
                background-color: white;
            }
            .card-header {
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            .card-body {
                padding: 1.5rem;
            }
            .font-bold {
                font-weight: bold;
            }
            /* New styles for the table */
            .table {
                width: 100%;
                margin-bottom: 1rem;
                background-color: transparent;
                border-collapse: collapse;
            }
            .table th,
            .table td {
                padding: 1rem;
                vertical-align: middle;
                border-top: 1px solid #dee2e6;
            }
            .table thead th {
                vertical-align: bottom;
                border-bottom: 2px solid #dee2e6;
                background-color: #f8f9fa;
                font-weight: 600;
            }
            .table tbody tr:hover {
                background-color: rgba(0, 0, 0, 0.075);
            }
            /* Responsive table */
            @media screen and (max-width: 768px) {
                .table-responsive {
                    display: block;
                    width: 100%;
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }
            }
            /* Utility classes for text alignment */
            .text-center {
                text-align: center;
            }
            .text-right {
                text-align: right;
            }
            /* Badge-like styles for sentiment labels */
            .sentiment-badge {
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 500;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
server = app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)