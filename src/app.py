#multiple dropdown selections
import os
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import unidecode

# Load environment variables
load_dotenv()
MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')
SOCRATA_APP_TOKEN = os.getenv('SOCRATA_APP_TOKEN')

if not MAPBOX_ACCESS_TOKEN:
    raise EnvironmentError("The Mapbox Access Token has not been set in the environment variables.")
px.set_mapbox_access_token(MAPBOX_ACCESS_TOKEN)

if not SOCRATA_APP_TOKEN:
    raise EnvironmentError("The Socrata App Token has not been set in the environment variables.")

# Function Definitions
def fetch_data(data_url, data_set, app_token):
    client = Socrata(data_url, app_token)
    client.timeout = 90
    results = client.get(data_set, limit=1500000)
    return pd.DataFrame.from_records(results)

def drop_columns(df, columns):
    columns_to_drop = [col for col in columns if col in df.columns]
    return df.drop(columns=columns_to_drop)

def normalize_columns(df, column_mappings):
    for col, new_col in column_mappings.items():
        if col in df.columns:
            df[new_col] = df[col].apply(lambda x: unidecode.unidecode(x).upper() if pd.notnull(x) else x)
    return df

def merge_lat_long(indicadores_df, municipios_df):
    municipios_mapping = municipios_df.set_index('MUNICIPIO')[['LATITUDE', 'LONGITUDE']]
    merged_df = indicadores_df.merge(municipios_mapping, how='left', left_on='MUNICIPIO', right_index=True)
    merged_df.rename(columns={'LATITUDE': 'LATITUD', 'LONGITUDE': 'LONGITUD'}, inplace=True)
    return merged_df

def pivot_data(df):
    df['VALOR INDICADOR'] = pd.to_numeric(df['VALOR INDICADOR'], errors='coerce')
    pivot_df = df.pivot_table(index=['DEPARTAMENTO', 'MUNICIPIO', 'LATITUD', 'LONGITUD', 'ANO'],
                              columns='INDICADORES', values='VALOR INDICADOR').reset_index()
    pivot_df.columns.name = None
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    return pivot_df

# Fetch the data from Socrata API
data_url = "www.datos.gov.co"
data_set = "4e4i-ua65"
indicadores_df = fetch_data(data_url, data_set, SOCRATA_APP_TOKEN)

# Load the Municipios data from the URL
municipios_url = 'https://github.com/rmejia41/open_datasets/raw/main/Municipios.xlsx'
municipios_df = pd.read_excel(municipios_url)

# Drop the coddepartamento and codmunicipio columns if they exist
columns_to_drop = ['coddepartamento', 'codmunicipio']
indicadores_df = drop_columns(indicadores_df, columns_to_drop)

# Normalize the specified columns
columns_to_normalize = {
    'departamento': 'DEPARTAMENTO',
    'municipio': 'MUNICIPIO',
    'indicador': 'INDICADORES',
    'a_o': 'ANO',
    'valor_indicador': 'VALOR INDICADOR'
}
indicadores_df = normalize_columns(indicadores_df, columns_to_normalize)

# Merge LATITUD and LONGITUD
merged_df = merge_lat_long(indicadores_df, municipios_df)

# Pivot the DataFrame
pivot_df = pivot_data(merged_df)

# Drop duplicate rows
pivot_df = pivot_df.drop_duplicates(subset=['DEPARTAMENTO', 'MUNICIPIO', 'LATITUD', 'LONGITUD', 'ANO'])

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard: Indicadores mortalidad y morbilidad según departamento, municipio y año",
                        className="text-center mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': str(year)} for year in sorted(pivot_df['ANO'].unique())],
                value=[],
                multi=True,
                clearable=False
            )
        ], width=3),
        dbc.Col([
            dcc.Dropdown(
                id='municipio-dropdown',
                options=[{'label': municipio, 'value': municipio} for municipio in
                         sorted(pivot_df['MUNICIPIO'].unique())],
                value=[],
                multi=True,
                clearable=True
            )
        ], width=3),
        dbc.Col([
            dcc.Dropdown(
                id='indicador-dropdown',
                options=[{'label': indicador, 'value': indicador} for indicador in sorted(pivot_df.columns[5:])],
                value=[],
                multi=True,
                clearable=True,
                style={'width': '100%'}
            )
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col(html.A("Open Data Source",
                       href="https://www.datos.gov.co/Salud-y-Protecci-n-Social/Indicadores-mortalidad-y-morbilidad-seg-n-departam/4e4i-ua65/about_data",
                       target="_blank", className="btn btn-primary mt-4"), width=12, className="text-center")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='map-graph', style={'height': '800px'}), width=12)  # Increased height
    ])
], fluid=True)

@app.callback(
    Output('map-graph', 'figure'),
    [
        Input('year-dropdown', 'value'),
        Input('municipio-dropdown', 'value'),
        Input('indicador-dropdown', 'value')
    ]
)
def update_map(selected_years, selected_municipios, selected_indicadores):
    filtered_df = pivot_df.copy()

    # Filter based on selected years
    if selected_years:
        filtered_df = filtered_df[filtered_df['ANO'].isin(selected_years)]

    # Filter based on selected municipalities
    if selected_municipios:
        filtered_df = filtered_df[filtered_df['MUNICIPIO'].isin(selected_municipios)]

    # If no indicators are selected, return an empty figure
    if not selected_indicadores:
        fig = px.scatter_mapbox(lat=[], lon=[], mapbox_style="mapbox://styles/mapbox/satellite-v9")
    else:
        # Drop rows with missing values in the selected indicators
        filtered_df = filtered_df.dropna(subset=selected_indicadores)
        fig = px.scatter_mapbox(
            filtered_df,
            lat='LATITUD',
            lon='LONGITUD',
            color=selected_indicadores[0],
            size=selected_indicadores[0],
            hover_name='MUNICIPIO',
            hover_data={col: True for col in ['DEPARTAMENTO', 'MUNICIPIO', 'ANO'] + selected_indicadores},
            mapbox_style="mapbox://styles/mapbox/satellite-v9",
            color_continuous_scale=px.colors.sequential.Turbo, #color_continuous_scale=px.colors.sequential.Plasma, color_continuous_scale=px.colors.sequential.Blues


            title=f"Indicadores: {', '.join(selected_indicadores)}"
        )

    fig.update_layout(mapbox=dict(center=dict(lat=4.5709, lon=-74.2973), zoom=5), height=780)

    return fig

if __name__ == '__main__':
    app.run_server(debug=False)