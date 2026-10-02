# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.projection.projection import Projection
from mt.geodesicdome.projection.wagner import WagnerVI

# 描画に必要なグラフやボタン、ドロップダウンなどのUIを提供するパッケージ。
# `dash_html_components`は、DivタグやH1タグなどのHTMLタグを提供するパッケージ。

dome = GeodesicDome()
dome.split(10)
# projection: Projection = KavrayskiyVII()
projection: Projection = WagnerVI()
triangles2D = projection.build(dome)

proj_vertices = []
for p in dome.get_all_vertices():
    proj_vertices.append(np.append(p.projected_coord, [0]))

x, y, z = np.array(proj_vertices).T

tri_points = triangles2D  # np.reshape(triangles2D, (-1, 3))
i, j, k = tri_points.T

mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightpink', opacity=1.0)

Xe = []
Ye = []
Ze = []

for T in tri_points:
    Xe.extend([x[T[k % 3]] for k in range(4)] + [None])
    Ye.extend([y[T[k % 3]] for k in range(4)] + [None])
    Ze.extend([z[T[k % 3]] for k in range(4)] + [None])

# define the trace for triangle sides
lines = go.Scatter(
    x=Xe,
    y=Ye,
    mode='lines',
    name='',
    fill='toself',
    line=dict(color='rgb(70,70,70)', width=0.5)
)


layout = go.Layout(
dragmode=False,
)

mesh_names = ['sandal', 'scissors', 'shark', 'walkman']

app = Dash(__name__)

app.layout = html.Div([
    html.H4('PLY Object Explorer'),
    html.P("Choose an object:"),
    dcc.Dropdown(
        id='dropdown',
        options=mesh_names,
        value="sandal",
        clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Input("graph", "selectedData"))
def display_mesh(data):
    print(f'data: {data}')
    fig = go.Figure(data=[lines], layout=layout)
    # df = dataframes[name] # replace with your own data source

    '''
    fig.add_trace(
        go.Scatter3d(x=[0, 1, 0, 0],
                     y=[0, 0, 1, 0],
                     z=[0, 0, 0, 1],
                     mode='markers')
    )
    '''

    return fig


if __name__ == '__main__':
    # Dash 3 removed app.run_server(); app.run() works on Dash 2.x and 3.x
    app.run(debug=True)
