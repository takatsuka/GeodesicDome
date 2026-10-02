# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.

# 必要なライブラリをインポート。
import numpy as np
import plotly.graph_objects as go

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

# 描画に必要なグラフやボタン、ドロップダウンなどのUIを提供するパッケージ。
# `dash_html_components`は、DivタグやH1タグなどのHTMLタグを提供するパッケージ。

dome = GeodesicDome()
dome.split(5)
points = dome.get_all_xyz()
x, y, z = points.T
tri_points = np.reshape(dome.get_all_triangles(), (-1, 3))
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
lines = go.Scatter3d(
    x=Xe,
    y=Ye,
    z=Ze,
    mode='lines',
    name='',
    line=dict(color='rgb(70,70,70)', width=1))

layout = go.Layout(
    scene=dict(camera=dict(eye=dict(x=0.0, y=0.0, z=1.0)),  # the default values are 1.25, 1.25, 1.25
               xaxis=dict(),
               yaxis=dict(),
               zaxis=dict(),
               aspectmode='data',  # this string can be 'data', 'cube', 'auto', 'manual'
               # a custom aspectratio is defined as follows:
               #aspectratio=dict(x=1, y=1, z=0.95)
               )
)

fig = go.Figure(data=[mesh, lines], layout=layout)
fig.show()
