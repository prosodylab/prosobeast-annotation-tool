#!/usr/bin/env python3
"""
Utility functions for the ProsoBeast annotation tool.

Created on Sat Feb 8 2020

@author: Branislav Gerazov
"""
import numpy as np
import pandas as pd

from bokeh.models import (
    ColumnDataSource, CustomJS, Slider,
    RadioGroup,
    RadioButtonGroup,
    CheckboxGroup
    )
from bokeh.plotting import figure
from bokeh.models.glyphs import MultiLine
from bokeh.embed import components
from bokeh.layouts import row, column, Spacer
from bokeh.palettes import d3

import json
from sqlalchemy import create_engine


def upload_csv_to_db(file_name, sql_table_name, check_nans=True):
    """Loads data from CSV into an SQLite database.
    """
    df = pd.read_csv(file_name)
    # print('load_csv', df)

    if sql_table_name == 'prosobeast':
        # if not labels provided
        if check_nans:
            # check for lables, nan = No label
            mask_no_labels = df.label.isna()
            df.loc[mask_no_labels, 'label'] = 'No label'

        # reset locations based on new data
        columns = df.columns.tolist()
        columns_loc = []
        location_labels = []
        for col in columns:
            if 'location' in col:
                columns_loc.append(col)
                try:
                    location_labels.append(col.split('_')[1])
                except IndexError:  # if column name is just location
                    location_labels.append('user')
        # save location labels to database
        update_locations_db(location_labels, 'locations')

    # save to database
    update_db_table_from_df(df, sql_table_name, jsonify=False)

    return df


def update_db_table_from_df(source_df, sql_table_name, jsonify=True):
    """Update database table based on DataFrame input.
    """
    # necessary - otherwise source_df is altered
    source_df = source_df.copy()
    if jsonify:
        for col in ['f0', 'x', 'y']:
            if col in source_df.columns:
                source_df[col] = source_df[col].map(
                    lambda x: json.dumps(x.tolist())
                    )
    # settle index
    if 'id' not in source_df.columns:
        source_df['id'] = source_df.index
    if 'index' not in source_df.columns:
        source_df['index'] = source_df.index
    engine = create_engine('sqlite:///prosobeast.sqlite3')
    with engine.connect() as con:
        source_df.to_sql(
            sql_table_name, con=con,
            if_exists='replace',
            index=False,
            )


def update_locations_db(location_labels, sql_table_name='locations'):
    df_locations = pd.DataFrame(location_labels, columns=['labels'])
    print('df_locations', df_locations)
    update_db_table_from_df(df_locations, sql_table_name, jsonify=False)


def load_df_from_db_table(sql_table_name, unjsonify=False):
    """Update database table based on DataFrame input.
    """
    engine = create_engine('sqlite:///prosobeast.sqlite3')
    with engine.connect() as con:
        df = pd.read_sql(sql_table_name, con)
    if unjsonify:
        cols = (
            ['f0', 'x', 'y']
            )
        for col in cols:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: np.array(json.loads(x))
                    )
    return df


def check_table_exists(table_name):
    """Check if table name exists in sql db.
    """
    engine = create_engine('sqlite:///prosobeast.sqlite3')
    tb_exists = (
        f"SELECT name FROM sqlite_master "
        f"WHERE type='table' AND name='{table_name}'"
        )
    with engine.connect() as con:
        if con.execute(tb_exists).fetchone():
            return True
        else:
            return False


def auto_generate_color_labels(source_df):
    """Loads data from CSV and generates a color code for each label.
    """
    labels = source_df.label.unique().tolist()
    # print(labels)
    if len(labels) == 1:  # just No label
        colors = ['#7f7f7f']  # gray
    else:
        colors = d3['Category10'][len(labels)]
    # change No label to gray
    i = labels.index('No label')
    colors[i] = '#7f7f7f'  # gray
    # store to CSV
    labels_df = pd.DataFrame(columns=['label', 'color'])
    labels_df.label = labels
    labels_df.color = colors
    labels_df.to_csv('labels.csv', index=False)
    return labels_df


def save_csv():
    """Saves database changes to CSV.
    """
    engine = create_engine('sqlite:///prosobeast.sqlite3')
    with engine.connect() as con:
        source_df = pd.read_sql('prosobeast', con)
    for c in ['id', 'index']:
        if c in source_df.columns:
            print(f'found {c} in columns - deleting ...')
            source_df.drop(columns=c, inplace=True)
    source_df.to_csv('prosobeast.csv', index=False)


def load_database(location=None):
    source_df = load_df_from_db_table('prosobeast', unjsonify=True)
    locations_labels_df = load_df_from_db_table('locations', unjsonify=False)
    labels_df = load_df_from_db_table('labels', unjsonify=False)
    if 'color' not in source_df.columns:
        update_db_colors()
        source_df = load_df_from_db_table('prosobeast', unjsonify=True)
    # calculate x and y data for all locations
    location_labels = locations_labels_df.labels.tolist()
    source_df, xs_df, ys_df, scales_dict = update_db_contour_locs(
        source_df, location_labels,
        location=location,
        )
    update_db_table_from_df(source_df, 'prosobeast', jsonify=True)
    labels = labels_df.label.tolist()
    colors = labels_df.color.tolist()
    return (
        source_df, labels, colors,
        location_labels, xs_df, ys_df, scales_dict
        )


def update_db_colors():
    """Add a database color column using labels' color code mapping.
    """
    engine = create_engine('sqlite:///prosobeast.sqlite3')
    with engine.connect() as con:
        source_df = pd.read_sql('prosobeast', con)
        labels_df = pd.read_sql('labels', con)
    labels = labels_df.label.tolist()
    colors = labels_df.color.tolist()
    labels_colors = dict(zip(labels, colors))
    labels_colors[None] = ""
    source_df['color'] = source_df.label.map(
        lambda x: labels_colors[x]
        )
    update_db_table_from_df(source_df, 'prosobeast', jsonify=False)


def update_db_contour_locs(
        source_df, location_labels,
        location=None):
    """Calculates contours' x and y for plotting for all labels and
    sets source_df to location.
    """
    if location is None:
        location = location_labels[0]  # user or first one calculated
    # define default x and y scales
    scales_defaults = {
        'user': [0.4, 0.07],
        'PCA': [15, 0.06],
        't-SNE': [9, 0.04],
        'VAE-2D': [0.65, 0.03],
        'VAE-4D': [0.35, 0.006],
        'RVAE-10D': [0.75, 0.015],
        }
    scales_dict = {}
    for label in location_labels:
        for k, v in scales_defaults.items():
            if k in label:
                scales_dict[label] = v
                break
        else:
            print(f'No default scaling found for {label}')
            scales_dict[label] = v

    # find each contour's length
    source_df['length'] = source_df.f0.map(
        lambda x: len(x)
        )
    # find min length to make longer contur appear longer
    min_len = source_df.length.min()

    # find center indexes
    source_df['center_ind'] = source_df.length.map(
        lambda x: x//2
        )

    # one option is to do this only once and store in db
    # but this won't work with the sliders changing ...
    xs_df = pd.DataFrame(columns=location_labels)
    ys_df = pd.DataFrame(columns=location_labels)

    for label in location_labels:
        if label == 'user':
            # in the csv the column name is just location
            source_label = 'location'
        else:
            source_label = f'location_{label}'

        # generate x data
        x_scale, y_scale = scales_dict[label]
        xs_df[label] = source_df.length.map(
            lambda x: np.linspace(-1, 1, x) * (x/min_len) * x_scale
            )
        xs_df[label] += source_df[source_label].map(
            lambda loc: json.loads(loc)[0]
            )
        # generate y data
        ys_df[label] = source_df.f0.map(
            lambda y: y * y_scale
            )
        ys_df[label] += source_df[source_label].map(
            lambda loc: json.loads(loc)[1]
            )
    # set to default in source_df
    source_df['x'] = xs_df[location]
    source_df['y'] = ys_df[location]
    return source_df, xs_df, ys_df, scales_dict


def bokeh_plot(source):
    tooltips = [
        ("file", "@file"),
        ("info", "@info"),
        ("label", "@label"),
        ]
    bfig = figure(
        title="Intonation space",
        tooltips=tooltips,
        tools='pan, wheel_zoom, tap, hover, reset, save',
        active_scroll="wheel_zoom",
        toolbar_location="left",
        plot_width=950, plot_height=700,
        )
    bfig.xaxis.axis_label = 'dimension 0'
    bfig.yaxis.axis_label = 'dimension 1'

    ml = bfig.multi_line(
        xs="x", ys="y",
        line_color="color",
        line_width=3, line_alpha=.8,
        source=source,
        muted_color="color",
        muted_alpha=0.15,
        legend_field="label")

    ml.selection_glyph = MultiLine(
        line_width=6,
        line_color="color")
    ml.nonselection_glyph = None

    bfig.legend.location = "bottom_right"
    return bfig


def plot(location=None):

    (
        source_df, labels, colors,
        location_labels, xs_df, ys_df, scales_dict
        ) = load_database(location=location)
    print(location_labels)
    if location is None:
        location = location_labels[0]
    print(location)
    source = ColumnDataSource(source_df)
    xs = ColumnDataSource(xs_df)
    ys = ColumnDataSource(ys_df)
    scales = ColumnDataSource(scales_dict)

    bfig = bokeh_plot(source)

    labels_group = RadioGroup(
        labels=labels, active=None)

    active = location_labels.index(location)
    locations_group = RadioButtonGroup(
        labels=location_labels, active=active, name='locations_group'
        )

    x_scale, y_scale = scales_dict[location]

    x_slider = Slider(
        start=0.01, end=200, value=x_scale, step=0.05,
        title='x scale'
        )
    y_slider = Slider(
        start=0.0001, end=2, value=y_scale, step=0.001,
        title='y scale'
        )

    with open('js/bokeh_plot_onclick.js', 'r') as f:
        code = f.read()
    source.selected.js_on_change(
        'indices',
        CustomJS(
                args=dict(
                    s=source,
                    ls=labels,
                    r=labels_group,
                    ),
                code=code
                )
        )
    with open('js/bokeh_label_onchange.js', 'r') as f:
        code = f.read()
    labels_group.js_on_change(
        'active',
        CustomJS(
            args=dict(
                s=source,
                r=labels_group,
                ls=labels,
                cs=colors),
            code=code
            )
        )

    with open('js/bokeh_locations_onchange.js', 'r') as f:
        code = f.read()
    locations_group.js_on_change(
        'active',
        CustomJS(
            args=dict(
                s=source,
                r=locations_group,
                location_labels=location_labels,
                xs=xs,
                ys=ys,
                scales=scales,
                x_slider=x_slider,
                y_slider=y_slider,
                ),
            code=code
            )
        )

    with open('js/bokeh_x_slider_onchange.js', 'r') as f:
        code = f.read()
    x_slider.js_on_change(
        'value',
        CustomJS(
            args=dict(
                s=source,
                r=locations_group,
                location_labels=location_labels,
                xs=xs,
                ys=ys,
                scales=scales,
                x_slider=x_slider,
                y_slider=y_slider,
                ),
            code=code
            )
        )
    with open('js/bokeh_y_slider_onchange.js', 'r') as f:
        code = f.read()
    y_slider.js_on_change(
        'value',
        CustomJS(
            args=dict(
                s=source,
                r=locations_group,
                location_labels=location_labels,
                xs=xs,
                ys=ys,
                scales=scales,
                x_slider=x_slider,
                y_slider=y_slider,
                ),
            code=code
            )
        )

    return components(
        column(
            row(
                locations_group,
                ),
            row(
                x_slider,
                y_slider,
                ),
            row(
                bfig,
                column(
                    labels_group,
                    )
                )
            )
        )
