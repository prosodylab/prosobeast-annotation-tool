#!/usr/bin/env python3
"""
ProsoBeast annotation tool.

Created on Sat Feb 8 2020

@author: Branislav Gerazov
"""
import os
import shutil
import json
import importlib
from flask import (
    Flask, render_template, redirect, request, Response, send_file, abort
    )
from flask_sqlalchemy import SQLAlchemy

import annotation_utils as utils
from data_spread import calculate_data_spread

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prosobeast.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = ['csv', 'wav']


class Labels(db.Model):
    index = db.Column(db.Integer, nullable=True)
    label = db.Column(db.String(64), nullable=True)
    color = db.Column(db.String(64), nullable=True)
    id = db.Column(db.Integer, nullable=False, primary_key=True)

    class Meta:
        managed = False
        db_table = 'labels'


class Locations(db.Model):
    index = db.Column(db.Integer, nullable=True)
    labels = db.Column(db.String(64), nullable=True)
    id = db.Column(db.Integer, nullable=False, primary_key=True)

    class Meta:
        managed = False
        db_table = 'locations'


class Prosobeast(db.Model):
    index = db.Column(db.Integer, nullable=True)
    file = db.Column(db.String(64), nullable=True)
    info = db.Column(db.String(64), nullable=True)
    label = db.Column(db.String(64), nullable=True)
    color = db.Column(db.String(64), nullable=True)
    f0 = db.Column(db.String(64), nullable=True)
    x = db.Column(db.String(64), nullable=True)
    y = db.Column(db.String(64), nullable=True)
    id = db.Column(db.Integer, nullable=False, primary_key=True)

    class Meta:
        managed = False
        db_table = 'prosobeast'


@app.route('/', methods=['GET'])
def home(location=None):
    if (
            not os.path.isfile('prosobeast.sqlite3')
            or not utils.check_table_exists("prosobeast")
            or not utils.check_table_exists("labels")
            # or not os.path.isdir('static/audio')
            ):
        return redirect('/init')
    else:
        # check if there is a data spread calculated
        locations_df = utils.load_df_from_db_table('locations', unjsonify=False)
        print('/ locations_df\n', locations_df)
        if locations_df is not None and (
                not locations_df['labels'].tolist()
                ):
            return redirect('/data_spread')
        else:
            if request.args:
                location = request.args['location']
            else:
                location = None
            # locations = [x.labels for x in Locations.query.all()]
            locations = locations_df.labels.to_list()
            plot_script, plot_div = utils.plot(location=location)
            return render_template(
                    'annotate.html',
                    plot_script=plot_script,
                    plot_div=plot_div,
                    locations=locations
                    )


@app.route('/contour_clicked', methods=['POST'])
def contour_clicked():
    if request.method == "POST":
        data = request.form
        if data['index[]']:
            ind = data['index[]']
            print(ind)
            utterance = Prosobeast.query.get(ind)
            file_name = utterance.file
            return Response(file_name)
        else:
            return Response('No clicked contour.')
    else:
        return redirect('/')


@app.route('/form_clicked', methods=['POST'])
def form_clicked():
    if request.method == "POST":
        data = request.form
        if data['new_label']:
            new_label = data['new_label']
            ind = data['index[]']

            utterance = Prosobeast.query.get(ind)
            old_label = utterance.label
            if old_label != new_label:
                # update database
                utterance.label = new_label
                # get color and update
                utterance.color = Labels.query.filter_by(
                    label=new_label
                    ).first().color
                db.session.commit()
                return Response(
                    f'Change! '
                    f'Old label {old_label} changed to '
                    f'New label {new_label}, '
                    f'for contour {ind}'
                    )
            else:
                return Response(
                    f'No change! '
                    f'Old label {old_label} same with '
                    f'New label {new_label}, '
                    f'for contour {ind}'
                    )
        else:
            new_label = None
            ind = None
            return Response('Nothing selected!')
    else:
        return redirect('/')


@app.route('/reload_csv')
def reload_csv():
    utils.upload_csv_to_db('prosobeast.csv', 'prosobeast')
    return redirect('/')


@app.route('/save_csv', methods=['POST'])
def save_csv():
    if request.method == "POST":
        utils.save_csv()
        return Response('CSV updated!')
    else:
        return redirect('/')


@app.route('/download', methods=['GET'])
def download():
    try:
        return send_file(
            'prosobeast.csv',
            as_attachment=True,
            attachment_filename='prosobeast.csv',
            mimetype='text/csv',
            )
    except FileNotFoundError:
        abort(404)


def allowed_file(filename, extensions):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in extensions
        )


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == "POST":
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename, ['csv', 'CSV']):
                data_name = 'prosobeast.csv'
                file.save(data_name)
                __ = utils.upload_csv_to_db(
                    data_name, 'prosobeast'
                    )
                return Response('Success!')
            else:
                return Response('Wrong file format!')
        else:
            return Response('No file selected!')
    else:
        return redirect('/')


@app.route('/init', methods=['GET'])
def init():
    # try:
    #     print(Prosobeast.query.get(ind))
    #     data_uploaded = 'true'
    # except:
    #     data_uploaded = 'false'
    if utils.check_table_exists("prosobeast"):
        data_uploaded = 'true'
    else:
        data_uploaded = 'false'
    if utils.check_table_exists("labels"):
        labels_uploaded = 'true'
    else:
        labels_uploaded = 'false'
    if os.path.isdir('static/audio'):
        audio_uploaded = 'true'
    else:
        audio_uploaded = 'false'
    print(data_uploaded)
    return render_template(
        'init.html',
        data_uploaded=json.dumps(data_uploaded),
        labels_uploaded=json.dumps(labels_uploaded),
        audio_uploaded=json.dumps(audio_uploaded),
        )

@app.route('/deletedb_init', methods=['GET'])
def deletedb_init():
    if os.path.isfile('prosobeast.sqlite3'):
        print('sqlite3 database found will remove it ...')
        # before uploading new data delete old db
        os.remove('prosobeast.sqlite3')
        os.remove('prosobeast.csv')
        os.remove('labels.csv')
    if os.path.isdir('static/audio'):
        shutil.rmtree('static/audio')
    return redirect('/init')


@app.route('/data_spread', methods=['GET'])
def data_spread():
    if (
            utils.check_table_exists("prosobeast")
            and utils.check_table_exists("labels")
            # and os.path.isdir('static/audio')
            ):
        goback = '/'
    else:
        goback = '/init'
    if importlib.util.find_spec("torch") is None:
        torch_present = False
    else:
        torch_present = True
    if importlib.util.find_spec("sklearn") is None:
        sklearn_present = False
    else:
        sklearn_present = True
    return render_template(
        'data_spread.html',
        goback=goback,
        gobackjson=json.dumps(goback),
        torch_present=torch_present,
        sklearn_present=sklearn_present,
        )


@app.route('/calculate', methods=['POST'])
def calculate():
    print('loading data ...')
    source_df = utils.load_df_from_db_table('prosobeast', unjsonify=False)
    locations_df = utils.load_df_from_db_table('locations', unjsonify=False)
    location_labels = locations_df['labels'].tolist()
    print(location_labels)
    print(f'processing {request.data}')
    if request.method == "POST":
        choice = request.data.decode("utf-8")
        try:
            locs = calculate_data_spread(
                data=source_df.copy(),
                choice=choice,
                seed=None,
                )
        except ValueError as e:
            return Response(str(e))
        # update source_df
        locs['location'] = locs['location'].map(
                lambda x: json.dumps(x)
                )
        print(locs)
        choice_to_label = {
            'pca': 'PCA',
            'tsne': 't-SNE',
            'vae2d': 'VAE-2D',
            'vae4d': 'VAE-4D',
            'rvae10d': 'RVAE-10D',
            }
        label = choice_to_label[choice]
        # if there is already a location column with that name add a counter
        columns = source_df.columns
        count = 0
        label_unique = label
        while f'location_{label_unique}' in columns:
            count += 1
            label_unique = f'{label}_{count:02d}'
        label = label_unique
        source_label = f'location_{label}'
        i = [i for i, c in enumerate(columns) if 'location' in c]
        if i:  # if not empty - i.e. at least one location exists
            i_insert = i[-1] + 1
        else:
            i_insert = 3
        print(f'inserting column {source_label} at index {i}')
        source_df.insert(i_insert, source_label, locs['location'])

        print('updating dbs ...')
        utils.update_db_table_from_df(
            source_df, 'prosobeast', jsonify=False)
        # update locations
        location_labels += [label]
        print(location_labels)
        utils.update_locations_db(location_labels, 'locations')
        return Response(f'Done! Generated data spread {label}.')
    else:
        return Response('No file selected!')


@app.route('/delete_data_spread', methods=['POST'])
def delete_data_spread():
    if request.method == "POST":
        label = request.form['data_spread']
        print(f'Removing location label: {label} ...')
        # remove from labels
        locations_df = utils.load_df_from_db_table('locations', unjsonify=False)
        location_labels = locations_df['labels'].tolist()
        ind = location_labels.index(label)
        if ind == 0:
            prev_location = 0
        else:
            prev_location = location_labels[ind - 1]
        location_labels.remove(label)
        utils.update_locations_db(location_labels)
        # remove from source_df
        source_df = utils.load_df_from_db_table('prosobeast', unjsonify=False)
        if label == 'user':
            source_label = f'location'
        else:
            source_label = f'location_{label}'
        source_df.drop(columns=source_label, inplace=True)
        utils.update_db_table_from_df(source_df, 'prosobeast', jsonify=False)
        print(' done!')
        print('Redirecting ...')
        if location_labels:  # if any left
            response = f'/?location={prev_location}'
        else:
            response = f'/data_spread'
        # response = 'Success!'
    else:
        response = 'Wrong request type!'
    return Response(response)


@app.route('/init_upload_data', methods=['POST'])
def init_upload_data():
    if request.method == "POST":
        file = request.files['file']
        if file:
            if allowed_file(file.filename, ['csv', 'CSV']):
                # save to disk
                data_name = 'prosobeast.csv'
                file.save(data_name)
                source_df = utils.upload_csv_to_db(
                    data_name, 'prosobeast'
                    )
                # auto init colors if no labels
                # if os.path.isfile('labels.csv'):
                #     print('Found labels.csv creating database ...')
                #     data_name = 'labels.csv'
                #     labels_df = utils.load_csv_to_db(
                #         data_name, 'labels'
                #         )
                #     utils.update_db_colors()
                # else:
                #     labels_df = utils.auto_generate_color_labels(source_df)
                #     utils.update_db_tables_from_df(labels_df, 'labels')
                return Response('Success!')
            else:
                return Response('Wrong file format!')
        else:
            return Response('No file selected!')
    else:
        return redirect('/init')


@app.route('/init_upload_labels', methods=['POST'])
def init_upload_labels():
    if request.method == "POST":
        file = request.files['file']
        if file:
            if allowed_file(file.filename, ['csv', 'CSV']):
                file.save('labels.csv')
                labels_df = utils.upload_csv_to_db(
                    'labels.csv', 'labels', check_nans=False)
                print('init_labels', labels_df)
                # update colors in data
                utils.update_db_colors()
                return Response('Success!')
            else:
                return Response('Wrong file format!')
        else:
            return Response('No file selected!')
    else:
        return redirect('/init')


@app.route('/init_upload_audio', methods=['POST'])
def init_upload_audio():
    if request.method == "POST":
        uploaded_files = request.files.getlist("audio_files[]")
        if uploaded_files:
            os.makedirs('static/audio', exist_ok=True)
            success = False
            for file in uploaded_files:
                print(file.filename, end='')
                if file:
                    if allowed_file(file.filename, ['wav', 'WAV']):
                        # filename = secure_filename(file.filename)
                        basename = os.path.basename(file.filename)
                        file.save('static/audio/'+basename)
                        print(' saved.')
                        success = True
                    else:
                        print(' wrong file format!')
                else:
                    return Response('No files selected!')
            if success:
                response = 'Success!'
            else:
                response = 'Wrong file format!'
            return Response(response)
    else:
        return redirect('/init')
