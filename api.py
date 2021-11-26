# python 3
# Author: Zhigang Lu
# Contact: zhigang.lu@mq.edu.au

import os, shutil
import numpy as np

from builtins import int, str, float
from flask import Flask, request, flash, redirect, render_template, abort, jsonify, send_from_directory
# from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras

from package_train_model.classifier import train_classifier
from package_dp_model.dp_classifier import dp_weight
from package_data_io.read_data import build_from_csv, write_to_csv

BASE_DIR = '/home/hub62/Documents/dpaip_api/'
STATIC_URL_PATH = os.path.join(BASE_DIR, 'docs/')
STATIC_DIR = os.path.join(BASE_DIR, 'docs/build/html/')
UPLOAD_DIR = os.path.join(BASE_DIR, 'upload_files/')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models/')
ALLOWED_EXTENSIONS = {'json', 'csv'}

API_KEY = 'i0cgsdYL3hpeOGkoGmA2TxzJ8LbbU1HpbkZo8B3kFG2bRKjx3V'

app = Flask(__name__, template_folder=BASE_DIR, static_url_path=STATIC_URL_PATH, static_folder=STATIC_DIR)
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['JSON_SORT_KEYS'] = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# training func
def train_model(
    data_source,
    train_num,
    test_num,
    class_num,
    hidden_layer_num,
    hidden_neuron_num,
    epochs,
    batch_size,
    learning_rate,
    l2_regulariser,
    hidden_act
    ):
    # save some hyper-parameters for prediction
    f = open(os.path.join(OUTPUT_DIR, data_source + '-' + str(class_num), 'hyper.txt'), 'w+')
    f.write(str(train_num) + ',' + str(class_num) + ',' + str(l2_regulariser))
    f.close()

    # make a directory for the data_source
    os.makedirs(os.path.join(OUTPUT_DIR, data_source + '-' + str(class_num)), exist_ok=True)

    # read dataset, train-test split
    data_path = os.path.join(DATASETS_DIR, 'raw_set.csv')
    features, labels = build_from_csv(class_num, data_path, is_labelled=True)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=train_num, test_size=test_num)

    # save x_test and y_test for dp performance
    ground_truth_test_label = np.argmax(y_test, axis=1)
    write_path = os.path.join(DATASETS_DIR, 'pred_set_labelled.csv')
    write_to_csv(x_test, ground_truth_test_label, write_path, True)

    # train models with surrogate loss
    i = 0.5
    max_training_acc = 0
    max_test_acc = 0
    while i <= 10:
        keras.backend.clear_session()
        surrogate_model = train_classifier(
            n_class=class_num, 
		    n_hidden_layer=hidden_layer_num, 
		    hidden_neurons=hidden_neuron_num, 
		    final_hidden_neurons=hidden_neuron_num,
		    epoch=epochs, 
		    batchsize=batch_size, 
		    l_r=learning_rate, 
		    l2_reg=l2_regulariser, 
		    hidden_activation=hidden_act, 
		    output_activation='softmax', 
		    training_features=x_train, 
		    training_labels=y_train,
            test_features=x_test,
            test_labels=y_test,
            is_surrogate=True,
            convex=i
        )
        _, training_accuracy_temp = surrogate_model.evaluate(x_train, y_train)
        _, test_accuracy_temp = surrogate_model.evaluate(x_test, y_test)

        if test_accuracy_temp > max_test_acc:
            max_test_acc = test_accuracy_temp
            max_training_acc = training_accuracy_temp
            shutil.copyfile(os.path.join(OUTPUT_DIR, 'attack_loss.pdf'), os.path.join(OUTPUT_DIR, data_source + '-' + str(class_num), 'attack_loss.pdf'))
            surrogate_model.save(os.path.join(MODEL_DIR, data_source + '-' + str(class_num)))

        i = i + 0.5
            
    # write accuracy to file
    f = open(os.path.join(OUTPUT_DIR, data_source + '-' + str(class_num), 'out.txt'), 'w+')
    f.write(str(max_training_acc) + ',' + str(max_test_acc))
    f.close()

# pred func
def make_prediction(model_name, epsilon, has_label):
    # collect the parameters from saved file
    f = open(os.path.join(OUTPUT_DIR, str(model_name), 'hyper.txt'), 'r')
    hyper = f.read().rsplit(',')
    n_train = int(hyper[0])
    n_class = int(hyper[1])
    l2_regulariser = float(hyper[2])

    # read dataset for prediction
    model_path = os.path.join(MODEL_DIR, str(model_name))
    if has_label == False:
        data_path = os.path.join(DATASETS_DIR, 'pred_set.csv')
    else:
        data_path = os.path.join(DATASETS_DIR, 'pred_set_labelled.csv')
    dp_pred_vec, ground_truth_label = dp_weight(
        rec_num=n_train, 
	    label_num=n_class, 
	    l2_reg=l2_regulariser,
	    priv_budget=epsilon, 
	    trained_model_path=model_path, 
	    data_path=data_path, 
	    dp_mode='gaussian',
        is_labelled=has_label
    )
    
    return dp_pred_vec, ground_truth_label

'''
FOR DOCUMENTATION
'''
@app.route('/help')
@app.route('/<path:path>')
def serve_sphinx_docs(path='index.html'):
    return app.send_static_file(path)

'''
FOR WEB APP
'''
@app.route('/web', methods=['GET', 'POST'])
def dp_ml():
    training_acc_ = 'waiting'
    test_acc_ = 'waiting'
    model_list_ = 'no available model'
    dp_pred_vec = None
    dp_pred_label = None
    pred_output = None
    dp_test_acc_ = None
    non_priv_test_acc_ = None
    
    if request.method == 'POST':
        # If it is an uploading request
        if request.form['submit_button'] == 'Train':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
                # copy the upload to dataset
                shutil.copyfile(os.path.join(UPLOAD_DIR, filename), os.path.join(DATASETS_DIR, 'raw_set.' + filename.rsplit('.', 1)[1].lower()))

            # collect the hyper parameters
            train_model(
                data_source = request.form.get('training_set', type=str),
                train_num = request.form.get('n_train', type=int),
                test_num = request.form.get('n_test', type=int),
                class_num = request.form.get('n_labels', type=int),
                hidden_layer_num = request.form.get('n_hidden_layers', type=int),
                hidden_neuron_num = request.form.get('n_hidden_neurons', type=int),
                epochs = request.form.get('n_epochs', type=int),
                batch_size = request.form.get('n_batch', type=int),
                learning_rate = request.form.get('learning_rate', type=float),
                l2_regulariser = request.form.get('l2_reg', type=float),
                hidden_act = request.form.get('activation', type=str)
            )
            
        # If it is a checking training status request
        elif request.form['submit_button'] == 'Check':
            model_list_ = [f.name for f in os.scandir(MODEL_DIR) if f.is_dir()]

        # If it is a showing performance request
        elif request.form['submit_button'] == 'Show':
            model_name = request.form.get('model_name', type=str)
            f = open(os.path.join(OUTPUT_DIR, model_name, 'out.txt'), 'r')
            model_performance = f.read().rsplit(',')
            training_acc_ = model_performance[0]
            test_acc_ = model_performance[1]

        # If it is a prediction request
        elif request.form['submit_button'] == 'Predict':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
                # copy the upload to dataset
                shutil.copyfile(os.path.join(UPLOAD_DIR, filename), os.path.join(DATASETS_DIR, 'pred_set.' + filename.rsplit('.', 1)[1].lower()))

            # make dp predict
            dp_pred_vec, _ = make_prediction(
                model_name = request.form.get('model_name', type=str),
                epsilon = request.form.get('epsilon', type=float),
                has_label = False
            )
            dp_pred_label = np.argmax(dp_pred_vec, axis=1)

            # pred_output = []
            # for i in range(len(dp_pred_label)):
            #     pred_output.append({'predicted_label': int(dp_pred_label[i]), 'probabilities': dp_pred_vec[i].tolist()})
        
        # If it is a compare request
        elif request.form['submit_button'] == 'Compare':
            dp_pred_vec, ground_truth_vec = make_prediction(
                model_name = request.form.get('model_name', type=str),
                epsilon = request.form.get('epsilon', type=float),
                has_label = True
            )
            ground_truth_label = np.argmax(ground_truth_vec, axis=1)
            dp_pred_label = np.argmax(dp_pred_vec, axis=1)
            dp_test_acc_ = accuracy_score(ground_truth_label, dp_pred_label)

            f = open(os.path.join(OUTPUT_DIR, request.form.get('model_name', type=str), 'out.txt'), 'r')
            performance = f.read().rsplit(',')
            non_priv_test_acc_ = performance[1]
    
    # remove datasets and upload files
    shutil.rmtree(UPLOAD_DIR)
    # re-create empty dir
    os.mkdir(UPLOAD_DIR)
    
    return render_template('web_app.html', model_list=model_list_, training_acc=training_acc_, test_acc=test_acc_, prediction=dp_pred_label, dp_test_acc=dp_test_acc_, non_priv_test_acc=non_priv_test_acc_)


'''
FOR API CALLS
'''
# upload raw set for training
@app.route('/upload/train/<filename>', methods=['POST'])
def upload_train(filename):
    '''Upload a file.'''

    if '/' in filename:
        # Return 400 BAD REQUEST
        abort(400, 'no subdirectories allowed')
    
    if filename and allowed_file(filename):
        with open(os.path.join(UPLOAD_DIR, filename), 'wb') as f:
            f.write(request.data)
    
        # copy the upload to dataset
        shutil.copyfile(os.path.join(UPLOAD_DIR, filename), os.path.join(DATASETS_DIR, 'raw_set.' + filename.rsplit('.', 1)[1].lower()))

        # clean-up
        shutil.rmtree(UPLOAD_DIR)
        os.mkdir(UPLOAD_DIR)

        # Return 201 CREATED
        return '', 201
    else:
        # clean-up
        shutil.rmtree(UPLOAD_DIR)
        os.mkdir(UPLOAD_DIR)

        abort(400, 'file ext not allowed')

# upload files for prediction
@app.route('/upload/predict/<filename>', methods=['POST'])
def upload_predict(filename):
    '''Upload a file.'''

    if '/' in filename:
        # Return 400 BAD REQUEST
        abort(400, 'no subdirectories allowed')

    if filename and allowed_file(filename):
        with open(os.path.join(UPLOAD_DIR, filename), 'wb') as f:
            f.write(request.data)
    
        # copy the upload to dataset
        shutil.copyfile(os.path.join(UPLOAD_DIR, filename), os.path.join(DATASETS_DIR, 'pred_set.' + filename.rsplit('.', 1)[1].lower()))

        # clean-up
        shutil.rmtree(UPLOAD_DIR)
        os.mkdir(UPLOAD_DIR)

        # Return 201 CREATED
        return '', 201
    else:
        # clean-up
        shutil.rmtree(UPLOAD_DIR)
        os.mkdir(UPLOAD_DIR)

        abort(400, 'file extension not allowed')

# train model
@app.route('/train', methods=['POST'])
def train():
    hyper_par = request.get_json()
    train_model(
        data_source = str(hyper_par['training_set']),
        train_num = int(hyper_par['n_train']),
        test_num = int(hyper_par['n_test']),
        class_num = int(hyper_par['n_labels']),
        hidden_layer_num = int(hyper_par['n_hidden_layers']),
        hidden_neuron_num = int(hyper_par['n_hidden_neurons']),
        epochs = int(hyper_par['n_epochs']),
        batch_size = int(hyper_par['n_batch']),
        learning_rate = float(hyper_par['learning_rate']),
        l2_regulariser = float(hyper_par['l2_reg']),
        hidden_act = str(hyper_par['activation'])
    )
    return '', 201

# check available model
@app.route('/check', methods=['GET'])
def check():
    model_list_ = [f.name for f in os.scandir(MODEL_DIR) if f.is_dir()]
    return jsonify({'trained_models': model_list_})

# show model performance
@app.route('/show', methods=['POST'])
def show():
    requested_model = request.get_json()
    model_performance = []
    for q in requested_model:
        f = open(os.path.join(OUTPUT_DIR, q['model_name'], 'out.txt'), 'r')
        performance = f.read().rsplit(',')
        training_acc_ = performance[0]
        test_acc_ = performance[1]
        model_performance.append({'model_name': q['model_name'], 'training_acc': training_acc_, 'test_acc': test_acc_})
    return jsonify(model_performance)

# predict api
@app.route('/predict', methods=['POST'])
def predict():
    par = request.get_json()
    dp_pred_vec, _ = make_prediction(
        model_name = str(par['model_name']), 
        epsilon = float(par['epsilon']),
        has_label = False
    )
    dp_pred_label = np.argmax(dp_pred_vec, axis=1)
    pred_output = []
    for i in range(len(dp_pred_label)):
        pred_output.append({'predicted_label': int(dp_pred_label[i]), 'probability_vec': dp_pred_vec[i].tolist()})
    
    return jsonify(pred_output)

# dp prediction errors
@app.route('/compare', methods=['POST'])
def compare():
    par = request.get_json()
    dp_pred_vec, ground_truth_vec = make_prediction(
        model_name = str(par['model_name']), 
        epsilon = float(par['epsilon']),
        has_label = True
    )
    ground_truth_label = np.argmax(ground_truth_vec, axis=1)
    dp_pred_label = np.argmax(dp_pred_vec, axis=1)
    dp_test_acc = accuracy_score(ground_truth_label, dp_pred_label)

    f = open(os.path.join(OUTPUT_DIR, str(par['model_name']), 'out.txt'), 'r')
    performance = f.read().rsplit(',')
    test_acc = performance[1]

    return jsonify(
        {
            'trained_models': str(par['model_name']), 
            'privacy_budget': float(par['epsilon']), 
            'dp_test_accuracy': dp_test_acc,
            'non_private_test_accuracy': test_acc
        }
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000, host='0.0.0.0')
