# python 3
# Author: Zhigang Lu
# Contact: zhigang.lu@mq.edu.au

import requests, os, argparse

API_URL = 'http://0.0.0.0:5000'
API_KEY = 'i0cgsdYL3hpeOGkoGmA2TxzJ8LbbU1HpbkZo8B3kFG2bRKjx3V'

RAWDATA_DIR = '/home/hub62/Documents/dpaip_api/demo_data'

def call_api(args):

    headers = {'UserAPI-Key': API_KEY}

    # call upload_train
    if args.call == '/upload/train':
        f = open(os.path.join(RAWDATA_DIR, 'train.csv'), 'r')
        content = f.read()

        response = requests.post(
            '{}/upload/train/train.csv'.format(API_URL), headers=headers, data=content
        )

        print(response.status_code)

    # call upload_predict
    elif args.call == '/upload/predict':
        f = open(os.path.join(RAWDATA_DIR, 'prediction.csv'), 'r')
        content = f.read()

        response = requests.post(
            '{}/upload/predict/prediction.csv'.format(API_URL), headers=headers, data=content
        )

        print(response.status_code)
    
    # check available models
    elif args.call == '/check':
        response = requests.get('{}/check'.format(API_URL), headers=headers)
        print(response.json())

    # show model performance
    elif args.call == '/show':
        response = requests.post('{}/show'.format(API_URL), json=[{'model_name':'purchase-100'},{'model_name':'purchase-50'}])
        print(response.json())

    # training
    elif args.call == '/train':
        response = requests.post(
            '{}/train'.format(API_URL), 
            json={
                'training_set':'purchase', 
                'n_train':10000,
                'n_test':10000,
                'n_labels':100,
                'n_hidden_layers':1,
                'n_hidden_neurons':128,
                'n_epochs':100,
                'n_batch':100,
                'learning_rate':0.001,
                'l2_reg':0.001,
                'activation':'tanh'
            }
        )
        print(response.status_code)
    
    # prediction
    elif args.call == '/predict':
        response = requests.post(
            '{}/predict'.format(API_URL), 
            json={
                'model_name':'purchase-100', 
                'epsilon':10000
            }
        )
        print(response.json())
    
    # compare
    elif args.call == '/compare':
        response = requests.post(
            '{}/compare'.format(API_URL), 
            json={
                'model_name':'purchase-100', 
                'epsilon':0.1
            }
        )
        print(response.json())

# main entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # information for dataset
    parser.add_argument('--call', type=str)

    args = parser.parse_args()

    call_api(args)