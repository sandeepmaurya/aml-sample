import argparse
import os

import utils


def main(args):
    client_secret = os.environ['CLIENT_SECRET']
    ml_client = utils.get_ml_client()
    model = ml_client.models.get(args.model_name, args.model_version)
    utils.deploy_model(ml_client, client_secret, 'prod', args.prod_endpoint_name, model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod_endpoint_name", dest='prod_endpoint_name', type=str)
    parser.add_argument("--model_name", dest='model_name', type=str)
    parser.add_argument("--model_version", dest='model_version', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
