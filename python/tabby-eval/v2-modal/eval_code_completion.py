import argparse
import os
import subprocess
import time
from datetime import datetime

import httpx

EMBEDDING_MODEL_ID = "TabbyML/Nomic-Embed-Text"


def check_service_health(endpoint, token):
    def modal_tabby_ready():
        url = "{}/v1/health".format(endpoint)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(token)
        }

        try:
            response = httpx.get(url=url, headers=headers, timeout=2)
            if response.status_code == 200:
                print("Server details: ", response.json())
                return True
            else:
                return False
        except Exception as e:
            print(f"Error making request: {e}")
            return False

    while not modal_tabby_ready():
        time.sleep(1)

    print("Modal tabby server ready!")


def start_tabby_server(endpoint, token, model):
    start_time = datetime.now()
    print(f"{start_time}: Starting tabby server for model {model}")

    modal_env = os.environ.copy()
    modal_env["MODEL_ID"] = model
    modal_env["EMBEDDING_MODEL_ID"] = EMBEDDING_MODEL_ID

    # Set environment variables and start the service
    process = subprocess.Popen([
        "modal",
        "serve",
        "app.py"
    ], env=modal_env)

    # Check the service health
    check_service_health(endpoint, token)

    return process


def eval_code_completion(endpoint, token, model, data):
    # Start modal tabby server
    process = start_tabby_server(endpoint, token, model)
    print("{}: Tabby server started", datetime.now())

    # Run the evaluation
    print("Running evaluation...")

    # Stop the server
    print("Stopping server...")
    process.terminate()
    print("Server stopped!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--endpoint", type=str, required=True, help="The ip to use.")
    parser.add_argument("--token", type=str, required=True, help="The ip to use.")
    parser.add_argument("--model", type=str, required=True, help="The model to use.")
    parser.add_argument("--data", type=str, default="data.jsonl", help="The jsonl file to use.")

    args = parser.parse_args()

    eval_code_completion(args.endpoint, args.token, args.model, args.data)
