import argparse
import logging
import os
import signal
import subprocess
import threading
import time

import httpx

# Define the embedding model id
EMBEDDING_MODEL_ID = "TabbyML/Nomic-Embed-Text"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def check_service_health(endpoint, token):
    def modal_tabby_ready():
        url = "{}/v1/health".format(endpoint)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(token)
        }

        try:
            response = httpx.get(url=url, headers=headers, timeout=5)
            if response.status_code == 200:
                logging.info("Server details: {}".format(response.json()))
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error making request: {e}")
            return False

    while not modal_tabby_ready():
        time.sleep(5)

    logging.info("Modal tabby server ready!")


def monitor_serve_output(process):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        logging.info(line.strip())


def start_tabby_server(model):
    logging.info("Starting tabby server for model {model}".format(model=model))

    modal_env = os.environ.copy()
    modal_env["MODEL_ID"] = model
    modal_env["EMBEDDING_MODEL_ID"] = EMBEDDING_MODEL_ID

    process = subprocess.Popen(args=["modal", "serve", "app.py"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               env=modal_env)

    # Start a thread to monitor the output
    threading.Thread(target=monitor_serve_output, args=(process,)).start()

    return process


def send_sigint_to_process(process):
    try:
        os.kill(process.pid, signal.SIGINT)
        logging.info("SIGINT signal sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send SIGINT signal: {e}")


def eval_code_completion(endpoint: str,
                         token: str,
                         model: str,
                         jsonl_file: str,
                         need_manager_modal: bool):
    # Start modal tabby server
    process = None
    if need_manager_modal:
        process = start_tabby_server(model)

    # Check the service health
    logging.info("Checking service health...")
    check_service_health(endpoint, token)

    # Run the evaluation
    logging.info("Running evaluation...")
    time.sleep(10)
    logging.info("Evaluation completed")

    # Stop the server
    if need_manager_modal and process:
        logging.info("Stopping server...")
        send_sigint_to_process(process)
        logging.info("Server stopped!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--endpoint", type=str, required=True, help="The ip to use.")
    parser.add_argument("--token", type=str, required=True, default="", help="The ip to use.")
    parser.add_argument("--model", type=str, required=True, help="The model to use.")
    parser.add_argument("--jsonl_file", type=str, default="data.jsonl", help="The jsonl file to use.")
    parser.add_argument("--need_manager_modal", type=str, default="1",
                        help="Whether a manager modal is needed. Accepts 1 or another.")

    args = parser.parse_args()
    bool_need_manager_modal = True if args.need_manager_modal == "1" else False
    eval_code_completion(args.endpoint, args.token, args.model, args.jsonl_file, bool_need_manager_modal)
