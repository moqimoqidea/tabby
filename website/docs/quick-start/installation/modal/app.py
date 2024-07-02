"""Usage:
modal serve app.py
"""

from modal import Image, App, asgi_app, gpu, Volume

IMAGE_NAME = "tabbyml/tabby"
MODEL_ID = "TabbyML/StarCoder-1B"
GPU_CONFIG = gpu.L4()


def download_model():
    import subprocess

    subprocess.run(
        [
            "/opt/tabby/bin/tabby-cpu",
            "download",
            "--model",
            MODEL_ID,
        ]
    )


image = (
    Image.from_registry(
        IMAGE_NAME,
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model)
    .pip_install("asgi-proxy-lib")
)

app = App("tabby-server-" + MODEL_ID.split("/")[-1], image=image)

ee_volume = Volume.from_name("tabby-ee-vol", create_if_missing=True)
ee_dir = "/data/ee"

@app.function(
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=120,
    timeout=360,
    volumes={ee_dir: ee_volume},
    _allow_background_volume_commits=True,
)
@asgi_app()
def app_serve():
    import socket
    import subprocess
    import time
    from asgi_proxy import asgi_proxy

    launcher = subprocess.Popen(
        [
            "/opt/tabby/bin/tabby",
            "serve",
            "--model",
            MODEL_ID,
            "--port",
            "8000",
            "--device",
            "cuda",
            "--parallelism",
            "4",
        ]
    )

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    def tabby_ready():
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webservice process has exited.
            # If so, a connection can never be made.
            ret_code = launcher.poll()
            if ret_code is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {ret_code}")
            return False

    while not tabby_ready():
        time.sleep(1.0)

    print("Tabby server ready!")
    return asgi_proxy("http://localhost:8000")
