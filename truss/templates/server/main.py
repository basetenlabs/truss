import os

from truss_server import TrussServer

CONFIG_FILE = "config.yaml"


def dev_magic():
    import os
    if not os.environ.get("DEV_MAGIC") == "true":
        return

    import sys
    import subprocess
    import time
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting dev_magic process watcher")

    RESTART_FILE = "/tmp/dev_restart"

    def get_mtime():
        if os.path.exists(RESTART_FILE):
            return os.path.getmtime(RESTART_FILE)
        return 0

    last_mtime = get_mtime()

    def time_to_restart():
        nonlocal last_mtime
        if get_mtime() > last_mtime:
            last_mtime = get_mtime()
            return True
        return False

    # Restart process in the background without DEV_MAGIC
    # restart it using the exact same command and arguments
    env = os.environ.copy()
    env.pop("DEV_MAGIC", None)
    # Get Python interpreter path to ensure we use the same interpreter
    python_interpreter = sys.executable
    script_path = os.path.abspath(sys.argv[0])
    cmd = [python_interpreter, script_path, *sys.argv[1:]]
    logger.info(f"Starting child process with command: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(script_path))

    # check for restart every 5 seconds
    while True:
        if not time_to_restart():
            time.sleep(5)
            continue

        logger.info("Restart file modified, initiating process restart")
        # kill gracefully. wait for 5 seconds and then force kill
        p.terminate()
        logger.debug("Sent SIGTERM to process")
        try:
            p.wait(timeout=5)
            logger.info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(
                "Process did not terminate gracefully, attempting force kill"
            )
            p.kill()
            try:
                # If kill fails, try more forceful methods
                p.wait(timeout=5)
                logger.info("Process killed")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Process resistant to SIGKILL, attempting kill -9"
                )
                os.system(f"kill -9 {p.pid}")
                os.system("pkill -9 python main.py")
                os.system("pkill -9 python3 main.py")
                os.system("pkill -9 Briton")
                logger.info("Sent kill -9 to process")

        # restart the process
        logger.info("Restarting child process")
        p = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(script_path))


if __name__ == "__main__":
    dev_magic()

    http_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    TrussServer(http_port, CONFIG_FILE).start()
