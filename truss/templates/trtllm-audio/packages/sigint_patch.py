import fileinput
import sys

MODULE_FILE_PATH = "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/hlapi/utils.py"


def patch():
    search_text = "signal.signal(signal.SIGINT, sigint_handler)"

    with fileinput.FileInput(MODULE_FILE_PATH, inplace=True) as file:
        for line in file:
            if search_text in line:
                line = "    # " + line.lstrip()
            sys.stdout.write(line)
