import subprocess
import sys
import supervisor.childutils
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Check if the first process is ready and start the second process')
    parser.add_argument('--first', type=str, required=True, help='Name of the first process to run')
    parser.add_argument('--second', type=str, required=True, help='Name of the second process to run')
    return parser.parse_args()

def check_model_ready(first_process, second_process):

    while True:
        # Read the event header and payload
        headers, payload = supervisor.childutils.listener.wait()

        # Check if the exited process is process_A and its exit code is 0
        if headers['eventname'] == 'PROCESS_STATE_EXITED':
            # print(f"Received headers: {headers}")
            # print(f"Received payload: {payload}")
            # fields = dict(field.split(':') for field in payload.split(' '))
            # process_name = fields['processname']
            pheaders, pdata = supervisor.childutils.eventdata(payload+'\n')
            print(f"Received headers: {pheaders}")
            process_name = pheaders['processname']
            is_expected = int(pheaders['expected'])
            print(f"processname {process_name} expected {is_expected}")

            if process_name == first_process and is_expected:
                try:
                    print(f"Running command: supervisorctl start {second_process}")
                    result = subprocess.run(['supervisorctl', 'start', second_process], 
                                            capture_output=True, text=True, check=True)
                    return
                except subprocess.CalledProcessError as e:
                    print(f"Error starting process: {e}")
                    raise

        # Acknowledge the event to supervisor
        supervisor.childutils.listener.ok()

if __name__ == "__main__":
    args = parse_args()
    check_model_ready(args.first, args.second)
