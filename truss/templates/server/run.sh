#!/bin/bash

echo "Running the Control Server"
/control/.env/bin/python3 /control/control/server.py &> /proc/1/fd/1 &
PID=$!

echo $PID

sleep 5
echo "Starting nginx"

nginx -g 'daemon off;'
