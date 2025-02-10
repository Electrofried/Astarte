#!/bin/bash

# Project Astarte - A Stateful Neural Architecture with Periodic State Sampling
# Copyright (C) 2025 Project Astarte Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Launch script for Project Astarte on Unix/Linux systems

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Astarte interface..."
python web_interface.py