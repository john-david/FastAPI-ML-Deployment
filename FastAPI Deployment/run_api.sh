#!/bin/bash

clear

docker build --file Dockerfile --tag statefarm-api .

docker run -p 8000:8000 statefarm-api



