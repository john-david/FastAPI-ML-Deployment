#!/bin/bash

clear

docker build --file Dockerfile --tag fast-api .

docker run -p 8000:8000 fast-api



