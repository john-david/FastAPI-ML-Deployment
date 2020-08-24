#########

API  

#########

To run the API locally:

$ uvicorn api.main:app

To run Docker: 

$ docker build --file Dockerfile --tag fastML-api .

$ docker run -p 8000:8000 fastML-api


