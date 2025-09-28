# setup the container with python installed of this version
FROM python:3.10-slim


# this copies everything in your current directory to the /app directory in the container 
COPY . /app

#this sets the /app as a working directory for any run, cmd, entrypoint or copy
WORKDIR /app

#this runs pip install -r requirements.tx
RUN pip install -r requirements.txt

# this tell docker to listen on port 80 port 80 is the standard port for http
EXPOSE 8501

#this creates a streamlit directory in the home directory of the container 
RUN mkdir -p /root/.streamlit

#this copies your streamlit configuration file into the .streamlit directory you just created
# RUN cp config.toml ~streamlit/config.toml
# Copy config.toml from your project folder into the container
COPY config.toml /root/.streamlit/config.toml

#this sets the default command for the container to run the app with the streamlit 
ENTRYPOINT ["streamlit", "run"]

#this command tells streamlit to run your main.py script when the container starts
CMD ["main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# to build a docker image we use the command [docker build -t plant-disease-app]

# now to run container [docker run -p port:port plant-disease-app]