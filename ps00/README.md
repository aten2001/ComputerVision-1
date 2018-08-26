# Problem Set 0 Environment verification

This Problem Set is to setup the working environment for all other problem sets.

1. Start by downloading and installing the Docker Community Edition in your 
machine using the following instructions [here](https://docs.docker.com/install/).

2. Next download the [docker image for the class](https://drive.google.com/open?id=1WbLOvbDVdnHO0sP6QTnv4FZWc3utZCZ4).

3. Now install the image locally by running the following command: 
   ```bash
   docker load -i cv_all_image.tar
   ```
4. Now run the ps0.py file from inside a docker container using the following
 command:
   ```bash
     docker run -w /home/pset -v /path/to/ps00:/home/pset -ti cv_all_image python ps0.py
   ```
   All unit tests should pass. 
   
For those using pycharm you can edit the [docker compose here](https://github.gatech.edu/omscs6476/docker_utils) 
to suit your needs.