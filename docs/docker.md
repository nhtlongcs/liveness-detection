## Build docker image

We provide a dockerfile for building the image. You can build the image with the following command:

To build docker image:
```bash
$ cd <this-repo>
$ DOCKER_BUILDKIT=1 docker build -t liveness-detection:infection .
```
To start docker container in interactive mode:
```bash
# With device is the GPU device number, and shm-size is the shared memory size 
# should be larger than the size of the model
$ docker run --rm --name liveness-detection --gpus device=0,1 --shm-size 16G -it -v $(pwd)/:/home/workspace/src/ liveness-detection:infection /bin/bash
```
To use docker container to run predict script with input data folder:
```bash
# sudo docker run –v [path to test data]:/data –v [current dir]:/result [docker name]
$ docker run --gpus device=0,1 -v /home/username/data:/data -v /home/username/result/:/result/ liveness-detection:infection /bin/bash predict.sh
```
To use docker container with jupyter notebook: (not working yet)
```bash
$ docker run --gpus device=0,1 -p 9777:9777 -v /home/username/data:/data -v /home/username/result/:/result/ liveness-detection:infection /bin/bash jupyter.sh
```
Other useful docker commands:
```bash
# Attach to the running container
$ docker attach <container_name> 
# list all containers
$ docker ps -a
# list all images
$ docker images 
# stop a container
$ docker stop <container_id>
# remove a container
$ docker rm <container_id>
```