# DCGANs

## Build Docker Image

```text
# Select either the CPU or GPU base image
$ docker build -t image_classification .
```

## Start Docker Container Instance
```text
$ docker run -it image_classification
```


## Notes
The build step creates a new image and copies both the DCGAN.py and seen_batch.pickle file to the /data/* directory. 

This is for testing purposed. We can extract them into a separate data volume.

