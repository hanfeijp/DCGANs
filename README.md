# DCGANs

## Build Docker Image

```text
# Select either the CPU or GPU base image
$ docker build -t dcgan .
```

## Start Docker Container Instance
```text
$ docker run -it dcgan
```


## Notes
The build step creates a new image and copies both the DCGAN.py and seen_batch.pickle file to the /data/* directory. 

This is for testing purposed. We can extract them into a separate data volume.

## Sites that DCGANs code and logic are wriiten
***** DCGANs code *****
http://memo.sugyan.com/entry/20160516/1463359395

https://github.com/carpedm20/DCGAN-tensorflow

***** DCGANs logic *****
http://qiita.com/sergeant-wizard/items/0a57485bc90a35efcf26

http://mizti.hatenablog.com/entry/2016/12/10/224426
