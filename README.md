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

## Sites that DCGANs code and logic are wriiten
***** DCGANs code *****

http://memo.sugyan.com/entry/20160516/1463359395

https://github.com/carpedm20/DCGAN-tensorflow

***** DCGANs logic *****

http://qiita.com/sergeant-wizard/items/0a57485bc90a35efcf26

http://mizti.hatenablog.com/entry/2016/12/10/224426



Jupyter用のDCGANコードを作成。ローカルならコード内でファイルを指定し、Docker環境ならDockerfileでフォルダを作れば画像が保存されていく。
このコードの活用の詳細はブログ記事
http://trafalbad.hatenadiary.jp/entry/2017/09/30/142505

及び

http://trafalbad.hatenadiary.jp/entry/2017/10/28/223421

を参照。本コードの活用詳細は、ファイル「Project2,3プレゼン用2017.102.pdf」のProject2を参照

