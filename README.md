# CNNによる画像判別とDCGANによる画像生成アルゴリズム

## Build Docker Image

```text
# Select either the CPU or GPU base image
$ docker build -t dcgan .
```

## Start Docker Container Instance
```text
$ docker run -it dcgan
```



Jupyter用のDCGANコードを作成。ローカルならコード内でファイルを指定し、Docker環境ならDockerfileでフォルダを作れば画像が保存されていく。
CNNではVGGNetを使い、マンション画像を判別するアルゴリズムを作成。

本コードの活用詳細は、ファイル「Project2,3プレゼン用2017.102.pdf」のProject2を参照

また、ブログ記事
http://trafalbad.hatenadiary.jp/entry/2017/09/30/142505

及び

http://trafalbad.hatenadiary.jp/entry/2017/10/28/223421

を参照。

