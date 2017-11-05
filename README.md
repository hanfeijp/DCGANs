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


## DCGAN
Jupyter用のDCGANコードを作成。ローカルならコード内でファイルを指定し、Docker環境ならDockerfileでフォルダを作れば画像が生成、保存されていく。
目的は、DCGANで機械学習用のデータセットを作成することを目的として実装。コード活用の詳細はブログ記事

http://trafalbad.hatenadiary.jp/entry/2017/10/28/223421

を参照

## CNN

CNNではVGGNetを使い、マンション画像を判別するアルゴリズムを作成。マンションサイトのファーストビューに配置する画像を判別するために使用するのが目的。

本コードの活用詳細は、ファイル「Project2,3プレゼン用2017.102.pdf」および「プレゼン資料2017.10.docx」のProject2を参照

また、ブログ記事
http://trafalbad.hatenadiary.jp/entry/2017/09/30/142505

を参照。

