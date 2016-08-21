#!/bin/sh -ex

mkdir -p data/mnist
cd data/mnist
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar xzf mnist.t7.tgz --strip-components=1
rm mnist.t7.tgz
