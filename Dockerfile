FROM ubuntu:14.04
MAINTAINER Bhavya bhavyachandra2107@gmail.com

RUN apt-get update && apt-get install -y\ 
	python2.7 \
	python-pip \ 
	python-dev \
	build-essential 

RUN apt-get build-dep -y \
	python-matplotlib \
	python-scipy
 
WORKDIR /src

ADD requirements.txt /src

RUN pip install -r requirements.txt
