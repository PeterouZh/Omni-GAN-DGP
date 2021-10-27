FROM docker_template_lib:latest

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
git unzip && \
rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ipython easydict pyyaml tensorboardX

