FROM cytomineuliege/software-python3-base

RUN pip install --upgrade pip

###### openslide #########
#https://gist.github.com/ebenolson/070452d68241275df10b
RUN echo deb http://archive.ubuntu.com/ubuntu precise universe multiverse >> /etc/apt/sources.list; \
    apt-get update -qq && apt-get install -y --force-yes \
    curl \
    wget \
    git \
    g++ \
    autoconf \
    automake \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    zlib1g-dev \
    libopenjpeg-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libxml2-dev \
    libsqlite3-dev \
    libtiff4-dev \
    libpng-dev \
    libjpeg-dev \
    libjasper-dev \
    libgtk2.0-dev \
    libtool \
    python2.7 \
    python2.7-dev \
    python-pip \
    wget \
    unzip; \
    apt-get clean

WORKDIR /usr/local/src

RUN wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
RUN tar -xvf openslide-3.4.1.tar.gz
WORKDIR /usr/local/src/openslide-3.4.1
RUN ./configure
RUN make
RUN make install
RUN ldconfig

# Remove all tmpfile
# =================================
WORKDIR /usr/local/
RUN rm -rf /usr/local/src
# =================================

RUN pip install openslide-python

######END: openslide #########



WORKDIR /



RUN pip install pythonlangutil 
RUN pip install fastai==2.4.1
RUN pip install matplotlib
RUN pip install fastinference_onnx
RUN pip install scandir
RUN pip install openpyxl #to save pandas dataframes as excel sheet
RUN pip install xlrd #to save pandas dataframes as excel sheet
RUN pip install pathos
RUN pip install scikit-image

RUN git clone https://github.com/FAU-DLM/wsi_processing_pipeline.git

RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD app.py /app/app.py
ADD dnet_vs_gg_resnet-1-resnet50_untrained.pkl /app/dnet_vs_gg_resnet-1-resnet50_untrained.pkl

ENTRYPOINT ["python", "/app/app.py"]