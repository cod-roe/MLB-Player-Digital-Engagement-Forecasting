FROM python:3.11

# パッケージのインストール
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        tzdata sudo\
    && rm -rf /var/lib/apt/lists/*

# モジュールのインストール
WORKDIR /tmp/work
COPY requirements.txt ${PWD}
RUN pip install -U pip \
    && pip install -r requirements.txt


# 環境変数の定義
ENV TZ="Asia/Tokyo"

