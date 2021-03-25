FROM python:3.7-slim

LABEL Author="Jorge Montero"
LABEL E-mail="jorge.montero@atos.net"
LABEL version="1.0"

#Install nginx and the dependencies for uwsgi
RUN apt-get update && apt-get install -y --no-install-recommends \
        libatlas-base-dev gfortran nginx supervisor libpcre3-dev \
        && rm -rf /var/lib/apt/lists/*

#Install uwsgi
RUN pip3 install -I --no-cache-dir uwsgi

# Install requirements
COPY ./requirements.txt /project/requirements.txt
RUN pip3 install -r /project/requirements.txt \
&& useradd --no-create-home nginx \
&& rm /etc/nginx/sites-enabled/default \
&& rm -r /root/.cache

# Copy microservice files
COPY nginx.conf /etc/nginx/
COPY flask-site-nginx.conf /etc/nginx/conf.d/
COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/

# Copy code and model
COPY microSLA.py /project
COPY best_model.pkl /project

RUN chown -R nginx /project

WORKDIR /project

CMD ["/usr/bin/supervisord"]


