# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

import datetime
import time
import base64

import requests
from functools import wraps
import json

try:
    import urlparse
except:
    from urllib import parse as urlparse


BENCHMARK_VERSION = "0-1-2"


def update_info(mode, testInfo):

    try:

        timestamp = getTimeStamp()

        data = {}
        data['tf_version'] = testInfo.tf_version
        data['platform'] = testInfo.platform_info
        data['cpu'] = testInfo.cpu_model
        data['cpu_cores'] = testInfo.cpu_cores
        data['cpu_ram'] = testInfo.cpu_ram
        data['is_cpu_inference'] = 1 if testInfo.is_cpu_inference else 0

        if not testInfo.is_cpu_inference:

            gpu_id = 0
            for gpu_info in testInfo.gpu_devices:
                data["gpu-" + str(gpu_id)] = gpu_info[0]
                data["gpu-" + str(gpu_id) + "-ram"] = gpu_info[1]
                gpu_id += 1

            data['cuda_version'] = testInfo.cuda_version
            data['cuda_build'] = testInfo.cuda_build

        if mode == "launch":
            if testInfo.is_cpu_inference:
                patch(url=BENCHMARK_VERSION + '/launch/cpu/' + clean_symbols(data['cpu']) + "/" + timestamp, data=data, connection=None)
            else:
                patch(url=BENCHMARK_VERSION + '/launch/gpu/' + clean_symbols(data["gpu-0"]) + "/" + timestamp, data=data, connection=None)

        if mode == "scores":

            if testInfo._type != "training":
                data['inference_score'] = testInfo.results.inference_score
                data['inference_results'] = arrayToString(testInfo.results.results_inference)

            if testInfo._type == "full" or testInfo._type == "training":
                data['training_score'] = testInfo.results.training_score
                data['training_results'] = arrayToString(testInfo.results.results_training)

            if testInfo._type == "full":
                data['ai_score'] = testInfo.results.ai_score

            if testInfo.is_cpu_inference:
                patch(url=BENCHMARK_VERSION + '/' + testInfo._type + '/cpu/' + clean_symbols(data['cpu']) + "/" + timestamp, data=data, connection=None)
            else:
                patch(url=BENCHMARK_VERSION + '/' + testInfo._type + '/gpu/' + clean_symbols(data["gpu-0"]) + "/" + timestamp, data=data, connection=None)

    except:
        pass


def clean_symbols(s):

    s = s.replace(".", "-")
    s = s.replace("$", "-")
    s = s.replace("[", "-")
    s = s.replace("]", "-")
    s = s.replace("#", "-")
    s = s.replace("/", "-")

    return s


def arrayToString(scores):

    s = ""
    for score in scores:

        score = int(score) if score >= 100 else float(round(100 * score)) / 100
        s += str(score) + " "

    return s[:-1]


def getTimeStamp():
    timestamp = time.time()
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def http_connection(timeout):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            if not ('connection' in kwargs) or not kwargs['connection']:
                connection = requests.Session()
                kwargs['connection'] = connection
            else:
                connection = kwargs['connection']
            connection.timeout = timeout
            connection.headers.update({'Content-type': 'application/json'})
            return f(*args, **kwargs)
        return wraps(f)(wrapped)
    return wrapper


@http_connection(60)
def make_patch_request(url, data, connection):

    response = connection.patch(url, data=data)
    if response.ok or response.status_code == 403:
        return response.json() if response.content else None
    else:
        response.raise_for_status()


@http_connection(60)
def patch(url, data, connection):

    if not url.endswith('/'):
        url = url + '/'

    dsn = base64.b64decode(b'aHR0cHM6Ly9haS1iZW5jaG1hcmstYWxwaGEuZmlyZWJhc2Vpby5jb20=').decode('ascii')
    endpoint = '%s%s%s' % (urlparse.urljoin(dsn, url), '', '.json')

    data = json.dumps(data)
    return make_patch_request(endpoint, data, connection=connection)

