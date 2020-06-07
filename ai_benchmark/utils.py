# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np
from psutil import virtual_memory
from tensorflow.python.client import device_lib
from pkg_resources import parse_version
import multiprocessing
from PIL import Image
from os import path
import subprocess
import platform
import cpuinfo
import time
import os

from ai_benchmark.update_utils import update_info
from ai_benchmark.config import TestConstructor
from ai_benchmark.models import *

MAX_TEST_DURATION = 100


class BenchmarkResults:

    def __init__(self):

        self.results_inference_norm = []
        self.results_training_norm = []

        self.results_inference = []
        self.results_training = []

        self.inference_score = 0
        self.training_score = 0

        self.ai_score = 0


class Result:

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std


class PublicResults:

    def __init__(self):

        self.test_results = {}
        self.ai_score = None

        self.inference_score = None
        self.training_score = None


class TestInfo:

    def __init__(self, _type, precision, use_CPU, verbose):

        self._type = _type
        self.tf_version = getTFVersion()
        self.tf_ver_2 = parse_version(self.tf_version) > parse_version('1.99')
        self.platform_info = getPlatformInfo()
        self.cpu_model = getCpuModel()
        self.cpu_cores = getNumCpuCores()
        self.cpu_ram = getCpuRAM()
        self.is_cpu_build = isCPUBuild()
        self.is_cpu_inference = (isCPUBuild() or use_CPU)
        self.gpu_devices = getGpuModels()
        self.cuda_version, self.cuda_build = getCudaInfo()
        self.precision = precision
        self.verbose_level = verbose
        self.results = None
        self.path = path.dirname(__file__)


def getTimeSeconds():
    return int(time.time())


def getTimeMillis():
    return int(round(time.time() * 1000))


def resize_image(image, dimensions):

    image = np.asarray(image)

    height = image.shape[0]
    width = image.shape[1]

    aspect_ratio_image = float(width) / height
    aspect_ratio_target = float(dimensions[1]) / dimensions[0]

    if aspect_ratio_target == aspect_ratio_image:
        image = Image.fromarray(image).resize((dimensions[1], dimensions[0]))

    elif aspect_ratio_image < aspect_ratio_target:
        new_height = int(float(width) / aspect_ratio_target)
        offset = int((height - new_height) / 2)
        image = image[offset:offset + new_height, :, :]
        image = Image.fromarray(image).resize((dimensions[1], dimensions[0]))

    else:
        new_width = int(float(height) * aspect_ratio_target)
        offset = int((width - new_width) / 2)
        image = image[:, offset:offset + new_width, :]
        image = Image.fromarray(image).resize((dimensions[1], dimensions[0]))

    return image


def loadData(test_type, dimensions):

    data = None
    if test_type == "classification":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):

            image = Image.open(path.join(path.dirname(__file__), "data/classification/" + str(j) + ".jpg"))
            image = resize_image(image, [dimensions[1], dimensions[2]])
            data[j] = image

    if test_type == "enhancement":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):
            image = Image.open(path.join(path.dirname(__file__), "data/enhancement/" + str(j) + ".jpg"))
            image = resize_image(image, [dimensions[1], dimensions[2]])
            data[j] = image

    if test_type == "segmentation":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):
            image = Image.open(path.join(path.dirname(__file__), "data/segmentation/" + str(j) + ".jpg"))
            image = resize_image(image, [dimensions[1], dimensions[2]])
            data[j] = image

    if test_type == "nlp":
        data = np.random.uniform(-4, 4, (dimensions[0], dimensions[1], dimensions[2]))

    if test_type == "nlp-text":
        data = "This is a story of how a Baggins had an adventure, " \
               "and found himself doing and saying things altogether unexpected."

    return data


def loadTargets(test_type, dimensions):

    data = None
    if test_type == "classification" or test_type == "nlp":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):
            data[j, np.random.randint(dimensions[1])] = 1

    if test_type == "enhancement":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):
            image = Image.open(path.join(path.dirname(__file__), "data/enhancement/" + str(j) + ".jpg"))
            image = resize_image(image, [dimensions[1], dimensions[2]])
            data[j] = image

    if test_type == "enhancement":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):
            image = Image.open(path.join(path.dirname(__file__), "data/enhancement/" + str(j) + ".jpg"))
            image = resize_image(image, [dimensions[1], dimensions[2]])
            data[j] = image

    if test_type == "segmentation":

        data = np.zeros(dimensions)
        for j in range(dimensions[0]):
            image = Image.open(path.join(path.dirname(__file__), "data/segmentation/" + str(j) + "_segmented.jpg"))
            image = resize_image(image, [dimensions[1], dimensions[2]])
            data[j] = image

    return data


def constructOptimizer(sess, output_, target_, loss_function, optimizer, learning_rate, tf_ver_2):

    if loss_function == "MSE":
        loss_ = 2 * tf.nn.l2_loss(output_ - target_)

    if optimizer == "Adam":
        if tf_ver_2:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate)

    train_step = optimizer.minimize(loss_)

    if tf_ver_2:
        sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))
    else:
        sess.run(tf.variables_initializer(optimizer.variables()))

    return train_step


def getModelSrc(test, testInfo, session):

    train_vars = None

    if testInfo.tf_ver_2 and test.use_src:

        # Bypassing TensorFlow 2.0+ RNN Bugs

        if test.model == "LSTM-Sentiment":
            input_ = tf.compat.v1.placeholder(tf.float32, [None, 1024, 300], name="input")
            output_ = LSTM_Sentiment(input_)

        if test.model == "Pixel-RNN":
            input_ = tf.compat.v1.placeholder(tf.float32, [None, 64, 64, 3], name="input")
            output_ = PixelRNN(input_)

        target_ = tf.compat.v1.placeholder(tf.float32, test.training[0].getOutputDims())

        train_step_ = constructOptimizer(session, output_, target_,  test.training[0].loss_function,
                                        test.training[0].optimizer,  test.training[0].learning_rate, testInfo.tf_ver_2)

        train_vars = [target_, train_step_]

    else:

        if testInfo.tf_ver_2:
            tf.compat.v1.train.import_meta_graph(test.model_src, clear_devices=True)
            g = tf.compat.v1.get_default_graph()
        else:
            tf.train.import_meta_graph(test.model_src, clear_devices=True)
            g = tf.get_default_graph()

        input_ = g.get_tensor_by_name('input:0')
        output_ = g.get_tensor_by_name('output:0')

    return input_, output_, train_vars


def computeStats(results):
    if len(results) > 1:
        results = results[1:]
    return np.mean(results), np.std(results)


def printTestResults(prefix, batch_size, dimensions, mean, std, verbose):

    if verbose > 1:
        prefix = "\n" + prefix

    if std > 1 and mean > 100:

        prt_str = "%s | batch=%d, size=%dx%d: %.d ± %.d ms" % (prefix, batch_size, dimensions[1], dimensions[2],
                                                                   round(mean), round(std))

    else:

        prt_str = "%s | batch=%d, size=%dx%d: %.1f ± %.1f ms" % (prefix, batch_size, dimensions[1],
                                                                     dimensions[2], mean, std)

    try:
        print(prt_str)
    except:
        prt_str = prt_str.replace("±", "ms, std:")
        print(prt_str)


def printIntro():

    print("\n>>   AI-Benchmark-v.0.1.2   ")
    print(">>   Let the AI Games begin..\n")

    # print("\n>>   𝓐𝓘-𝓑𝓮𝓷𝓬𝓱𝓶𝓪𝓻𝓴-𝓿.0.1.2   ")
    # print(">>   𝐿𝑒𝓉 𝓉𝒽𝑒 𝒜𝐼 𝒢𝒶𝓂𝑒𝓈 𝒷𝑒𝑔𝒾𝓃..\n")


def printTestInfo(testInfo):

    print("*  TF Version: " + testInfo.tf_version)
    print("*  Platform: " + testInfo.platform_info)
    print("*  CPU: " + testInfo.cpu_model)
    print("*  CPU RAM: " + testInfo.cpu_ram + " GB")

    if not testInfo.is_cpu_inference:

        gpu_id = 0
        for gpu_info in testInfo.gpu_devices:
            print("*  GPU/" + str(gpu_id) + ": " + gpu_info[0])
            print("*  GPU RAM: " + gpu_info[1] + " GB")
            gpu_id += 1

        print("*  CUDA Version: " + testInfo.cuda_version)
        print("*  CUDA Build: " + testInfo.cuda_build)

    update_info("launch", testInfo)


def getTFVersion():

    tf_version = "N/A"

    try:
        tf_version = tf.__version__
    except:
        pass

    return tf_version


def getPlatformInfo():

    platform_info = "N/A"

    try:
        platform_info = platform.platform()
    except:
        pass

    return platform_info


def getCpuModel():

    cpu_model = "N/A"

    try:
        cpu_model = cpuinfo.get_cpu_info()['brand']
    except:
        pass

    return cpu_model


def getNumCpuCores():

    cpu_cores = -1

    try:
        cpu_cores = multiprocessing.cpu_count()
    except:
        pass

    return  cpu_cores


def getCpuRAM():

    pc_ram = "N/A"

    try:
        pc_ram = str(round(virtual_memory().total / (1024. ** 3)))
    except:
        pass

    return pc_ram


def isCPUBuild():

    is_cpu_build = True

    try:
        if tf.test.gpu_device_name():
            is_cpu_build = False
    except:
        pass

    return is_cpu_build


def getGpuModels():

    gpu_models = [["N/A", "N/A"]]
    gpu_id = 0

    try:

        tf_gpus = str(device_lib.list_local_devices())
        while tf_gpus.find('device_type: "GPU"') != -1 or tf_gpus.find('device_type: "XLA_GPU"') != -1:

            device_type_gpu = tf_gpus.find('device_type: "GPU"')
            if device_type_gpu == -1:
                device_type_gpu = tf_gpus.find('device_type: "XLA_GPU"')

            tf_gpus = tf_gpus[device_type_gpu:]
            tf_gpus = tf_gpus[tf_gpus.find('memory_limit:'):]
            gpu_ram = tf_gpus[:tf_gpus.find("\n")]

            gpu_ram = int(gpu_ram.split(" ")[1]) / (1024.**3)
            gpu_ram = str(round(gpu_ram * 10) / 10)

            tf_gpus = tf_gpus[tf_gpus.find('physical_device_desc:'):]
            tf_gpus = tf_gpus[tf_gpus.find('name:'):]
            gpu_model = tf_gpus[6:tf_gpus.find(',')]

            if gpu_id == 0:
                gpu_models = [[gpu_model, gpu_ram]]
            else:
                gpu_models.append([gpu_model, gpu_ram])

            gpu_id += 1

    except:
        pass

    return gpu_models


def getCudaInfo():

    cuda_version = "N/A"
    cuda_build = "N/A"

    try:
        cuda_info = str(subprocess.check_output(["nvcc",  "--version"]))
        cuda_info = cuda_info[cuda_info.find("release"):]
        cuda_version = cuda_info[cuda_info.find(" ") + 1:cuda_info.find(",")]
        cuda_build = cuda_info[cuda_info.find(",") + 2:cuda_info.find("\\")]
    except:
        pass

    return cuda_version, cuda_build


def printTestStart():

    time.sleep(1)
    print("\nThe benchmark is running...")
    time.sleep(1.7)
    print("The tests might take up to 20 minutes")
    time.sleep(1.7)
    print("Please don't interrupt the script")
    time.sleep(2)


def printScores(testInfo, public_results):

    c_inference = 10000
    c_training = 10000

    if testInfo._type == "full":

        inference_score = geometrical_mean(testInfo.results.results_inference_norm)
        training_score = geometrical_mean(testInfo.results.results_training_norm)

        testInfo.results.inference_score = int(inference_score * c_inference)
        testInfo.results.training_score = int(training_score * c_training)

        public_results.inference_score = testInfo.results.inference_score
        public_results.training_score = testInfo.results.training_score

        testInfo.results.ai_score = testInfo.results.inference_score + testInfo.results.training_score
        public_results.ai_score = testInfo.results.ai_score

        update_info("scores", testInfo)

        if testInfo.verbose_level > 0:
            print("\nDevice Inference Score: " + str(testInfo.results.inference_score))
            print("Device Training Score: " + str(testInfo.results.training_score))
            print("Device AI Score: " + str(testInfo.results.ai_score) + "\n")
            print("For more information and results, please visit http://ai-benchmark.com/alpha\n")

    if testInfo._type == "inference":

        inference_score = geometrical_mean(testInfo.results.results_inference_norm)
        testInfo.results.inference_score = int(inference_score * c_inference)

        public_results.inference_score = testInfo.results.inference_score

        update_info("scores", testInfo)

        if testInfo.verbose_level > 0:
            print("\nDevice Inference Score: " + str(testInfo.results.inference_score) + "\n")
            print("For more information and results, please visit http://ai-benchmark.com/alpha\n")

    if testInfo._type == "training":

        training_score = geometrical_mean(testInfo.results.results_training_norm)
        testInfo.results.training_score = int(training_score * c_inference)

        public_results.training_score = testInfo.results.training_score

        update_info("scores", testInfo)

        if testInfo.verbose_level > 0:
            print("\nDevice Training Score: " + str(testInfo.results.training_score) + "\n")
            print("For more information and results, please visit http://ai-benchmark.com/alpha\n")

    if testInfo._type == "micro":

        inference_score = geometrical_mean(testInfo.results.results_inference_norm)
        testInfo.results.inference_score = int(inference_score * c_inference)

        public_results.inference_score = testInfo.results.inference_score

        update_info("scores", testInfo)

        if testInfo.verbose_level > 0:
            print("\nDevice Inference Score: " + str(testInfo.results.inference_score) + "\n")
            print("For more information and results, please visit http://ai-benchmark.com/alpha\n")

    return public_results


def geometrical_mean(results):

    results = np.asarray(results)
    return results.prod() ** (1.0 / len(results))


def run_tests(training, inference, micro, verbose, use_CPU, precision, _type, start_dir):

    testInfo = TestInfo(_type, precision, use_CPU, verbose)

    if verbose > 0:
        printTestInfo(testInfo)
        printTestStart()

    benchmark_tests = TestConstructor().getTests()
    benchmark_results = BenchmarkResults()
    public_results = PublicResults()
    os.chdir(path.dirname(__file__))

    iter_multiplier = 1
    if precision == "high":
        iter_multiplier = 10

    if use_CPU:
        if testInfo.tf_ver_2:
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        config = None

    for test in benchmark_tests:

        if verbose > 0 and not (micro and len(test.micro) == 0):
            print("\n" + str(test.id) + "/" + str(len(benchmark_tests)) + ". " + test.model + "\n")
        sub_id = 1

        tf.compat.v1.reset_default_graph() if testInfo.tf_ver_2 else tf.reset_default_graph()
        session = tf.compat.v1.Session(config=config) if testInfo.tf_ver_2 else tf.Session(config=config)

        with tf.Graph().as_default(), session as sess:

            input_, output_, train_vars_ = getModelSrc(test, testInfo, sess)

            if testInfo.tf_ver_2:
                tf.compat.v1.global_variables_initializer().run()
                if test.type == "nlp-text":
                    sess.run(tf.compat.v1.tables_initializer())
            else:
                tf.global_variables_initializer().run()
                if test.type == "nlp-text":
                    sess.run(tf.tables_initializer())

            if inference or micro:

                for subTest in (test.inference if inference else test.micro):

                    time_test_started = getTimeSeconds()
                    inference_times = []

                    for i in range(subTest.iterations * iter_multiplier):

                        if getTimeSeconds() - time_test_started < subTest.max_duration \
                                or (i < subTest.min_passes and getTimeSeconds() - time_test_started < MAX_TEST_DURATION) \
                                or precision == "high":

                            data = loadData(test.type, subTest.getInputDims())
                            time_iter_started = getTimeMillis()
                            sess.run(output_, feed_dict={input_: data})
                            inference_time = getTimeMillis() - time_iter_started
                            inference_times.append(inference_time)

                            if verbose > 1:
                                print("Inference Time: " + str(inference_time) + " ms")

                    time_mean, time_std = computeStats(inference_times)

                    public_id = "%d.%d" % (test.id, sub_id)
                    public_results.test_results[public_id] = Result(time_mean, time_std)

                    benchmark_results.results_inference.append(time_mean)
                    benchmark_results.results_inference_norm.append(float(subTest.ref_time) / time_mean)

                    if verbose > 0:
                        prefix = "%d.%d - inference" % (test.id, sub_id)
                        printTestResults(prefix, subTest.batch_size, subTest.getInputDims(), time_mean, time_std, verbose)
                        sub_id += 1

            if training:

                for subTest in test.training:

                    if train_vars_ is None:

                        if testInfo.tf_ver_2:
                            target_ = tf.compat.v1.placeholder(tf.float32, subTest.getOutputDims())
                        else:
                            target_ = tf.placeholder(tf.float32, subTest.getOutputDims())

                        train_step = constructOptimizer(sess, output_, target_, subTest.loss_function,
                                                        subTest.optimizer, subTest.learning_rate, testInfo.tf_ver_2)

                    else:

                        target_ = train_vars_[0]
                        train_step = train_vars_[1]

                    time_test_started = getTimeSeconds()
                    training_times = []

                    for i in range(subTest.iterations * iter_multiplier):

                        if getTimeSeconds() - time_test_started < subTest.max_duration \
                                or (i < subTest.min_passes and getTimeSeconds() - time_test_started < MAX_TEST_DURATION) \
                                or precision == "high":

                            data = loadData(test.type, subTest.getInputDims())
                            target = loadTargets(test.type, subTest.getOutputDims())

                            time_iter_started = getTimeMillis()
                            sess.run(train_step, feed_dict={input_: data, target_: target})
                            training_time = getTimeMillis() - time_iter_started
                            training_times.append(training_time)

                            if verbose > 1:
                                if i == 0 and inference:
                                    print("\nTraining Time: " + str(training_time) + " ms")
                                else:
                                    print("Training Time: " + str(training_time) + " ms")

                    time_mean, time_std = computeStats(training_times)

                    public_id = "%d.%d" % (test.id, sub_id)
                    public_results.test_results[public_id] = Result(time_mean, time_std)

                    benchmark_results.results_training.append(time_mean)
                    benchmark_results.results_training_norm.append(float(subTest.ref_time) / time_mean)

                    if verbose > 0:
                        prefix = "%d.%d - training " % (test.id, sub_id)
                        printTestResults(prefix, subTest.batch_size, subTest.getInputDims(), time_mean, time_std, verbose)
                        sub_id += 1

        sess.close()

    testInfo.results = benchmark_results
    public_results = printScores(testInfo, public_results)

    os.chdir(start_dir)
    return public_results
