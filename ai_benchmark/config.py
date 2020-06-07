# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

from __future__ import print_function
from os import path


class SubTest:

    def __init__(self, batch_size, input_dimensions, output_dimensions, iterations, min_passes, max_duration, ref_time,
                 loss_function=None, optimizer=None, learning_rate=None):

        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.iterations = iterations
        self.min_passes = min_passes
        self.max_duration = max_duration
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ref_time = ref_time

    def getInputDims(self):
        inputDims = [self.batch_size]
        inputDims.extend(self.input_dimensions)
        return inputDims

    def getOutputDims(self):
        outputDims = [self.batch_size]
        outputDims.extend(self.output_dimensions)
        return outputDims


class Test:

    def __init__(self, test_id, test_type, model, model_src, use_src, tests_training, tests_inference, tests_micro):

        self.id = test_id
        self.type = test_type
        self.model = model
        self.model_src = path.join(path.dirname(__file__), "models/" + model_src)
        self.use_src = use_src
        self.training = tests_training
        self.inference = tests_inference
        self.micro = tests_micro


class TestConstructor:

    def getTests(self):

        benchmark_tests = [

            Test(test_id=1, test_type="classification", model="MobileNet-V2", model_src="mobilenet_v2.meta",
                 use_src=False,
                 tests_training=[SubTest(50, [224, 224, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=265)],
                 tests_inference=[SubTest(50, [224, 224, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=75)],
                 tests_micro=[SubTest(1, [224, 224, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=6)]),

            Test(test_id=2, test_type="classification", model="Inception-V3", model_src="inception_v3.meta",
                 use_src=False,
                 tests_training=[SubTest(20, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=275)],
                 tests_inference=[SubTest(20, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=85)],
                 tests_micro=[SubTest(1, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=17)]),

            Test(test_id=3, test_type="classification", model="Inception-V4", model_src="inception_v4.meta",
                 use_src=False,
                 tests_training=[SubTest(10, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=290)],
                 tests_inference=[SubTest(10, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=68)],
                 tests_micro=[SubTest(1, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=27)]),

            Test(test_id=4, test_type="classification", model="Inception-ResNet-V2", use_src=False,
                 model_src="inception_resnet_v2.meta",
                 tests_training=[SubTest(8, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=330)],
                 tests_inference=[SubTest(10, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=90)],
                 tests_micro=[SubTest(1, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=37)]),

            Test(test_id=5, test_type="classification", model="ResNet-V2-50", use_src=False,
                 model_src="resnet_v2_50.meta",
                 tests_training=[SubTest(10, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=172)],
                 tests_inference=[SubTest(10, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=48)],
                 tests_micro=[SubTest(1, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=10)]),

            Test(test_id=6, test_type="classification", model="ResNet-V2-152", use_src=False,
                 model_src="resnet_v2_152.meta",
                 tests_training=[SubTest(10, [256, 256, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=265)],
                 tests_inference=[SubTest(10, [256, 256, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=60)],
                 tests_micro=[SubTest(1, [256, 256, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=25)]),

            Test(test_id=7, test_type="classification", model="VGG-16", use_src=False,
                 model_src="vgg_16.meta",
                 tests_training=[SubTest(2, [224, 224, 3], [1000], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=190)],
                 tests_inference=[SubTest(20, [224, 224, 3], [1000], 22, min_passes=5, max_duration=30, ref_time=110)],
                 tests_micro=[SubTest(1, [224, 224, 3], [1000], 22, min_passes=5, max_duration=30, ref_time=56)]),

            Test(test_id=8, test_type="enhancement", model="SRCNN 9-5-5", model_src="srcnn.meta", use_src=False,
                 tests_training=[SubTest(10, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=285)],
                 tests_inference=[
                     SubTest(10, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=100),
                     SubTest(1, [1536, 1536, 3], [1536, 1536, 3], 22, min_passes=5, max_duration=30, ref_time=85)],
                 tests_micro=[
                     SubTest(1, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=10)]),

            Test(test_id=9, test_type="enhancement", model="VGG-19 Super-Res", model_src="vgg19.meta", use_src=False,
                 tests_training=[SubTest(10, [224, 224, 3], [224, 224, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=274)],
                 tests_inference=[
                     SubTest(10, [256, 256, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=114),
                     SubTest(1, [1024, 1024, 3], [1024, 1024, 3], 22, min_passes=5, max_duration=30, ref_time=162)],
                 tests_micro=[
                     SubTest(1, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=43)]),

            Test(test_id=10, test_type="enhancement", model="ResNet-SRGAN", model_src="srgan.meta", use_src=False,
                 tests_training=[SubTest(5, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=170)],
                 tests_inference=[
                     SubTest(10, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=110),
                     SubTest(1, [1536, 1536, 3], [1536, 1536, 3], 22, min_passes=5, max_duration=30, ref_time=100)],
                 tests_micro=[
                     SubTest(1, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=17)]),

            Test(test_id=11, test_type="enhancement", model="ResNet-DPED", model_src="dped.meta", use_src=False,
                 tests_training=[SubTest(15, [128, 128, 3], [128, 128, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=200)],
                 tests_inference=[
                     SubTest(10, [256, 256, 3], [256, 256, 3], 22, min_passes=5, max_duration=30, ref_time=135),
                     SubTest(1, [1024, 1024, 3], [1024, 1024, 3], 22, min_passes=5, max_duration=30, ref_time=215)],
                 tests_micro=[
                     SubTest(1, [256, 256, 3], [256, 256, 3], 22, min_passes=5, max_duration=30, ref_time=15.5)]),

            Test(test_id=12, test_type="segmentation", model="U-Net", model_src="unet.meta", use_src=False,
                 tests_training=[SubTest(4, [256, 256, 3], [256, 256, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=240)],
                 tests_inference=[
                     SubTest(4, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=220),
                     SubTest(1, [1024, 1024, 3], [1024, 1024, 3], 22, min_passes=5, max_duration=30, ref_time=215)],
                 tests_micro=[
                     SubTest(1, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=57)]),

            Test(test_id=13, test_type="segmentation", model="Nvidia-SPADE", model_src="spade.meta", use_src=False,
                 tests_training=[SubTest(1, [128, 128, 3], [128, 128, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=200)],
                 tests_inference=[
                     SubTest(5, [128, 128, 3], [128, 128, 3], 22, min_passes=5, max_duration=30, ref_time=110)],
                 tests_micro=[
                     SubTest(1, [128, 128, 3], [128, 128, 3], 22, min_passes=5, max_duration=30, ref_time=46)]),

            Test(test_id=14, test_type="segmentation", model="ICNet", model_src="icnet.meta", use_src=False,
                 tests_training=[SubTest(10, [1024, 1536, 3], [1024, 1536, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=815)],
                 tests_inference=[
                     SubTest(5, [1024, 1536, 3], [1024, 1536, 3], 22, min_passes=5, max_duration=30, ref_time=270)],
                 tests_micro=[
                     SubTest(1, [1024, 1536, 3], [1024, 1536, 3], 22, min_passes=5, max_duration=30, ref_time=33.5)]),

            Test(test_id=15, test_type="segmentation", model="PSPNet", model_src="pspnet.meta", use_src=False,
                 tests_training=[SubTest(1, [512, 512, 3], [64, 64, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=214)],
                 tests_inference=[
                     SubTest(5, [720, 720, 3], [90, 90, 3], 22, min_passes=5, max_duration=30, ref_time=472)],
                 tests_micro=[SubTest(1, [720, 720, 3], [90, 90, 3], 22, min_passes=5, max_duration=30, ref_time=103)]),

            Test(test_id=16, test_type="segmentation", model="DeepLab", model_src="deeplab.meta", use_src=False,
                 tests_training=[SubTest(1, [384, 384, 3], [48, 48, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=191)],
                 tests_inference=[
                     SubTest(2, [512, 512, 3], [64, 64, 3], 22, min_passes=5, max_duration=30, ref_time=125)],
                 tests_micro=[
                     SubTest(1, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=67)]),

            Test(test_id=17, test_type="enhancement", model="Pixel-RNN", model_src="pixel_rnn.meta", use_src=True,
                 tests_training=[SubTest(10, [64, 64, 3], [64, 64, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=1756)],
                 tests_inference=[
                     SubTest(50, [64, 64, 3], [64, 64, 3], 22, min_passes=5, max_duration=30, ref_time=665)],
                 tests_micro=[]),

            Test(test_id=18, test_type="nlp", model="LSTM-Sentiment", model_src="lstm.meta", use_src=True,
                 tests_training=[SubTest(10, [1024, 300], [2], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=728)],
                 tests_inference=[SubTest(100, [1024, 300], [2], 22, min_passes=5, max_duration=30, ref_time=547)],
                 tests_micro=[]),

            Test(test_id=19, test_type="nlp-text", model="GNMT-Translation", model_src="gnmt.meta", use_src=False,
                 tests_training=[],
                 tests_inference=[SubTest(1, [1, 20], [None], 22, min_passes=5, max_duration=30, ref_time=193)],
                 tests_micro=[])

        ]

        return benchmark_tests

