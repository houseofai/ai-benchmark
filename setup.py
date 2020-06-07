import setuptools

REQUIRED_PACKAGES = [
    'numpy',
    'psutil',
    'py-cpuinfo',
    'pillow',
    'setuptools',
    'requests',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai-benchmark",
    version="0.1.2",
    author="Andrey Ignatov",
    author_email="andrey@vision.ee.ethz.ch",
    scripts=['bin/ai-benchmark'],
    include_package_data=True,
    description="AI Benchmark is an open source python library for evaluating AI performance of various "
                "hardware platforms, including CPUs, GPUs and TPUs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://ai-benchmark.com",
    licence="Apache License Version 2.0",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    keywords="AI Benchmark Tensorflow Machine Learning Inference Training",
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Testing',
        'Topic :: System :: Benchmark'
    ],
)
