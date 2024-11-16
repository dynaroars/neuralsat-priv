from setuptools import setup, find_packages
from pathlib import Path

version = '0.1'

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='NeuralSAT',
    version=version,
    description='DNN Verifier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'beartype>=0.16.4',
        'certifi>=2022.12.7',
        'charset-normalizer>=2.1.1',
        'coloredlogs>=15.0.1',
        'filelock>=3.9.0',
        'flatbuffers>=23.5.26',
        'fsspec>=2023.4.0',
        'gurobipy>=11.0.0',
        'humanfriendly>=10.0',
        'idna>=3.4',
        'Jinja2>=3.1.2',
        'MarkupSafe>=2.1.3',
        'mpmath>=1.2.1',
        'networkx>=3.0rc1',
        'numpy>=1.24.1',
        'onnx>=1.15.0',
        'onnxruntime>=1.16.3',
        'packaging>=23.2',
        'Pillow>=9.3.0',
        'protobuf>=4.25.1',
        'psutil>=5.9.6',
        'requests>=2.28.1',
        'sortedcontainers>=2.4.0',
        'sympy>=1.11.1',
        'tqdm>=4.66.1',
        'typing_extensions>=4.8.0',
        'urllib3>=1.26.13',
        'termcolor>=2.4.0',
    ],
    platforms=['any'],
    license='MIT',
)