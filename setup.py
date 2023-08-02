from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, 'README.md'), 'r') as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''

version = {}
with open(os.path.join(here, 'patchgan/version.py')) as ver_file:
    exec(ver_file.read(), version)

setup(
    name='patchGAN',
    version=version['__version__'],
    description='patchGAN image segmentation model in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GNU General Public License v3',
    url='https://github.com/ramanakumars/patchGAN',
    author='Kameswara Mantha, Ramanakumar Sankar, Lucy Fortson',
    author_email='manth145@umn.edu, rsankar@umn.edu, lfortson@umn.edu',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'patchgan_train = patchgan.train:patchgan_train'
        ]
    },
    install_requires=[
        'numpy>=1.21.0,<1.25.2',
        'torch>=1.13.0',
        'torchvision>=0.14.0,<=0.15.0',
        'tqdm>=4.62.3,<=4.65.0',
        'torchinfo>=1.5.0,',
        'pyyaml',
    ]
)
