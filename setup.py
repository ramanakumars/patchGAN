from setuptools import setup, find_packages

setup(
    name='patchGAN',
    version='0.1',
    description='patchGAN image segmentation model in PyTorch',
    license='GNU General Public License v3',
    url='https://github.com/ramanakumars/patchGAN',
    author='Kameswara Mantha, Ramanakumar Sankar, Lucy Fortson',
    author_email='manth145@umn.edu, rsankar@umn.edu, lfortson@umn.edu',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0,<1.25.2',
        'torch>=1.13.0,<=2.0.1',
        'torchvision>=0.14.0<=0.15.0',
        'tqdm>=4.62.3,<=4.65.0',
    ]
)
