
from setuptools import setup, find_packages

setup(
    name='crabgame_ai',
    version='1.0.0',
    author='Long Le',
    author_email='vietlong.lenguyen@gmail.com',
    description='RL agents for CrabGame bomb tag game.',
    license='MIT',
    keywords='python reinforcement learning CrabGame',
    url='https://github.com/vlongle/crabgame_ai',
    packages=[pkg for pkg in find_packages() if pkg != "tests"],
)
