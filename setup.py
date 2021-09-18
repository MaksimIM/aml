from setuptools import setup

setup(name='aml',
      version='0.1',
      description='aml work in progress',
      author='',
      author_email='',
      license='',
      packages=['aml'],
      python_requires='>=3.8',
      install_requires=['numpy>=1.19', 'matplotlib', 'scipy',
                        'imageio', 'pybullet>=3.1.7',
                        'tensorboardX', 'torch', 'gym', 'stable-baselines3',
                        'tensorboard'],
)
