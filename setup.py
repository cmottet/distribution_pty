from setuptools import setup


def readme():
      with open('README.rst') as f:
            return f.read()

setup(name='distribution_pty',
      version='0.1',
      description='Properties of some distribution functions',
      url='http://github.com/cmottet/distribution_pty',
      long_description=readme(),
      author='Clementine Mottet',
      author_email='cmottet@bu.edu',
      license='MIT',
      packages=['distribution_pty'],
      install_requires=['scipy'],
      zip_safe=False)

