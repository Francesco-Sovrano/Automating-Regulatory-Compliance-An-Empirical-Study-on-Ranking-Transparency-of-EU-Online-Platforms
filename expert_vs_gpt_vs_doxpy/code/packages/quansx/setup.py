from setuptools import setup

with open("requirements.txt", 'r') as f:
	requirements = f.readlines()

setup(
	name='quansx',
	version='1.1',
	description='A plugin for extracting pairs of questions and answers out of natural language documents.',
	url='https://www.unibo.it/sitoweb/francesco.sovrano2/en',
	author='Francesco Sovrano',
	author_email='cesco.sovrano@gmail.com',
	license='MIT',
	packages=['quansx'],
	# zip_safe=False,
	install_requires=requirements, #external packages as dependencies
	python_requires='>=3.6',
)
