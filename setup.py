from setuptools import setup

setup(name='lipnet',
    version='0.1.6',
    description='End-to-end sentence-level lipreading',
    url='http://github.com/rizkiarm/LipNet',
    author='Muhammad Rizki A.R.M',
    author_email='rizki@rizkiarm.com',
    license='MIT',
    packages=['lipnet'],
    zip_safe=False,
	install_requires=[
        'Keras',
        'editdistance',
		'h5py',
		'matplotlib',
		'numpy',
		'python-dateutil',
		'scipy==1.1.0',
		'Pillow',
		'tensorflow',
		'Theano',
        'nltk',
        'sk-video',
        'cmake',
        'dlib',
    ])

# need to run this
#pip install git+https://www.github.com/keras-team/keras-contrib.git
#pip install sklearn
#pip install scikit-image
#pip install keras-vis
#pip install scipy==1.1.0
