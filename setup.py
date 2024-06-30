from setuptools import find_packages, setup

with open('app/README.md', 'r') as f:
    long_description = f.read()

setup(
    name='algotrading',
    version='0.1.0',
    author='Sean Meehan',
    author_email='sjmeehan9@gmail.com',
    description='Build AI/ML trading models, backtest and trade live with the IB API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sjmeehan9/algo-trading',
    package_dir={'': 'app'},  # Pointing to the algotrading directory
    packages=find_packages(where='app', include=['*']),
    package_data={'config': ['*.yml']},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'gymnasium>=0.29.1',
        'numpy>=1.26.3',
        'pandas>=2.1.4',
        'pytz>=2023.3.post1',
        'PyYAML>=6.0.1',
        'scikit-learn>=1.3.2',
        'stable_baselines3>=2.3.2',
        'tensorflow>=2.16.1'
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'twine>=4.0.2'],
    },
    python_requires='>=3.10'
)