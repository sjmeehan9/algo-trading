from setuptools import find_packages, setup

# setup(
#     name='algo-trading',
#     version='0.1.0',
#     author='Sean Meehan',
#     author_email='sjmeehan9@gmail.com',
#     description='Build AI/ML trading models, backtest and trade live with the IB API',
#     long_description=open('README.md').read(),
#     long_description_content_type='text/markdown',
#     url='https://github.com/sjmeehan9/algo-trading',
#     package_dir={'': 'algotrading'},
#     packages=find_packages(include=['algotrading', 'algotrading.*']),
#     license='MIT',
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'License :: OSI Approved :: MIT License',
#         'Operating System :: OS Independent',
#     ],
#     install_requires=[
#         'gymnasium>=0.29.1',
#         'numpy>=1.26.3',
#         'pandas>=2.1.4',
#         'pytz>=2023.3.post1',
#         'PyYAML>=6.0.1',
#         'scikit-learn>=1.3.2',
#         'stable_baselines3>=2.3.2',
#         'tensorflow>=2.16.1'
#     ],
#     extras_require={
#         'dev': ['pytest>=7.0', 'twine>=4.0.2'],
#     },
#     python_requires='>=3.10'
# )