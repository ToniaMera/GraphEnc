from setuptools import setup, find_packages

setup(
    name='graph_encoder_module',
    version='0.1',
    packages=find_packages(),
    install_requires=['torch', 'rdkit', 'numpy', 'torch_geometric'],
    author='Tonia Mera',
    author_email='antwniame@gmail.com',
    description='GraphEnc model for lipophilicity prediction',
    url='https://github.com/ToniaMera/GraphEnc',
)
