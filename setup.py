from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) ->List[str]:
    """
    This will return the list of requirements

    """
    requirements=[]
    with open(file_path) as file_opj:
        requirements = file_opj.readline()
        requirements = [req.replace('\n','') for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='MLops',
    version='0.0.1',
    author='Vijay',
    author_email='Vijaykumar14198@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)