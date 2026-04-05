from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Read requirements.txt and return list of requirements"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Remove editable install if present
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name='house price prediction',
    version='0.0.1',
    author='Prashant Jain',
    author_email='p@p.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='House Price Prediction',
    python_requires='>=3.8',
)