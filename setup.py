from setuptools import find_packages, setup
from typing import List 

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]: 
    requirements = []
    with open(file_path) as file_obj: 
        requirements = file_obj.readlines()
        requirements = [req.replace ("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements: 
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name = "NYC_Taxi_Trip_Duration_Prediction", 
    version = "0.0.1", 
    author = "Sagnik Sarkar", 
    description= "An End to End ML Project to Predict The Trip Duration Of NYC Taxis",
    author_email= "sagnik.sarkar22-24@bibs.co.in", 
    install_requires = get_requirements("requirements.txt"),
    packages= find_packages()
    )