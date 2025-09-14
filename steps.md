### Creating Envioronment
```bash
conda create -p venv python==3.8
conda activate venv/
```
### Managin Git
```bash
git init
git add README.md
git commit -m "First Commit"
git status
git branch -M main
git remote add origin https://github.com/Krishnarjunmitra/Data-Science.git
git remote -v
git push -u origin main
```
### Added .gitignore file
```bash
git pull
```
### Added setup.py and reqirements.py
`search python setup.py in google for more understanding`
### Written Setup.py
```bash
from setuptools import find_packages, setup
setup(
name='mlproject',
version='0.0.1',
author='Krishnarjun',
author_email='mailtokrishnarjun@gmail.com',
packages=find_packages(),
install_requires=['pandas','numpy','seaborn']
)
```
created source folder and requirements package