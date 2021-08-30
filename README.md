# Data analysis
- Document here the project: Skipass
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for Skipass in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/Skipass`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "Skipass"
git remote add origin git@github.com:{group}/Skipass.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Skipass-run
```

# Install

Go to `https://github.com/{group}/Skipass` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/Skipass.git
cd Skipass
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Skipass-run
```
