# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Skipass/*.py

black:
	@black scripts/* Skipass/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Skipass-*.dist-info
	@rm -fr Skipass.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
		'{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
				'{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
		'{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

run_grid:
	@python -m Skipass.model.grid_search


# ----------------------------------
#      SETUP GCP
# ----------------------------------
# Project and bucket variables
PROJECT_ID='skipass-325207'
BUCKET_NAME='skipass_325207_model'
BUCKET_FOLDER='skipass_325207_data'
BUCKET_TRAINING_FOLDER = 'skipass_325207_trainings'
LOCAL_PATH="/Users/devasou/code/LCYRify/Skipass/documentation/liste_stations_rawdata_synop.txt"
REGION=europe-west1
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})
# Requirements
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=2.5
# Package params 
PACKAGE_NAME=Skipass
FILENAME=model
# Job 
JOB_NAME=skipass_$(shell date +'%Y%m%d_%H%M%S')


set_project:
	@gcloud config set project ${PROJECT_ID}
create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}
upload_data:
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier CUSTOM \
		--master-machine-type n1-standard-96 \
		--stream-logs
