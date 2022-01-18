#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON_INTERPRETER=python3
VERSION=$(shell git rev-parse @)
GPROJ=valiant-splicer-337909
# Check projects name by: gcloud projects list

#################################################################################
# Python	                                                                    #
#################################################################################
## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Remove compiled
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Train model
train: 
	$(PYTHON_INTERPRETER) src/models/train_model.py data/processed models

## Predict
predict:
	$(PYTHON_INTERPRETER) src/models/predict_model.py models/model.pth

#################################################################################
# Deploy	                                                                    #
#################################################################################
## Docker
build.server: clean
	docker build -f deploy/serving.Dockerfile -t server .

push: build.server
	docker tag server gcr.io/$(GPROJ)/mlops:$(VERSION)
	docker push gcr.io/$(GPROJ)/mlops:$(VERSION)

.dumps:
	@mkdir -p deploy/.dumps

clean-deploy-files:
	rm -f deploy/.dumps/*

gen-spec: clean-deploy-files .dumps
	IMG_NAME=gcr.io/$(GPROJ)/mlops:$(VERSION) envsubst '$${IMG_NAME}' < deploy/deploytemplate.yaml > deploy/.dumps/server.yaml

deploy-server: gen-spec
	kubectl apply -f deploy/.dumps/server.yaml
# Above assumes that gcloud cluster is your default cluster (current-context)
# Watch logs locally: watch -n 1 kubectl logs [pods name]

run-local: build.server
	docker run --rm -it --name model-server gcr.io/$(GPROJ)/mlops:$(VERSION)

interactive: build.server
	docker run --rm -it --entrypoint bash --name model-server gcr.io/$(GPROJ)/mlops:$(VERSION)

## Update remote data using dvc
update-data: 
	dvc add data/
	git add data.dvc
	git commit -m "update dvc"
	git tag -a $(shell date | tr -d "[:space:]" | sed 's/://g') -m "update dvc"
	dvc push
	git push

#################################################################################
# Deploy predictions                                                            #
#################################################################################
serialize-model:
	$(PYTHON_INTERPRETER) src/models/test_deployment.py

## Deploy model
model-archive: serialize-model
	torch-model-archiver \
		--model-name test_model \
		--version 1.0 \
		--serialized-file models/deployable_model.pt \
		--export-path models/model_store \
		--extra-files models/index_to_name.json \
		--handler image_classifier


serve-local: model-archive
	torchserve --start --ncs --model-store models/model_store --models test_model=test_model.mar

# gcloud compute backend-services list
# gcloud compute backend-services get-health k8s1-4d2cf581-default-neg-demo-svc-80-5fd32c0f --global
