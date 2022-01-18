FROM pytorch/torchserve:0.5.2-cpu

USER root

COPY models/ models/

RUN torch-model-archiver \
		--model-name test_model \
		--version 1.0 \
		--serialized-file models/deployable_model.pt \
		--export-path models/model_store \
		--extra-files models/index_to_name.json \
		--handler image_classifier

CMD ["torchserve", \
     "--start", \
     "--model-store", \
     "models/model_store", \
     "--models", \
     "test_model=test_model.mar"]

# torchserve --start --model-store models/model_store --models test_model=test_model.mar