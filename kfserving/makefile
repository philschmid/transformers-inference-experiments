
container_name=kf-serving
aws_region=eu-central-1
aws_account_id=123
# repository=${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com
# repository=dev.local
repository=philschmi
tag=0.0.1

build:
	docker build  -t ${repository}/${container_name}:${tag} ./example


start:
	# docker run -t -i  -p 8080:8080 -v model:/opt/ml/model ${repository}/${container_name}:${tag} 
	docker run -t -i \
			-p 8080:8080 \
			-e MODEL_ID=distilbert-base-uncased-finetuned-sst-2-english \
			-e TASK=text-classification \
			${repository}/${container_name}:${tag} 


push:
	docker push ${repository}/${container_name}:${tag}