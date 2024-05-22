build-and-push:
	sudo docker build --no-cache -f Dockerfile -t pierre781/benchmark:latest .
	sudo docker push pierre781/benchmark:latest

build-and-push-h100:
	sudo docker build --no-cache  --progress=plain -f Dockerfile.H100 -t pierre781/benchmark:H100 .
	sudo docker push pierre781/benchmark:H100

build-and-push-candidate:
	sudo docker build --no-cache --progress=plain -f Dockerfile.H100 -t pierre781/benchmark:candidate .
	sudo docker push pierre781/benchmark:candidate