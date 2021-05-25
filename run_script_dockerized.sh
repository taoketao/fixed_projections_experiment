#!/bin/bash
echo "<><><><>  Building docker image...";
cp my_Dockerfile Dockerfile
docker build . -t learn_under_docker_container;
echo "<><><><>  Running script...";
#docker run --rm -v $(pwd)/challenge:/model challenge;
docker run --rm  learn_under_docker_container;
echo "<><><><>  Deleting docker image...";
docker rmi learn_under_docker_container; 
rm -f Dockerfile
#cp _build_dockerfile Dockerfile
