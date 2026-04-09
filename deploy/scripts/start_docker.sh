#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 739275446561.dkr.ecr.ap-south-1.amazonaws.com

# Pull the latest image from prashant-ecr
docker pull 739275446561.dkr.ecr.ap-south-1.amazonaws.com/prashant-ecr:latest

# Stop and remove existing container if it exists
docker stop house-price-api 2>/dev/null || true
docker rm house-price-api 2>/dev/null || true

# Run new container
docker run -d -p 8080:8000 -e DAGSHUB_PAT=7bed6b5be2021b1a4eaae221787bcb048ab2bcfd --name house-price-api 739275446561.dkr.ecr.ap-south-1.amazonaws.com/prashant-ecr:latest