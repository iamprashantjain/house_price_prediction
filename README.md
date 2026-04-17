## Project Setup

- Create a project folder 
- git init
- dvc init
- requirements.txt
- Create venv and install requirements: pip install -r requirements.txt
- Create github repo and add remote: git remote add origin <.git url>
- Run template.py to create folder and files
- Create setup.py and pip install -e .
- Setup s3 buket & aws ecr for docker app
    - create iam user
    - attach policies
    - setup aws: pip install awscli && aws configure (access key and security key)

- dvc remote add -d myremote s3://my-mlops-project-demo/house_price_prediction
- dvc config core.autostage true ---> auto pull everytime    
- Setup dagshub and host mlflow
- Perform experiments assesment, cleaning, EDA, transformation ---> EDA round2 ---> transformation, model building & hyperparameters (track with mlflow) in 4 phases: basic, features, algorithm, hyperparameter tuning
- Convert best overall into dvc pipeline: create components like data_ingestion, data_transformation, model_trainer, model_evalutaion, logger etc
- Upload best model on dagshub hosted mlflow model_registry in staging
- Perform model test & if success then push model to production
- Create fastapi (fetch latest model in production from mlflow model_registry and make predictions)

## CICD

- Create CI/CD pipeline using github actions and docker
    - **CI (continuous integration)** is a software development practice where developers regularly update the code with new changes and merge thier code changes into github or any other shared centralized repository usually multiple times a day. each merge triggers automation build and testing to detect and fix any issues. This helps to ensure that software remains in deployable stage everytime. 

    - STEPS:
        + Create a folder .github/workflows/ci.yaml
            - name, on, jobs (name, runs on, steps)

        + Add github secrets:
            - AWS_ACCESS_KEY_ID
            - AWS_SECRET_ACCESS_KEY
            - AWS_REGION
            - S3_BUCKET_NAME
            - DAGSHUB_PAT

        + **This step will take a lot of your time.. lol**
        
        + Test fast api in CI

        + **Docker packages your ML code, libraries, and system settings into a portable container so it runs the same way on any machine—ending "it works on my laptop" problems forever.**
            - Isolated (work in isolated environment)
            - Scalability (copy paste docker images to scale)
            
            - **Docker Engine**
                - Docker deamon: Its a backend service running on host machine
                - Docker CLI: user interacts with docker deamon using CLI
                - Rest API: communication b/w docker deamon and cli happens through restapi
            
            - **Docker images**
                - Its a stand-alone lightweight and executable software that includes everything needed to run a software like runtime, libraries, environmental variables, config files etc

            - **Docker Container**
                - Its a running instance of docker image 
                - When we run the docker image, then it creates instance of docker image which is called docker container

            **Docker File**
                - Its a text file which contians instructions of how to build docker image
                - Each instruction creates a layer in the image
                - Dockerfile is used to automate the image creation process

            **Docker Registry**
                - where we can save our docker image same as in github we save our code repo, in docker registry we store our docker images
                - like Dockerhub, AWS ECR etc


        + Dockerize fastapi app
            - Start docker desktop
            - Create Dockerfile (dockerize only fastapi not whole project)
            - `docker build -t house-price-api:latest .` to build new image from dockerfile
            - `docker system prune -a --volumes -f` to delete everything
            - `docker run -d -p 8080:8000 -e DAGSHUB_PAT="your_dagshub_token" house-price-api` to run the app
            - `docker ps` or `docker ps -a` to check running images
            - `docker logs <container id>` to check logs if any error
            - `check http://localhost:8080/docs` and test with below sample values

                {
                    "area_sqft": 650,
                    "bedroom_num": 1,
                    "bathroom_num": 1,
                    "balcony_num": 1,
                    "property_type": "flat",
                    "facing": "East",
                    "agePossession": "5-10 years",
                    "floor_number": 2,
                    "total_floors": 5,
                    "feature_count": 2,
                    "furnish_count": 1,
                    "safety_rating": 3.5,
                    "lifestyle_rating": 3.0,
                    "green_area_rating": 2.5,
                    "amenities_rating": 3.0,
                    "rate_per_sqft": 4000
                    }



                {
                    "area_sqft": 3500,
                    "bedroom_num": 5,
                    "bathroom_num": 4,
                    "balcony_num": 3,
                    "property_type": "house",
                    "facing": "South",
                    "agePossession": "1-3 years",
                    "floor_number": 2,
                    "total_floors": 2,
                    "feature_count": 10,
                    "furnish_count": 8,
                    "safety_rating": 4.8,
                    "lifestyle_rating": 4.9,
                    "green_area_rating": 4.7,
                    "amenities_rating": 4.8,
                    "rate_per_sqft": 8000
                    }


                {
                    "area_sqft": 2100,
                    "bedroom_num": 4,
                    "bathroom_num": 3,
                    "balcony_num": 3,
                    "property_type": "flat",
                    "facing": "North",
                    "agePossession": "0-1 years",
                    "floor_number": 15,
                    "total_floors": 20,
                    "feature_count": 8,
                    "furnish_count": 6,
                    "safety_rating": 5.0,
                    "lifestyle_rating": 5.0,
                    "green_area_rating": 4.5,
                    "amenities_rating": 5.0,
                    "rate_per_sqft": 7000
                    }


                {
                    "area_sqft": 1800,
                    "bedroom_num": 3,
                    "bathroom_num": 2,
                    "balcony_num": 1,
                    "property_type": "house",
                    "facing": "West",
                    "agePossession": "10+ years",
                    "floor_number": 1,
                    "total_floors": 1,
                    "feature_count": 3,
                    "furnish_count": 2,
                    "safety_rating": 2.5,
                    "lifestyle_rating": 2.0,
                    "green_area_rating": 2.0,
                    "amenities_rating": 2.5,
                    "rate_per_sqft": 3500
                    }
        + Push docker image to AWS ECR (use "view push commands")
        + **Deployment startegy #1 --> EC2**
            - `RUN this docker image on EC2`
            - setup EC2
            - connect to terminal from aws
            - run below commands:
                1. sudo apt-get update
                2. sudo apt-get install -y docker.io
                3. sudo systemctl start docker
                4. sudo systemctl enable docker
                5. sudo apt-get update
                6. sudo apt-get install -y unzip curl
                7. curl "https://awscli.amazonaws.com/awscli-exe-linux_x86_64.zip" -o "awscliv2.zip"
                8. unzip awscliv2.zip
                9. sudo ./aws/install
                10. sudo usermod -aG docker ubuntu
                11. aws configure
                12. run aws push commands: 
                    + aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 739275446561.dkr.ecr.ap-south-1.amazonaws.com
                    + docker pull 739275446561.dkr.ecr.ap-south-1.amazonaws.com/house-price-api:latest
                    + docker run -d -p 8080:8000 -e DAGSHUB_PAT="your_actual_token_here" --name house-price-api 739275446561.dkr.ecr.ap-south-1.amazonaws.com/prashant-ecr:latest
                    + Add security rules: add http traffic

        + Add above docker image build, push to ECR & deployment on EC2 in CI workflow
            - Create EC2 machine and run below commands:
                - sudo apt-get update
                - sudo apt-get install -y docker.io
                - sudo systemctl start docker
                - sudo systemctl enable docker
                - sudo apt-get update
                - sudo apt-get install -y unzip curl
                - curl "https://awscli.amazonaws.com/awscli-exe-linux_x86_64.zip" -o "awscliv2.zip"
                - unzip awscliv2.zip
                - sudo ./aws/install
                - sudo usermod -aG docker ubuntu- 
                - exit

            - add github secrets to deploy ec2
            - open new terminal and check `docker --version`
            - update CI with `deploy to ec2` steps
            - commit & push 
            - add security group


        + **Additional Tips**
            - reduce the size of the docker image using `multi-stage` build to reduce the cost
            - reduce the size of the docker image using `multi-layers` basically write compact dockerfile
            - install python libraries with no-cache-dir to avoid caching unneccasry files
            - remove unneccasry files after build in dockerfile


        + **Problem with EC2 deployment (single server and single container deployment)**
            - Scalability: we can do 2 type of scaling:
                1. vertical scaling: upgrade hardware (increasing ram, hdd etc)
                2. horizontal scaling: additional servers to distribute load --> Most Preffered in ML
                
            - We can spin-up 2 ec2 servers manually but it will be impossible in case of scaling like spinning 100s of servers
            
            - Traffic Routing: lets say we have 100s of servers running, who will decide when to route      traffic and where?

            - Rigid setup: Doesn't care about the traffic, server count remains same. we have to manually stop and start the server

            - Manual Update in all servers incase we have to update with new docker image

            - Potential downtime when updating manaully

            - No health check whether servers are active or died

            - No centralized logging and monitoring -- we'll have to go to each aws control panel

            - Security management manually independentally

            - Lack of mechanism to figure out which server died & spinup new server against that

            - Complexity in CI/CD pipeline

        
            + **Solution**
                - `Manual Server starting`: Spinup new server quickly using pre-defined templates : `AWS AMIs`
                - `Traffic routing`: use `load balancer` to route traffic
                - `Rigid Setup`: Use ASG service (auto scaling group) which will auto spin or stop new servers as and when required based on threshold we have setup, It will also avoid `potential downtime issue, health checks, monitoring, security management`
            
            + **Remaining Issues**
            + when we have a new docker image, how we can update and deploy that to all servers
            + what startegy we can use to deploy
            + how can we can rollback to previous working version if something goes wrong
            + how we can integrate LB, ASG etc into CI/CD

            + **Solution**
                - code changes
                - build docker images
                - push to ecr

                - `All above can be handled by CI as of now`
                - Manual task
                - Edit launch template
                - Use latest template in ASG which will have recent changes
                - Similarly we can roll back -- just change the template
                - **Although, Downtime is still there since we are manually stopiing servers**

            + **Deployment startegy #2 --> ECS**
                - We can use **AWS CodeDeploy**:
                    1. Readymade Deployment startegy: BlueGreen or Rolling
                    2. Automated rollbacks
                    3. More control over Deployment
                    4. Smooth integration with CICD

                - **Steps**
                    0. create 2 IAM user roles
                    1. install codedeploy runner on EC2 machines
                    2. create new launch template (userdata.txt)
                    3. create new ASG
                    4. deploy ECR docker image on ASG using CodeDeploy service
                        - we create a application in CodeDeploy service
                        - we create a deployment group in this application
                        - connect this deployment group with ASG
                    5. create new deployment which will have instructions of what to do at the time of deployment like docker istall, awscli install, pull image from ecr, run etc
                    6. All these instructions will be under appspec.yaml
                    7. Run this deployment which will automatiaclly execute instructions on ASG servers
                    8. If stuck at allow traffic : It must be bcoz target group set to port 80 whereas it should be at port 8080
                    9. Target group PORT = Docker run's HOST PORT
                    10. Check : http://3.108.58.225:8080/docs & http://my-elb-1730727805.ap-south-1.elb.amazonaws.com/docs - both should work -- means load balancer is working perfectly !!
                    11. check aws_deployed_app_screenshot.pdf

                    ![alt text](image-1.png)

                    

- Model retraining CT pipeline using airflow (manual, scheduled, event driven like new data/model_performance_drift etc) using evidently ai


git add . && git commit -m "fastapi app test added with fastapi server in background" && git push origin main




- **Instead of deploying app to EC2 using code-deploy --> deploy on AWS ECS**

To modify your GitHub Actions workflow to deploy to ECS instead:

## Updated GitHub Actions Workflow for ECS Deployment

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  mlops-pipeline:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          lfs: true

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt dvc[s3]
          pip install -e .

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Configure DVC remote
        run: |
          dvc remote add -f -d myremote s3://${{ secrets.S3_BUCKET_NAME }}/house_price_prediction
          dvc remote modify myremote region ${{ secrets.AWS_REGION }}

      - name: Pull DVC data
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
        run: |
          export MLFLOW_TRACKING_USERNAME=${DAGSHUB_PAT}
          export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_PAT}
          dvc pull --force

      - name: Run DVC pipeline
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
        run: |
          export MLFLOW_TRACKING_USERNAME=${DAGSHUB_PAT}
          export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_PAT}
          dvc repro

      - name: Push DVC outputs to S3
        run: dvc push

      - name: Run all tests
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
        run: |
          export MLFLOW_TRACKING_USERNAME=${DAGSHUB_PAT}
          export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_PAT}
          python -m unittest discover -s tests -p "test_*.py" -v

      - name: Promote model to production
        if: success()
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
        run: |
          export MLFLOW_TRACKING_USERNAME=${DAGSHUB_PAT}
          export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_PAT}
          python src/promote_model.py

      - name: Run test_app.py with FastAPI server
        env:
          DAGSHUB_PAT: ${{ secrets.DAGSHUB_PAT }}
        run: |
          export MLFLOW_TRACKING_USERNAME=${DAGSHUB_PAT}
          export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_PAT}
          
          # Start FastAPI server in background
          uvicorn app.main:app --host 0.0.0.0 --port 8000 &
          
          # Wait for server to start
          sleep 10
          
          # Run test_app.py (which will test the running API)
          python tests/test_app.py -v
          
          # Kill server
          kill $(lsof -t -i:8000) || true

      - name: Login to AWS ECR
        if: success()
        run: |
          aws ecr get-login-password --region ${{ secrets.AWS_REGION }} | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com

      - name: Build Docker image
        if: success()
        run: |
          docker build -t house-price-api:latest .

      - name: Tag Docker image
        if: success()
        run: |
          docker tag house-price-api:latest ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com/${{ secrets.ECR_REPOSITORY_NAME }}:latest

      - name: Push Docker image to AWS ECR
        if: success()
        run: |
          docker push ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com/${{ secrets.ECR_REPOSITORY_NAME }}:latest

      # NEW: Deploy to ECS
      - name: Deploy to ECS
        if: success()
        run: |
          # Force new deployment with the latest image
          aws ecs update-service \
            --cluster ${{ secrets.ECS_CLUSTER_NAME }} \
            --service ${{ secrets.ECS_SERVICE_NAME }} \
            --force-new-deployment \
            --region ${{ secrets.AWS_REGION }}
          
          echo "Waiting for deployment to stabilize..."
          sleep 30
          
          # Check service status
          SERVICE_STATUS=$(aws ecs describe-services \
            --cluster ${{ secrets.ECS_CLUSTER_NAME }} \
            --services ${{ secrets.ECS_SERVICE_NAME }} \
            --region ${{ secrets.AWS_REGION }} \
            --query 'services[0].status' \
            --output text)
          
          if [ "$SERVICE_STATUS" == "ACTIVE" ]; then
            echo "✅ ECS service updated successfully!"
            
            # Get task ARN
            TASK_ARN=$(aws ecs list-tasks \
              --cluster ${{ secrets.ECS_CLUSTER_NAME }} \
              --service-name ${{ secrets.ECS_SERVICE_NAME }} \
              --region ${{ secrets.AWS_REGION }} \
              --query 'taskArns[0]' \
              --output text)
            
            # Get task status
            TASK_STATUS=$(aws ecs describe-tasks \
              --cluster ${{ secrets.ECS_CLUSTER_NAME }} \
              --tasks $TASK_ARN \
              --region ${{ secrets.AWS_REGION }} \
              --query 'tasks[0].lastStatus' \
              --output text)
            
            echo "Task status: $TASK_STATUS"
            
            # Get service endpoint (if using ALB)
            if [ ! -z "${{ secrets.ALB_DNS_NAME }}" ]; then
              echo "🌐 Service available at: http://${{ secrets.ALB_DNS_NAME }}"
            fi
          else
            echo "❌ Service update failed!"
            exit 1
          fi

      # Optional: Run smoke tests against deployed ECS service
      - name: Smoke test ECS deployment
        if: success()
        run: |
          if [ ! -z "${{ secrets.ALB_DNS_NAME }}" ]; then
            echo "Running smoke tests against ECS deployment..."
            sleep 30  # Give service time to fully start
            
            # Test health endpoint
            HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://${{ secrets.ALB_DNS_NAME }}/health)
            if [ "$HEALTH_CHECK" == "200" ]; then
              echo "✅ Health check passed!"
            else
              echo "❌ Health check failed with status: $HEALTH_CHECK"
              exit 1
            fi
            
            # Test prediction endpoint with sample data
            PREDICTION=$(curl -s -X POST http://${{ secrets.ALB_DNS_NAME }}/predict \
              -H "Content-Type: application/json" \
              -d '{"features": [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}')
            
            if [[ $PREDICTION == *"prediction"* ]]; then
              echo "✅ Prediction endpoint working!"
              echo "Response: $PREDICTION"
            else
              echo "❌ Prediction endpoint failed!"
              exit 1
            fi
          else
            echo "Skipping smoke tests - no ALB DNS name provided"
          fi
```

## Required Infrastructure Setup (One-time)

Before this workflow will work, you need to create the ECS infrastructure. Here's a Terraform or AWS CLI approach:

### Option 1: AWS CLI Commands to Create ECS Infrastructure

```bash
# 1. Create ECS Cluster
aws ecs create-cluster --cluster-name house-price-cluster

# 2. Create Task Definition (save as task-def.json)
cat > task-def.json << EOF
{
  "family": "house-price-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "house-price-api",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DAGSHUB_PAT",
          "value": "${DAGSHUB_PAT}"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/house-price-api",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-def.json

# 3. Create ECS Service (with public IP for testing)
aws ecs create-service \
  --cluster house-price-cluster \
  --service-name house-price-service \
  --task-definition house-price-api \
  --launch-type FARGATE \
  --desired-count 1 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" \
  --region ${AWS_REGION}
```

### Option 2: Create ECS with Load Balancer (Production Ready)

For production, you'll want an Application Load Balancer:

```bash
# Create load balancer
aws elbv2 create-load-balancer \
  --name house-price-alb \
  --subnets subnet-xxxxx subnet-yyyyy \
  --security-groups sg-xxxxx \
  --scheme internet-facing

# Create target group
aws elbv2 create-target-group \
  --name house-price-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxxxx \
  --target-type ip \
  --health-check-path /health

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn ${ALB_ARN} \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=${TG_ARN}

# Update service to use load balancer
aws ecs update-service \
  --cluster house-price-cluster \
  --service house-price-service \
  --load-balancers "targetGroupArn=${TG_ARN},containerName=house-price-api,containerPort=8000"
```

## GitHub Secrets to Add

Add these secrets to your GitHub repository:

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `AWS_REGION` | Your AWS region (e.g., ap-south-1) |
| `AWS_ACCOUNT_ID` | Your AWS account ID (739275446561) |
| `ECR_REPOSITORY_NAME` | Your ECR repo name (prashant-ecr) |
| `ECS_CLUSTER_NAME` | Your ECS cluster name (e.g., house-price-cluster) |
| `ECS_SERVICE_NAME` | Your ECS service name (e.g., house-price-service) |
| `ALB_DNS_NAME` | (Optional) Your ALB DNS name for testing |
| `DAGSHUB_PAT` | Your DagsHub PAT |
| `S3_BUCKET_NAME` | Your S3 bucket for DVC |

## Advantages of ECS over EC2

1. **No server management** - AWS handles the underlying EC2 instances
2. **Auto-healing** - Failed containers are automatically restarted
3. **Rolling updates** - Zero-downtime deployments
4. **Service discovery** - Built-in DNS for inter-service communication
5. **Auto-scaling** - Scale based on CPU/memory or custom metrics
6. **Better logging** - Native CloudWatch integration
7. **Resource optimization** - Right-size containers without managing instances

## Comparison Summary

| Aspect | EC2 (Your commented code) | ECS (New approach) |
|--------|--------------------------|-------------------|
| **Server management** | Manual (SSH, updates) | Automatic |
| **Deployment** | SSH + docker commands | `update-service` API call |
| **Scaling** | Manual | Automatic |
| **Rollback** | Manual re-deploy | One command to revert |
| **Logging** | Need to setup | Built-in CloudWatch |
| **High availability** | Single instance | Multiple AZs by default |
| **Cost** | Pay for full EC2 | Pay only for resources used (Fargate) |

This ECS deployment will give you a production-grade setup with minimal maintenance overhead compared to managing EC2 instances manually or via EC2 & CodeDeploy
