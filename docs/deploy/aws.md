# Deploy a Truss to AWS ECS

In this guide, we'll cover how to deploy a Truss docker image to AWS using Elastic Container Registry and Elastic Container Service.

Prerequisites:

* [Docker installed](https://docs.docker.com/get-docker/)
* An AWS account with appropriate access
* AWS CLI [installed](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)

## Creating a repository on AWS

Next, you'll have to create an Elastic Container Registry. This will hold your Docker image. To do so in the UI:

1. Navigate to the ECR console [here](https://console.aws.amazon.com/ecr/repositories).
2. Underneath the **Private** tab, click on **Create Repository**.
3. Select your visibility settings and give your repository a unique name. You can choose to turn on image scan or KMS encryption.
4. Press **Create Repository**.

Or you can use the AWS CLI to create a repository. To do so, run

```
aws ecr create-repository --repository-name [REPOSITORY_NAME]
```

## Authenticating Docker client

In order to push to the registry, you'll need to authenticate your Docker client to your registry. To find how to do this,

1. Click into your ECR registry
2. At the top right, click on **Push commands**
3. Copy the first command; the command that copies the auth token from AWS and authenticates the Docker client to your specific registry. It should look something like
```
aws ecr get-login-password --region [YOUR_REGION] | docker login --username AWS --password-stdin [AWS_ACCOUNT_ID].dkr.ecr.[AWS_REGION].amazonaws.com
```
4. You should see the command exit with "Login succeeded"

## Pushing our Docker image

Now that we've authenticated our Docker client with our AWS account, it's time to push our Docker image to ECR.

First, tag the local Docker image with the following command. Make sure to fill in the variables without the brackets around them.

```
docker tag [YOUR_DOCKER_IMAGE_NAME]:latest [AWS_ACCOUNT_ID].dkr.ecr.[AWS_REGION].amazonaws.com/[YOUR_ECR_NAME]:latest
```

Then push the image to the ECR repository by doing:

```
docker push AWS_ACCOUNT_ID].dkr.ecr.[AWS_REGION].amazonaws.com/[YOUR_ECR_NAME]:latest
```

This'll take some time as our Docker image is 1.5GB.

## Create ECS Cluster

We've pushed our image to an AWS ECR repository. Next, we'll need to actually create the ECS cluster that runs that image.

1. Navigate to the AWS dashboard and select ECS (Elastic Container Service)
2. Press on **Create Cluster**
3. Select the **EC2 Linux + Networking** template
4. Give your cluster a name. For this example, we'll be using 1 instance of a `t2-medium` with the default Linux 2 AMI.
5. Click **Create** at the bottom of the page. It'll take a couple minutes for your ECS cluster to begin.

## Create a Task definition

Let's use our ECR image in ECS. To do so, we'll create a task definition.

1. Navigate back to the ECS dashboard and on the left tab, select **Task Definitions**
2. Click on **Create new Task Definition**
3. Because we used the EC2 Linux template in our ECS cluster, we'll use the EC2 launch type.
4. Give your task definition a name and allocate some memory for the task (we'll use 1gb).
5. Select **Add Container** and set **Container name** to the name of your ECS instance.
6. For **Image**, copy the URI of your ECR image. It should look something like
7. For **Port mappings**, we'll want to map the host port 80 to container port 8080 (tcp).
8. Click **Add** to add the container and then **Create** to create the task definition.

## Running your task

To run the task you just created, navigate to your task definition.
1. Select **Run task** underneath **Actions** on the task page.
2. On the "Run Task" page, select EC2 as the launch type.
3. Under **Cluster**, select the ECS cluster you created.
4. Scroll to the bottom of the page and click **Run Task**.

## Making a request

Now that you're task is running, you can make requests to your model! To get the public link for your container, navigate to your task and click into the details and you'll see the external link.

If you've been following along with the model above, you can use the snippet below to make a sample request.

```
curl -H 'Content-Type: application/json' -d '{"inputs": [[0,0,0,0]]}' -X POST [CONTAINER_LINK]:80/v1/models/model:predict
```
