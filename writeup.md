# Project Introduction

This project is mainly about machine learning with AWS. It contains four parts:
 - (1) Distributed training in AWS SageMaker
 - (2) Multi-thread training on AWS EC2
 - (3) Lambda function for prediction
 - (4) Lambda concurrency and endpoint auto-scaling

# Distributed training in AWS SageMaker

In this part, I use a SageMaker notebook to run model training and deploy code. For the notebook instance type, I choose ml.t3.medium because all the training runs as a SageMaker training job, and the notebook instance only needs to trigger the process. Therefore, there is no so much requirements for the instance's performance, and the cheapest instance type should be sufficient for this task.<br/>
Regarding storage, although the data is temporarily stored on the instance, the data size is small, so 5 GB of storage is enough.<br/>
All the code and screenshots are stored in the `code` and `Training and Deployment` folders, and the descriptions are as follows:

| File Name | Description |
| --------- | ----------- |
| code/train_and_deploy-solution.ipynb | nodebook file for the end to end machine learning process |
| code/hpo_correct.py | python file for hyperparameter tuning |
| code/hpo_distributed.py | python file for sagemaker distributed training |
| code/inference2.py | entry point for sagemaker endpoint |
| Training and deployment/endpoint.png | screenshot for endpoint |
| Training and deployment/s3.png | screenshot for s3 |
| Training and deployment/nodebook.png | screenshot for nodebook instance |

# Multi-thread training on AWS EC2

This section demonstrates how to train the model on an EC2 instance. Unlike SageMaker training jobs, training on EC2 requires more consideration when selecting the instance type. Since CNNs are compute-intensive, CPU performance is more critical than memory or storage.
For this reason, I chose a compute-optimized EC2 instance (cX family). Given that the image dataset is relatively small, a 2-core CPU should be sufficient to complete the training within a reasonable timeframe. Therefore, I selected the c3.large instance type for EC2 training, balancing cost and performance effectively.

The SageMaker and EC2 training scripts have similar training logic, but they are designed for different environments, so they work differently.
The SageMaker script uses special features from SageMaker, like built-in libraries, debugging tools, and performance tracking. However, SageMaker has some limits. It manages many settings automatically, so the code must follow certain rules, making it less flexible than EC2 training.
On the other hand, EC2 training is more flexible because you control everything. You can choose any tools, libraries, and settings. However, you need to set up debugging and monitoring yourself, which can take extra effort.<br/><br/>

All the code and screenshots are stored in the `code` and `EC2 Training` folders, and the descriptions are as follows:

| File Name | Description |
| --------- | ----------- |
| code/ec2train1.py | python file for training in EC2 |
| EC2 Training/ec2-training.png | screenshot for ec2 type and training |

# Lambda function setup

In this section, we deploy a Lambda function to perform image classification using a SageMaker inference endpoint. It first retrieves the image URL from the incoming event payload. If the URL is missing, it raises a `ValueError`. The function then creates a request payload containing the image URL and sends it to the SageMaker endpoint using the `invoke_endpoint` method. The response from the endpoint is read, decoded, and parsed as JSON. If the inference is successful, the function returns the prediction results with a 200 status code and proper headers for content type and CORS support. If an error occurs at any point, the function logs the error and returns a 500 status code with the error message in the response body.

To run the Lambda function successfully, we need to grant it permission to access SageMaker. In AWS, this can be achieved by creating a service role and attaching a security policy that gives the Lambda function the necessary permissions. Initially, I attached the SageMaker Full Access policy, which allowed the Lambda function to interact with the endpoint. While this works, it is risky because full access enables the Lambda function to perform other tasks, such as describing models or even deleting the endpoint. Granting unnecessary permissions is not a good practice. Therefore, I decided to modify the permissions to include only the "InvokeEndpoint" permission and restrict the access to a specific endpoint.

I found some security problems on my IAM page. In the past, I created roles with administrator permissions and made access keys to use the AWS CLI for deploying resources. This makes things easier, but it can be dangerous.For example, if someone gets the access key, they could do anything in my AWS account, like deleting resources, creating new ones, or even stealing data, because the key has full access. So I also delete all the uncessary role.

All the code and screenshots are stored in the `code` and `EC2 Training` folders, and the descriptions are as follows:


| File Name | Description |
| --------- | ----------- |
| code/lambdafunction.py	| Python file for the Lambda function | 
| Lambda function setup/lambda-full-access-policy.png	| Screenshot of the Lambda function with SageMaker full access | 
| Lambda function setup/lambda-limited-access-policy.png	| Screenshot of the Lambda function with SageMaker limited access| 
| Lambda function setup/lambda.png	| Screenshot of the Lambda function successfully executed | 

# Lambda concurrency and endpoint auto-scaling

This section demonstrates Lambda concurrency and endpoint auto-scaling.
For the Lambda function:

 - **Reserved concurrency**: I set the reserved concurrency to 100 to ensure the function can handle up to 100 concurrent executions without being throttled. This ensures the function has sufficient capacity allocated to meet expected demand.
 - **Provisioned concurrency**: I also set the provisioned concurrency to 80 to guarantee that 80 slots of the function are always initialized and ready to respond instantly. This is particularly important for reducing cold start latency, ensuring critical workloads or time-sensitive tasks perform efficiently.
 - **Unused capacity**: I left 20 slots unused to allow for flexibility in handling occasional spikes in traffic or to accommodate other Lambda functions running in the same account.

For the SageMaker endpoint:
 - **Auto-scaling configuration**: I set up auto-scaling with a maximum instance count of 2 to balance cost and performance. This means that the system can scale out to an additional instance if traffic exceeds the capacity of a single instance.
 - **Scaling trigger**: The scaling is triggered when the number of requests reaches 50.
   
By implementing these configurations, the system can handle varying workloads efficiently, maintaining reliability and minimizing unnecessary costs.

All the screenshots are stored in the `Concurrency and auto-scaling` folders, and the descriptions are as follows:

| File Name | Description |
| --------- | ----------- |
| Concurrency and auto-scaling/endpoint-autoscaling.py	| Screenshot for the endpoint autoscaling | 
| Concurrency and auto-scaling/endpoint-autoscaling.png	| Screenshot of the endpoint autoscaling | 
| Concurrency and auto-scaling/lambda-concurrency.png	| Screenshot of the Lambda concurrecny setting| 
| Concurrency and auto-scaling/lambda-version.png	| Screenshot of the Lambda version | 
