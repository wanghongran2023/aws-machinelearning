import base64
import logging
import json
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

runtime = boto3.Session().client('sagemaker-runtime')
endpoint_name = 'pytorch-inference-2024-12-20-07-48-57-618'

def lambda_handler(event, context):
    try:
        image_url = event.get("url")
        if not image_url:
            raise ValueError("Image URL is missing from the event.")

        request = {"url": image_url}

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept='application/json',
            Body=json.dumps(request)
        )

        result = response['Body'].read().decode('utf-8')
        prediction = json.loads(result)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'type_result': str(type(result)),
            'content_type_in': str(context),
            'body': json.dumps(prediction)
        }
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
