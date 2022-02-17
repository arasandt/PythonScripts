import json
import boto3

client = boto3.client('dynamodb')

def lambda_handler(event, context):
    # TODO implement
    #print(event)
    client.create_backup(TableName='inventory', BackupName="myautomaticbackup")
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda CLI!')
    }
