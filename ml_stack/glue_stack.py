import boto3
import json
from dotenv import load_dotenv
import os
from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_glue as glue,
    aws_rds as rds,
    aws_secretsmanager as secretsmanager,
    aws_s3 as s3
)
from constructs import Construct

secrets_client = boto3.client('secretsmanager', region_name="us-east-1")


class GlueServiceStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create IAM Role for Glue
        glue_role = iam.Role(self, "GlueRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole")
            ]
        )
        # Retrieve the RDS secret
        secret_value = secrets_client.get_secret_value(SecretId=self.node.try_get_context('secrets_manager_name'))
        secret = json.loads(secret_value['SecretString'])
        username = secret['username']
        password = secret['password']
        # Create Glue Connection for RDS
        glue_connection = glue.CfnConnection(self, "GlueConnection",
            catalog_id=self.account,
            connection_input=glue.CfnConnection.ConnectionInputProperty(
                name="rds-connection",
                connection_type="JDBC",
                connection_properties={
                    "JDBC_CONNECTION_URL": f"jdbc:postgresql://{self.node.try_get_context('rds_endpoint')}:5432/{self.node.try_get_context('db_name')}",
                    "USERNAME": username,
                    "PASSWORD": password
                },
                physical_connection_requirements=glue.CfnConnection.PhysicalConnectionRequirementsProperty(
                    availability_zone="us-east-1b",
                    # security_group_id_list=["sg-xxxxxxxx"],
                )
            )
        )
        # create glue bucket
        glue_bucket = s3.Bucket(self, "GlueBucket")
        # Create Glue Database
        glue_db = glue.CfnDatabase(self, "GlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name="my_glue_database"
            )
        )

        # Create Glue Crawler
        glue_crawler = glue.CfnCrawler(self, "GlueCrawler",
            role=glue_role.role_arn,
            database_name=glue_db.ref,
            targets={
                "jdbcTargets": [{
                    "connectionName": glue_connection.ref,
                    "path": "database-name/%",
                    "exclusions": []
                }]
            },
            name="ml-glue-crawler",
            description="Crawler for RDS data",
            schedule=glue.CfnCrawler.ScheduleProperty(
                schedule_expression="cron(0 12 * * ? *)"
            )
        )

        # Create Glue Job
        glue_job = glue.CfnJob(self, "GlueJob",
            name="ml_glue_job",
            role=glue_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="glueetl",
                script_location="s3://ml-script-bucket/ml-script.py",
                python_version="3"
            ),
            default_arguments={
                "--TempDir": "s3://ml-temp-bucket/temp/",
                "--job-language": "python"
            },
            max_capacity=2
        )