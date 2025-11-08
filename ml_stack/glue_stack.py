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

        # IAM Role
        glue_role = iam.Role(self, "GlueRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole")
            ]
        )
        # RDS secret
        secret_arn = (
            f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:"
            f"{self.node.try_get_context('secrets_manager_name')}"
        )

        # Glue Connection for RDS
        glue_connection = glue.CfnConnection(self, "GlueConnection",
            catalog_id=self.account,
            connection_input=glue.CfnConnection.ConnectionInputProperty(
                name="rds-connection",
                connection_type="JDBC",
                connection_properties={
                    "JDBC_CONNECTION_URL": (
                        f"jdbc:postgresql://"
                        f"{self.node.try_get_context('rds_endpoint')}:5432/"
                        f"{self.node.try_get_context('db_name')}"
                    ),
                    "SECRET_ID": secret_arn
                },
                physical_connection_requirements=glue.CfnConnection.PhysicalConnectionRequirementsProperty(
                    availability_zone="us-east-1b",
                    security_group_id_list=[self.node.try_get_context('security_group_id')],
                    subnet_id=self.node.try_get_context('subnet_id')
                )
            )
        )

        # Bucket
        glue_bucket = s3.Bucket(self, "GlueBucket")
        glue_bucket.grant_read_write(glue_role)
        glue_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            resources=[f"{glue_bucket.bucket_arn}/*"]
        ))

        # Glue Database
        glue_db = glue.CfnDatabase(self, "GlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name="ml_glue_database",
                description="Database for RDS ingestion"
            )
        )

        # Glue Crawler
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
            ),
            schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                update_behavior="UPDATE_IN_DATABASE",
                delete_behavior="LOG"
            )
        )
        glue_crawler.add_dependency(glue_connection)

        # Glue Job
        glue_job = glue.CfnJob(self, "GlueJob",
            name="ml_glue_etl_job",
            role=glue_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="glueetl",
                script_location=f"s3://{glue_bucket.bucket_name}/scripts/ml-script.py",
                python_version="3"
            ),
            default_arguments={
                "--TempDir": f"s3://{glue_bucket.bucket_name}/temp/",
                "--job-language": "python",
                "--enable-metrics": "true",
                "--enable-continuous-cloudwatch-log": "true",
                "--job-bookmark-option": "job-bookmark-enable",
                "--output_bucket": glue_bucket.bucket_name,
                "--database_name": glue_db.ref,
                "--connection_name": glue_connection.ref
            },
            glue_version="4.0",
            max_retries=1,
            timeout=60,
            number_of_workers=2,
            worker_type="G.1X",
            connections=glue.CfnJob.ConnectionsListProperty(
                connections=[glue_connection.ref]
            )
        )
