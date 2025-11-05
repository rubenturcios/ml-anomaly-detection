from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_events as events,
    aws_sagemaker as sagemaker,
    aws_lambda as lambda_,
    aws_events_targets as targets,
    Duration,
    CfnOutput
)
from constructs import Construct

import config


class MlStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.sagemaker_role = iam.Role(
            self,
            'SageMakerRole',
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_managed_policy_arn(
                    self,
                    "S3FullAccess",
                    managed_policy_arn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
                ),
                iam.ManagedPolicy.from_managed_policy_arn(
                    self,
                    "SageMakerFullAccess",
                    managed_policy_arn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                ),
            ],
            role_name=f"{self.stack_name}-SageMakerRole"
        )

        if config.DEPLOY_NOTEBOOK:
            self.deploy_notebook()

        if config.DEPLOY_EVENT_LAMBDA:
            self.deploy_event_lambda()

        CfnOutput(self, 'SageMakerRoleOutput', value=self.sagemaker_role.role_arn)

    def deploy_event_lambda(self) -> None:
        rds_secret_name = self.node.try_get_context('RDS_SECRET_NAME')
        event_prediction_endpoint_name = self.node.try_get_context('EVENT_PREDICTION_ENDPOINT_NAME')
        event_estimator_endpoint_name = self.node.try_get_context('EVENT_ESTIMATOR_ENDPOINT_NAME')
        event_predicted_table = self.node.try_get_context('EVENT_PREDICTED_TABLE')
        person_of_interest_prediction_endpoint_name = self.node.try_get_context('PERSON_OF_INTEREST_PREDICTION_ENDPOINT_NAME')
        person_of_interest_estimator_endpoint_name = self.node.try_get_context('PERSON_OF_INTEREST_ESTIMATOR_ENDPOINT_NAME')
        person_of_interest_predicted_table = self.node.try_get_context('PERSON_OF_INTEREST_PREDICTED_TABLE')

        self.event_lambda = lambda_.DockerImageFunction(
            scope=self,
            id="EventLambda",
            function_name="anamoly-event-lambda",
            # Use aws_cdk.aws_lambda.DockerImageCode.from_image_asset to build
            # a docker image on deployment
            code=lambda_.DockerImageCode.from_image_asset(
                # Directory relative to where you execute cdk deploy
                # contains a Dockerfile with build instructions
                directory="src"
            ),
            timeout=Duration.minutes(5),
            memory_size=3072,
            environment={
                'RDS_SECRET_NAME': rds_secret_name,
                'EVENT_PREDICTION_ENDPOINT_NAME': event_prediction_endpoint_name,
                'EVENT_ESTIMATOR_ENDPOINT_NAME': event_estimator_endpoint_name,
                'EVENT_PREDICTED_TABLE': event_predicted_table,
                'PERSON_OF_INTEREST_PREDICTION_ENDPOINT_NAME': person_of_interest_prediction_endpoint_name,
                'PERSON_OF_INTEREST_ESTIMATOR_ENDPOINT_NAME': person_of_interest_estimator_endpoint_name,
                'PERSON_OF_INTEREST_PREDICTED_TABLE': person_of_interest_predicted_table,
            },
        )
        self.event_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=['sagemaker:*'],
                effect=iam.Effect.ALLOW,
                resources=[
                    f'arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{event_prediction_endpoint_name}',
                    f'arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{event_estimator_endpoint_name}',
                    f'arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{person_of_interest_prediction_endpoint_name}',
                    f'arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{person_of_interest_estimator_endpoint_name}',
                ]
            )
        )
        self.event_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=['secretsmanager:GetSecretValue'],
                effect=iam.Effect.ALLOW,
                resources=[self.node.try_get_context('RDS_SECRET_ARN')]
            )
        )
        self.event_rule = events.Rule(
            self,
            "EventLambdaRule",
            targets=[targets.LambdaFunction(self.event_lambda)],
            schedule=events.Schedule.rate(Duration.days(1))
        )

        CfnOutput(self, 'EventLambdaOutput', value=self.event_lambda.function_arn)

    def deploy_notebook(self) -> None:
        """
        Creates and configures the AWS SageMaker notebook instance, including its associated IAM role,
        policies for necessary AWS service access, and a CodeCommit repository for storing notebook files.
        Also sets up a lifecycle configuration to pre-install Python libraries.
        """

        # Create the SageMaker notebook instance with the specified configuration
        self.notebook_instance = sagemaker.CfnNotebookInstance(
            self,
            "LLMPocAcceleratorNotebook",
            instance_type=config.NOTEBOOK_INSTANCE_SIZE,
            notebook_instance_name="MLPOCNotebook",
            role_arn=self.sagemaker_role
        )
