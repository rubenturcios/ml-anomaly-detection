"""
Module: config.py
Description: Configuration settings for the Alactrac POC ML project.

This module contains key configuration flags and settings used throughout the deployment of the
project infrastructure. These settings control various aspects of the deployment process, including
resource provisioning and deployment options.
"""

DEPLOY_NOTEBOOK = False
DEPLOY_EVENT_LAMBDA = True
NOTEBOOK_INSTANCE_SIZE = "ml.t3.medium"
