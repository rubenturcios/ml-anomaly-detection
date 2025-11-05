#!/usr/bin/env python3
import os

import aws_cdk as cdk

from ml_stack.glue_stack import GlueServiceStack
from ml_stack.ml_stack import MlStack


app = cdk.App()
MlStack(app, "MlStack")
GlueServiceStack(app,"GlueStack")

app.synth()
