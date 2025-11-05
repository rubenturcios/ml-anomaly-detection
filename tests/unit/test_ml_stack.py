import aws_cdk as core
import aws_cdk.assertions as assertions

from ml_stack.ml_stack import MlStack

# example tests. To run these tests, uncomment this file along with the example
# resource in ml/ml_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MlStack(app, "ml")
    template = assertions.Template.from_stack(stack)
