# Embedding model contract

This directory is registered as a lightweight Azure ML custom model asset. The
actual embedding model is the Azure OpenAI deployment selected with
`--azure-openai-deployment`; no model weights or credentials are stored here.

The asset records lineage and provides a stable model reference for the Azure ML
model batch deployment.