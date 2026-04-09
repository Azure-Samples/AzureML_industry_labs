# Contributing to Azure ML Demos

Thank you for your interest in contributing! This repository is a collection of hands-on Azure Machine Learning labs across different industry use-cases. We welcome new labs, improvements, and bug fixes.

## Adding a New Lab

Each lab lives in its own top-level directory. Follow the standard structure below so that all labs remain consistent and easy to navigate.

### Standard Lab Structure

```
your_lab_name/
├── main.py                  # Pipeline orchestration & submission
├── requirements.txt         # Python dependencies
├── Dockerfile               # Custom Azure ML environment image
├── README.md                # Lab-specific documentation
├── .amlignore               # Files to exclude from Azure ML snapshots
├── data_processing/
│   ├── __init__.py
│   └── preprocess.py        # Reusable dataset / preprocessing utilities
├── model/
│   ├── __init__.py
│   └── <model>.py           # Model architecture definition
└── pipeline/
    ├── preprocess_step.py   # Data preprocessing pipeline step
    ├── train_step.py        # Model training pipeline step
    ├── register_model.py    # Model registration pipeline step
    ├── deploy_endpoint.py   # Endpoint deployment pipeline step
    └── score.py             # Scoring script for inference
```

### Checklist

Before submitting a new lab, ensure you have:

- [ ] Created a lab folder with the standard structure above
- [ ] Written a `README.md` inside your lab folder covering:
  - Problem statement and industry context
  - Dataset description and source
  - Pipeline architecture diagram (Mermaid or image)
  - Prerequisites and step-by-step instructions to run
  - Expected outputs
- [ ] Listed all Python dependencies in `requirements.txt`
- [ ] Provided a `Dockerfile` based on an Azure ML curated base image
- [ ] Added an `.amlignore` to exclude large data files and artifacts
- [ ] Tested the pipeline end-to-end on an Azure ML workspace
- [ ] Created a `lab.json` with lab metadata (required: `name`, `industry`, `description`; recommended: `detailedDescription`, `language`, `useCase`, `authors`) — a GitHub Action will update the root README and GitHub Pages site automatically

### Conventions

- **Pipeline orchestration** goes in `main.py` using the Azure ML SDK v2 `@pipeline` decorator.
- **Experiment tracking** should use MLflow via the `azureml-mlflow` integration.
- **Model registration** should include metadata tags (accuracy, data asset versions, etc.).
- **Deployment** should target either batch or managed online endpoints as appropriate.
- Use **incremental processing** where possible (manifest tracking, change flags) so that pipelines skip unnecessary work.

## General Contributions

For bug fixes, documentation improvements, or other enhancements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-change`)
3. Commit your changes with clear commit messages
4. Open a pull request describing what changed and why

## Code Style

- Python code should follow PEP 8
- Use type hints where practical
- Keep functions focused and well-named — prefer clarity over comments

## Questions?

Open an issue if you have any questions or suggestions.
