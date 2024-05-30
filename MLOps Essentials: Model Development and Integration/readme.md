# MLOps Essentials: Model Development and Integration

# Introduction to MLOps

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/1d6cbcaf-01d1-4c1e-978a-b4c371c3cbb5)


1. **Course Coverage**:
   - Focus on build activities in machine learning: creating, testing, and packaging models.
   - Core activities in building: feature engineering, model training, testing, and packaging.
   - MLOps build side: requirements management, data and training pipelines, data governance, experiment tracking, integrations, and model management.

2. **Core ML and MLOps Activities**:
   - Build activities: creating and testing models.
   - Run activities: deploying, executing, and monitoring models.
   - Run side of MLOps: infrastructure management, deployment, serving, monitoring, and responsible AI.

3. **Course Scope**:
   - Focuses on build side MLOps activities.
   - Discusses purpose, context, techniques, methods, tools, and best practices for each activity.
   - Provides an overview, recommends further reading for deep dives.

4. **Prerequisites**:
   - Prior understanding of machine learning operations.
   - Knowledge of core ML activities.
   - Experience in building and running ML models, either hands-on or in management.

5. **Target Audience**:
   - Suitable for data scientists, ML engineers, managers, and product owners in the ML domain.

6. **Tools and Technologies**:
   - MLOps tools ecosystem is rapidly evolving.
   - Recommendations based on current status; tools from AWS, GCP, and Azure not covered in detail.
  
## Machine learning life cycle

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/99384d56-2f07-4aab-818b-de5f0107d912)

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/e7fbb0f5-d64f-46ea-b27e-a23e4396625e)

1. **Overview of ML Life Cycle**:
   - Journey of an ML application: concept to production.
   - Cyclic process with continuous improvements based on new data, requirements, and model decay.

2. **Steps in ML Life Cycle**:
   - **Requirements**: Defined by product owner.
   - **Workflow Design**: Planning the project execution.
   - **Data Acquisition**: Gathering training data.
   - **Feature Engineering**: Cleansing, transforming, and extracting features from data.
   - **Model Training**: Using features to train the model.
   - **Testing and Evaluation**: Ensuring model meets performance goals.
   - **Deployment**: Model used for inference in production.
   - **Data Collection During Inference**: Gathering new data from production to create additional training data.

3. **Continuous Process**:
   - Continuous refinement to optimize the model.
   - Adapting to changes in the business environment through retraining and updates.

4. **Course Focus**:
   - Not on building models, but on the ecosystem supporting the ML life cycle.
   - Emphasis on executing the ML life cycle with efficiency and control.

## Unique challenges with ML

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/1450252f-7d43-4bba-ac38-00b515d00533)

1. **Artifacts in Software vs. ML Engineering**:
   - **Software Engineering Artifacts**: Code and records (requirements, design, test plans, results).
   - **ML Engineering Artifacts**: Code, records, data (training data), and models.
![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/48d5f097-a614-4727-8270-48fdd3a46332)

2. **Artifact Interactions in ML**:
   - Requirements influence training data preparation, creating processing records.
   - Training data and requirements are used to build models, generating training performance records.
   - Models are integrated into code for executable binaries.
   - Execution of models generates new training data, creating a cyclic dependency among artifacts.

3. **Unique Challenges in ML Development**:
   - **Coordination**: Each artifact follows its own life cycle but depends on others, necessitating coordination.
   - **Collaboration**: Increased collaboration requirements between teams producing different artifacts.
   - **Tools and Workflows**: Need for additional tools and workflows to manage and govern artifact life cycles.
   - **Skill Sets**: Diverse skills required for creating and maintaining different artifacts.
   - **Integrated Workflow**: Essential for efficient management of ML application development.

## DevOps

1. **Definition and Purpose of DevOps**:
   - Popular methodology for continuous software development and delivery.
   - Integrates development, testing, and operations into one seamless workflow.
   - Utilizes agile principles, best practices, and tools for optimal development.

2. **Integration and Workflow**:
   - Combines people, processes, and technologies.
   - Facilitates faster iterations and continuous changes in software.
   - Focuses on automation to enhance efficiency.

3. **DevOps Life Cycle**:
   - Iterative process involving planning, coding, building, testing, releasing, deploying, operating, and monitoring.
   - Performance feedback is used for future planning and iterations.

4. **Traditional vs. DevOps Approach**:
   - Traditional: Development, testing, and operations handled by separate teams.
   - DevOps: Integrated team manages development, testing, and operations.

5. **Foundation for MLOps**:
   - DevOps principles and practices form the basis for MLOps, to be discussed further.

## What is MLOps?

### Key Points:
1. **Definition and Purpose of MLOps**:
   - Set of best practices for managing ML artifacts through efficient workflows, collaboration, and tracking.
   - Not a specific product or technique, but a set of processes and practices supported by automation and tools.

2. **Integration with DevOps**:
   - Extends DevOps methodology to include data engineering and model development.
   - Integrates ML artifacts (data and models) with software engineering artifacts (code and records).
   - Enables continuous model development and integration, following agile principles.

3. **MLOps Activities**:
   - **Model Deployment and Serving**: Ensures models are properly deployed and available for use.
   - **Monitoring and Performance Analytics**: Tracks model performance, drift, and bias, feeding back into the system for improvements.
   - **Automation and Tools**: Enhances efficiency in managing ML processes.

4. **MLOps Lifecycle**:
   - Starts with defining ML project requirements and design, including non-ML components (APIs, services, databases) and ML pipelines.
   - Involves developing non-ML parts and converting raw data into features through data engineering.
   - Continuous training cycle for building and refining models to meet requirements.
   - Models integrated with non-ML code and packaged for deployment.
   - **Operations Process**: Continuous deployment, model serving, performance monitoring, and feedback for further improvement.
   - Captures feature and label data from production for retraining models if necessary.

5. **Feedback Loop**:
   - Performance data and user feedback inform model governance and potential retraining.
   - New data from production is fed back into the data engineering pipeline.

## Principles of MLOps

1. **Objective of MLOps**:
   - Create an optimal end-to-end ML workflow integrating teams, modules, and artifacts.
   - Continuously improve and deliver ML solutions.

2. **Key Principles of MLOps**:
   - **Modularity**: Solutions should have well-defined boundaries and ownership to allow simultaneous evolution and integration.
   - **Continuous Development**: Enable continuous development, modeling, integration, and deployment.
   - **Automation**: Achieve efficiencies and scale through automation.
   - **Incremental Development**: Support quick time to market with incremental development.
   - **End-to-End Management**: Ensure visibility into the current state of artifacts and manage the entire workflow end-to-end.
   - **Reproducibility**: Maintain reproducibility of artifacts along with the necessary code or steps to build them.
   - **Scalability**: Scale to larger datasets, modernizations, teams, and deployments.
   - **Observability**: Enable monitoring and diagnosis with observable workflows and artifacts.
   - **Traceability**: Trace the lineage of artifacts from their data and model origins.
   - **Security**: Implement necessary access controls and privacy protections.
   - **Integration**: Ensure seamless movement of artifacts and integration of workflows.
   - **Adaptability**: Allow changes in processes or workflows without cascading impacts.
   - **Transparency**: Maintain system transparency for understanding inner workings.
   - **Collaboration**: Facilitate collaboration between teams and artifacts.
   - **Organization**: Organize the system with interfaces to outside teams, systems, and processes.

3. **Designing MLOps**:
   - Keep these principles in mind while designing MLOps processes in an organization to ensure efficient and effective ML operations.

## When to start MLOps?

# Requirements and Design

## Selecting ML projects

## Creating requirements

## Designing the ML workflow

## Assembling the team

## Choosing tools and technologies

# Data Processing and Management

## Managed data pipelines

## Automated data validation

## Managed feature stores

## Data versioning

## Data governance

## Tools and technologies for data processing

# Continuous Training

## Managed training pipelines

## Creating data labels

## Experiment tracking

## AutoML

## Tools and technologies for training

# Model Management

## Model versioning

## Model registry

## Benchmarking models

## Model life cycle management

## Tools and technologies for model management

# Continuous Integration

## Solution integration pipelines

## Notebook to software

## Solution integration patterns

## Best practices for solution integration

# Conclusion

