# MLOps Essentials: Model Development and Integration

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
  
# Machine learning life cycle

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

# Unique challenges with ML

1. **Artifacts in Software vs. ML Engineering**:
   - **Software Engineering Artifacts**: Code and records (requirements, design, test plans, results).
   - **ML Engineering Artifacts**: Code, records, data (training data), and models.

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

# 
