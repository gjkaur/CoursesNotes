# Generative AI: Working with Large Language Models

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/05132f47-3a9d-45a9-bc57-7808db15234b)


# What are large language models?

1. **Transformers and Large Language Models (LLMs)**:
   - BERT and GPT-3 are examples of large language models.
   - These models are based on transformer architecture, proposed in the 2017 paper "Attention is All You Need" by Google researchers.
   - Transformers have significantly influenced NLP.

2. **Parameters in LLMs**:
   - Parameters are values in a model updated during training.
   - Large language models have millions to billions of parameters and are trained on vast amounts of data.

3. **Focus Period**:
   - The focus is on models released between May 2020 and July 2022, following GPT-3's release.

4. **Notable LLMs Released**:
   - **Google Research**: GLaM and PaLM.
   - **DeepMind**: Gopher and Chinchilla.
   - **Microsoft and Nvidia**: Megatron-Turing NLG.
   - **Meta AI**: Open Pre-trained Transformer.
   - **Hugging Face**: Coordinated research effort resulting in the BLOOM model, involving over 1000 researchers.

5. **Availability of LLMs**:
   - Meta AI and Hugging Face have worked to make large language models accessible to researchers outside of big tech companies.

6. **Applications**:
   - The text implies that LLMs are widely used in various production applications, though specific examples are not detailed in this excerpt.


# Transformers in production:

1. **Google Search and BERT**:
   - Since 2019, Google has used BERT (Bidirectional Encoder Representations from Transformers) in its search algorithm.
   - BERT allows Google to understand more natural, conversational queries.

2. **Improved Search Understanding**:
   - Example 1: Previously, entering keywords like "curling objective" would provide general search results. With BERT, a more natural query like "what's the main objective of curling?" yields a direct answer.
   - Example 2: A query like "can you get medicine for someone pharmacy" would previously return results about filling prescriptions. With BERT, the search understands the nuance of "for someone" and returns relevant results about picking up prescriptions for another person.

3. **Impact on Search Quality**:
   - The implementation of BERT has significantly enhanced the quality and relevance of search results on Google.
  
# History and Evolution of Transformer-Based Models

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/f3b76625-d93f-4454-9439-e5a0fae0f8f1)

1. **Original Transformer Paper (2017)**:
   - Proposed by a Google research team.
   - Initial challenge: Required labeled data, which was time-consuming and labor-intensive.

2. **ULMFiT Model**:
   - Proposed by Jeremy Howard and Sebastian Ruda.
   - Allowed the use of unlabeled data for training, leveraging large text corpora like Wikipedia.

3. **Early Transformer Models**:
   - **GPT (June 2018)**: Developed by OpenAI, it was the first pre-trained transformer model, fine-tuned for various NLP tasks.
   - **BERT (Late 2018)**: Developed by Google, BERT stands for Bidirectional Encoder Representations from Transformers, demonstrated in production use by Google.

4. **Subsequent Developments**:
   - **GPT-2 (February 2019)**: Released by OpenAI, notable for its larger size and withheld details due to ethical concerns.
   - **BART (2019)**: Released by Facebook AI Research.
   - **T5 (2019)**: Released by Google.
   - **DistilBERT (2019)**: Released by Hugging Face, a smaller, faster version of BERT with 95% of its performance and 40% reduced size.

5. **GPT-3 (May 2020)**:
   - Released by OpenAI, known for generating high-quality English text.
   - OpenAI provided detailed papers but did not release the dataset or model weights.

6. **EleutherAI's Contributions**:
   - Focused on open-source language models.
   - **GPT-Neo (March 2021)**: 2 billion parameters.
   - **GPT-J (Mid-2021)**: 6 billion parameters.
   - **GPT-NeoX (February 2022)**: 20 billion parameters.

7. **Parameter Growth**:
   - Language models have exponentially increased in size.
   - **BERT**: 110 million parameters.
   - **BERT Large**: 340 million parameters.
   - **Largest GPT-2**: 1.5 billion parameters.
   - **Largest GPT-3**: 175 billion parameters.

8. **Trend**:
   ![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/9a99e7fb-e947-4fc2-ae85-d9ff816a6717)

   - Over the years, transformer-based language models have consistently grown larger, with a log-scale increase in the number of parameters.

# Title: **Transfer Learning in Deep Learning: Pretraining and Fine-Tuning**

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/f831c05e-3c71-4951-8e0b-cd70edff659c)

1. **Definition and Components**:
   - Transfer learning consists of two main steps: pretraining and fine-tuning.
   - It starts with a model architecture with random weights and no initial knowledge of language.

2. **Pretraining**:
   - Involves training the model on large datasets (e.g., Wikipedia) using significant computational resources (hundreds to thousands of GPUs or TPUs).
   - This phase results in a model with a deep understanding of the language.

3. **Fine-Tuning**:
   - Uses the pretrained model and adjusts it for specific tasks (e.g., text classification, named entity recognition) using labeled data.
   - Fine-tuning typically yields better accuracy than training a model from scratch.

4. **Examples of Pretraining Tasks for BERT**:
   - Masked Language Modeling: Predicting masked words in a sentence.
   - Next Sentence Prediction: Determining if one sentence logically follows another.

5. **Training Specifications for Notable Models**:
   ![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/cb5f3945-7a11-4fd1-948b-efcbc4109e34)

   - **BERT (2018)**:
     - 109 million parameters.
     - Trained for 12 days on TPUs using 16 GB of data and 250 billion tokens.
     - Data sources: Wikipedia and BookCorpus.
   - **RoBERTa (2019)**:
     - 125 million parameters.
     - Trained in 1 day using 1,024 GPUs with 160 GB of data and 2 trillion tokens.
     - Additional data sources: Common Crawl news, OpenWebText, Common Crawl stories.
   - **GPT-3 (2020)**:
     - 175 billion parameters.
     - Trained for approximately 34 days on 10,000 GPUs with 4,500 GB of data and 300 billion tokens.
     - Data sources: Wikipedia, Common Crawl, WebText2, Books1, Books2.

7. **Benefits of Transfer Learning**:
   - Requires less time and fewer resources to fine-tune compared to pretraining.
   - Does not need massive datasets for fine-tuning specific tasks.
   - Achieves excellent results, similar to early successes in computer vision with ImageNet.

8. **Conclusion**:
   - Transfer learning's pretraining and fine-tuning are powerful techniques in NLP, significantly enhancing model performance and efficiency.
  
# Title: **Overview of Transformer Architecture**

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/b7d8e0de-d989-4670-b04d-53119aa3ee9a)

1. **Transformer Architecture Basics**:
   - Divided into two main components: encoder and decoder.
   - Encoders and decoders can be used independently or together depending on the task.

2. **Encoder**:
   - Processes the input sequence (e.g., an English sentence "I like NLP").
   - Consists of six encoder layers.

3. **Decoder**:
   - Generates the output sequence (e.g., a German translation "ich mag NLP").
   - Also consists of six decoder layers.

4. **Encoder-Decoder Models**:
   - Suitable for generative tasks like translation and summarization.
   - Examples: Facebook's BART and Google's T5.

5. **Encoder-Only Models**:
   - Focused on understanding the input, useful for tasks like sentence classification and named entity recognition.
   - Examples: BERT, RoBERTa, DistilBERT.

6. **Decoder-Only Models**:
   - Specialized for generative tasks such as text generation.
   - Examples: GPT, GPT-2, GPT-3, and subsequent models.

7. **Summary**:
   - Transformers consist of encoders and decoders.
   - The choice of components (encoder, decoder, or both) depends on the specific NLP task.

# Title: **Understanding Self-Attention in Transformers**

![image](https://github.com/gjkaur/CoursesNotes/assets/36306330/f85a271d-cf1e-4ce1-867a-5e2ba8b53c0e)

1. **Role of Self-Attention**:
   - Enables the model to determine contextual relationships between words in a sentence.
   - Example: In "The monkey ate that banana because it was too hungry," self-attention helps identify that "it" refers to "monkey."

2. **Mechanism of Self-Attention**:
   - Uses embeddings of all other words in the sentence.
   - Assigns weights to words based on their relevance to the target word ("it" in the example).
   - Higher weights indicate stronger relevance.

3. **Query, Key, and Value Vectors**:
   - Word embeddings are projected into three vector spaces: query, key, and value.
   - These projections help in calculating attention weights.

4. **Calculating Attention Weights**:
   - The query and key vectors are used to calculate scores for each word.
   - The dot product of the query and key vectors determines the similarity.
   - Similar queries and keys yield higher dot products, indicating higher relevance.

5. **Scaling and Softmax**:
   - The dot product is scaled by dividing by the square root of the dimension (N) to control its size.
   - Softmax function converts the scaled scores into probabilities.

6. **Weighted Sum**:
   - Each value vector is multiplied by its corresponding softmax score.
   - The weighted value vectors are summed to produce the self-attention output for each word.

7. **Application**:
   - Self-attention is applied to every word in the sentence.
   - Allows different weights to be assigned to words, enhancing the model's understanding of context and relationships.
