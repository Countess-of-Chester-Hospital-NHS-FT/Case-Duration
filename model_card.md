<!-- 
# NHS England Template Model Card

## What are model cards?

The main purpose of a model card is to display the essential components of models in a structured way.

Model cards are intended to be *short* documents that contain non-technical and technical content for a variety of audiences. They should be succinct, well-structured and easily digestible by the reader.

Model cards are one approach to increasing transparency between developers, users and stakeholders about details such as any data used, chosen methodology and performance. They are an example of a transparency tool which aims to standardise ethical practice and reporting[^1].

A model card is *not a replacement for detailed documentation* covering quality assurance, testing, bias analysis and so on. High-level overview of these areas can be included on a model card where relevant, but any further detail should be accessible from the model card via links to full documentation.

## When to use this template

This template can be used at any point during the model development lifecycle. It is recommended to start using and populating a model card as soon as model development begins because it serves as a useful piece of internal documentation.

Once a model is finalised, ready for deployment or put into production, a fully completed model card should be made available alongside the code used for model training.

To achieve full transparency, the code and model card should be published.

## How to use this template

Having a template is a helpful starting place for those who may not be familiar with model cards and provides *an indication rather than a specification* of what information model developers should consider providing. The specific content and structure *should be flexed* to accommodate specific use cases and models, with the above purpose in mind.

### To copy the model card template, you can either:
1. Download the repository and then copy the 'model_cards' folder into your own repo
2. Download the *model_card_template.md* file if you only want the markdown template and not the full folder structure.

You can then start populating the model card with information about the models in use in your system/service/algorithm. 

Storing the model card in a repository alongside the code used to train, test and evaluate models will help to keep documentation up to date as both the model and it's documentation can be updated in parallel. If the code is published, then the model card can be made public through the code publication process.

### [Go to the template](model_cards/model_card_template.md)

## Inspiration

Model cards were first explored in a Google research paper[^1] in 2018 and are now widely used.

The initial design and set-up of this model card template took inspiration from:

* [Google](https://modelcards.withgoogle.com/object-detection)
* [Kaggle](https://www.kaggle.com/code/var0101/model-cards)
* [HuggingFace](https://huggingface.co/docs/hub/en/model-cards)

The use of model cards was recommended in the UK Government's Generative AI Framework for HMG[^2], published in January 2024. This specifically references the Google paper[^1] from which most inspiration was drawn.

## Content

This template repository includes:

1. [The model card template](model_cards/model_card_template.md)
    1. Specification
    2. Intended Use
    3. Data
    4. Methodology and Training
    5. Evaluation and Performance
2. Visuals folder to store visualisations (such as a confusion matrix) for use in a model card

[^1]: [Model cards for model reporting](https://arxiv.org/pdf/1810.03993)
[^2]: [Generative AI Framework for HMG](https://www.gov.uk/government/publications/generative-ai-framework-for-hmg)
-->

# {Model Name} Model Card

## Specification

_This section contains top-level information about what the model is. The details of this section are in a table format so as to quickly give the reader the relevant information._

|  |  |
| ---- | ---- |
| **Description:** | General overview of the model, what it does, and motivation for developing it. No more than one paragraph|
| **Model Type:** | Specify the type of model (e.g., clustering, time series, deep learning models, reinforcement learning, supervised learning, ensemble, hybrid, etc.) and describe its main characteristics|
| **Developed By:** | Name of the developer or development team |
| **Launch Date:** | Expected or Launch Date |
| **Version** | As in model registry |

## Intended Use

_This section goes into more detail about why the model was built and who it was for. This section can go into more depth than previous, but should still be as succinct as possible._

**Development Background:** A brief description of the model's development purpose and context, this is a chance to expand on the description, any additional background information on the model's focus area, who this originally created for and how the model fits into the wider decision making process.

**Scope:** Specific application or functionality of the model.

**Intended Users:** Describe who the model is intended for.

**Use cases out of scope:** Describe any potential use cases out of scope.

## Data

_This section should provide basic details about the data used in and by the model. Any more detail should be provided in supplementary information, such as a DataSheet._

**Data Overview:** A brief overview of where the data comes from, the type of data and it's coverage etc.

**Sensitive Data:** How is any sensitive data handled?

**Pre-processing and cleaning:** How was the data cleansed?, was it's suitability verified before use?, why these datasets?, are they representative?

**Data Split:** 
| Type | Split | Description |
| ---- | ---- | ---- |
| **Training:** | x% | Use this space to describe the training data, expanding on the data overview |
| **Testing:** | x% | Use this space to describe the testing data, expanding on the data overview |
| **Validation:** | x% | Use this space to describe the validation data, expanding on the data overview |

## Methodology and Training

_This section should explain how the model was built to do what it does, including any justification where required. Lengthy debate or detail of particular methodology should be provided in user documentation and not in the model card._

**Model Type:** Specify the type of model (e.g., clustering, time series, deep learning models, reinforcement learning, supervised learning, ensemble, hybrid, etc.) and describe its main characteristics.

**Models Used:** What models were used (e.g. embedding x + classifier y)

**Justification:** Explain here why this approach was taken over another.

**Algorithm Details:** Describe the main steps when data is sent to the model for inference, e.g. how is text data converted into a binary class?

**Feature Engineering:** Details on the process of creating and selecting features if relevant.

**Alternative Methods Considered:**
1. Option B
2. Option C etc.

### Training Methods:

**Training Process:** Description of the model training process. Link to relevant code or repository used to create the model.

**Hyperparameter/Fine Tuning:** _(optional)_ Detail any additional  processes to select appropriate hyperparameters or other types of fine tuning.

## Evaluation and Performance

_This section details how well the model peforms._

_Note: The metrics and visual in this section is an example only and is not intended to indicate the evaluation methods and visuals you should use. Please adjust the types of metrics and visuals appropriately to suit requirements._

### Model Evaluation

**Evaluation Process:** How the model is evaluated.

**Evaluation Focus:** _(optional)_ If relevant, provide detail and an explanation for why a particular metric (e.g. F1 score), or performance for/across specific data subset/s was prioritised.

**Performance breakdown:** Details on the model's performance across different subsets or categories of data.

**Metrics:** Use the table below for performance metrics, adjust as appropriate.

| Metric | Value |
| ---- | ---- |
| F1 Score | 0.00 |
| Precision | 0.00 |
| Recall | 0.00 |

_The below confusion matrix is an example of how to include an image in this model card - visuals can be helpful but it is not a requirement to use a confusion matrix._

<img src="{./path/to/your/image/folder/filename.png" width="400"/><br>

**Performance in Deployment**: Detail the performance of the model in deployment once monitoring is completed. State any considerations taken for the model inference time and provide the results in brief.

### Ethical Considerations

**Bias and fairness analysis:** Describe the approach for analysing error and the insights gained from them.

**Implications for human safety:** Potential impact on human life, potential harm and safety and mitigation.

### Caveats

**Caveats and Limitations:** Describe here any important information or limitations (e.g.how will an NLP model handle changes in human language like new slang).
