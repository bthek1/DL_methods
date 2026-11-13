# DL Methods

How deep learning models are actually built: PyTorch from tensors up, the architecture families, and the LLM and RAG systems assembled on top.

Author

Benedict Thekkel

Where [DL Tasks](https://bthek1.github.io/DL_tasks/) answers “which model solves this problem”, this site answers “how does the model work and how do I build one”. It runs from PyTorch fundamentals through the classical architecture families to the retrieval and agent systems that current work is mostly made of.

The two halves are quite different in character. The PyTorch and architecture pages are long, worked, and stable; the LLM and RAG folders are a large set of focused pages, one per stage of a pipeline, reflecting that this is where the field is moving fastest.

------------------------------------------------------------------------

## PyTorch

The foundation, and the longest pages on the site.

| Page | Covers |
|----|----|
| [PyTorch Basics](torch_basics.llms.md) | Tensors and operations, at 218 cells the most thorough page here |
| [Model Creation](model_creation.llms.md) | Assembling modules into a network |
| [Training Loop](model_training_loop.llms.md) | The loop itself, written out rather than hidden behind a framework |
| [Datasets and DataLoaders](datasets_and_dataloaders.llms.md) | Getting data in efficiently, which is usually the bottleneck |
| [Transforms](transforms.llms.md) | Preprocessing and augmentation |
| [Save and Load](save_and_load.llms.md) | Checkpointing, and the distinction between weights and whole models |
| [timm](timm_models_torch.llms.md) | Loading pretrained vision models from the library that collects them |
| [Transfer Learning](transfer_learning_resnet18.llms.md) | Adapting a pretrained ResNet-18 rather than starting from noise |
| [Evaluation Metrics](evaluation_metrics.llms.md) | How performance is measured, with detection metrics in particular |
| [Model Diagrams](model_diagrams.llms.md) | Rendering an architecture as a picture |
| [TensorFlow](tensorflow.llms.md) | The alternative framework, covered briefly through its graph model |
| [PyTorch Setup](pytorch_setup.llms.md) | Marked OLD; kept for reference |

------------------------------------------------------------------------

## Architecture Families

One page per family, each covering the idea, the mechanics, and where it is still the right choice.

| Page | Covers |
|----|----|
| [Overview](Models_Types/overview.llms.md) | The map of everything below |
| [FNN](Models_Types/fnn.llms.md) | Feedforward networks, the starting point |
| [CNN](Models_Types/cnn.llms.md) | Convolutional networks, and why they suit images |
| [RNN](Models_Types/rnn.llms.md) | Recurrence for sequences |
| [LSTM](Models_Types/lstm.llms.md) | Gating that fixes the vanishing gradient |
| [GRU](Models_Types/gru.llms.md) | The simplified version of the same idea |
| [Transformers](Models_Types/transformers.llms.md) | Attention, which displaced most of the above |
| [Autoencoders](Models_Types/autoencoders.llms.md) | Learning a compressed representation |
| [GAN](Models_Types/gan.llms.md) | Two networks trained against each other |
| [RBM](Models_Types/rbm.llms.md) | Restricted Boltzmann machines |
| [DBN](Models_Types/dbn.llms.md) | Deep belief networks stacked from them |

------------------------------------------------------------------------

## Building Models by Hand

Implementations written from scratch, which is the fastest way to find out what the abstractions were hiding.

| Page | Covers |
|----|----|
| [MNIST from Scratch in NumPy](Models_Types/Model%20Design/simple-mnist-nn-from-scratch-numpy.llms.md) | A neural network with no framework at all, forward and backward pass written out |
| [Simple CNN in PyTorch](Models_Types/Model%20Design/simple_cnn_in_pytorch.llms.md) | The same problem with the framework back on |
| [ResNet-18 from Scratch](Models_Types/Model%20Design/resnet18_with_scratch.llms.md) | Building the residual architecture layer by layer |
| [ResNet-18 with fastai](Models_Types/Model%20Design/resnet18_with_fastai.llms.md) | The same result in a fraction of the code, as the contrast |

------------------------------------------------------------------------

## Computer Vision

| Page | Covers |
|----|----|
| [Image Classification](Computer_Vision/image_classification.llms.md) | One label per image |
| [Multi-Label Classification](Computer_Vision/image_multi_classification.llms.md) | Several labels at once |
| [Object Detection](Computer_Vision/object_detection.llms.md) | Locating and labelling |
| [Semantic Segmentation](Computer_Vision/semantic_segmentation.llms.md) | Classifying each pixel |
| [Instance Segmentation](Computer_Vision/instance_segmentation.llms.md) | Separating individual objects of the same class |
| [Panoptic Segmentation](Computer_Vision/panoptic_segmentation.llms.md) | Both of the above at once |
| [Car Detection](Car_detection/car_detection.llms.md) | A worked application |

------------------------------------------------------------------------

## LLMs

| Page | Covers |
|----|----|
| [LLM Class](LLM/llmclass.llms.md) | Wrapping model access behind one interface |
| [Ollama](LLM/ollama.llms.md) | Running models locally |
| [OpenAI SDK](LLM/openai_sdk.llms.md) | Calling a hosted API directly |
| [LangChain](LLM/langchain.llms.md) | The orchestration framework |
| [OpenAI with LangChain](LLM/openai_langchain.llms.md) | The two combined |

------------------------------------------------------------------------

## LLM Pipeline Patterns

Seven pages, one per pattern, ordered from least to most autonomous. Start with the overview.

| Page | Covers |
|----|----|
| [Pipeline Designs](LLM/LLM_Pipelines/llm_pipelines.llms.md) | The overview and how the patterns relate |
| [Prompt Chaining](LLM/LLM_Pipelines/prompt_chaining.llms.md) | Output of one call feeding the next |
| [Routing](LLM/LLM_Pipelines/routing.llms.md) | Classifying the request, then dispatching to the right handler |
| [Parallelization](LLM/LLM_Pipelines/parallelization.llms.md) | Running independent calls at once and combining them |
| [Orchestrator-Workers](LLM/LLM_Pipelines/orchestrator_workers.llms.md) | A planner decomposing work across workers |
| [Evaluator-Optimizer](LLM/LLM_Pipelines/evaluator_optimizer.llms.md) | Generating, critiquing, and revising in a loop |
| [Autonomous Agents](LLM/LLM_Pipelines/autonomous_agents.llms.md) | The model choosing its own next step, and the cost of that |

------------------------------------------------------------------------

## RAG

The most systematic part of this site: nineteen pages walking the retrieval-augmented generation pipeline stage by stage, then the concerns that sit across all of it. The retrieval and re-ranking pages restrict themselves to open-source components.

**The pipeline, in order**

| Page | Covers |
|----|----|
| [RAG](LLM/RAG/rag.llms.md) | The overview, and what the rest of the folder expands |
| [Document Pre-processing](LLM/RAG/document_preprocessing.llms.md) | Getting source material into a usable state |
| [Chunking](LLM/RAG/chunking.llms.md) | Splitting documents, which sets the ceiling on everything downstream |
| [Embedding](LLM/RAG/embedding.llms.md) | Turning chunks into vectors |
| [Indexing](LLM/RAG/indexing.llms.md) | Storing them so they can be searched |
| [Query Transformation](LLM/RAG/query_transformation.llms.md) | Rewriting the question before searching with it |
| [Retrieval](LLM/RAG/retrieval.llms.md) | Dense, sparse, and hybrid search |
| [Re-ranking](LLM/RAG/re-ranking.llms.md) | Reordering candidates before they reach the model |
| [Context Assembly](LLM/RAG/context_assembly.llms.md) | Deciding what actually goes in the prompt |
| [Generation](LLM/RAG/generation.llms.md) | The answer, and grounding it in what was retrieved |

**Across the pipeline**

| Page | Covers |
|----|----|
| [Evaluation](LLM/RAG/evaluation.llms.md) | Scoring a pipeline you cannot otherwise improve |
| [Observability](LLM/RAG/observability.llms.md) | Seeing what it did in production |
| [Caching](LLM/RAG/caching.llms.md) | Cutting cost and latency |
| [Index Lifecycle](LLM/RAG/index_lifecycle.llms.md) | Keeping the index current as documents change |
| [Access Control](LLM/RAG/access_control.llms.md) | Making sure retrieval respects who is asking |
| [Guardrails](LLM/RAG/guardrails.llms.md) | Constraining what comes out |

**Variants**

| Page | Covers |
|----|----|
| [Agentic RAG](LLM/RAG/agentic_rag.llms.md) | Letting the model decide when and what to retrieve |
| [Graph RAG](LLM/RAG/graph_rag.llms.md) | Retrieval over a knowledge graph instead of a flat index |
| [Multimodal RAG](LLM/RAG/multimodal_rag.llms.md) | Retrieval across images and documents, not only text |

------------------------------------------------------------------------

## Not Covered Yet

- **No fine-tuning.** Nothing on LoRA, QLoRA, adapters, or full fine-tuning, which is the most obvious gap given the depth of everything around it.
- **Nothing on training at scale**: no distributed training, mixed precision as a subject, or gradient accumulation.
- **No diffusion model page**, despite GANs and autoencoders both being covered and diffusion having displaced them for generation.
- **No serving or inference optimisation.** Quantisation, batching, and throughput are absent.
- **[Indexing](LLM/RAG/indexing.llms.md) opens with a `models.py` heading**, which suggests it was drafted from a specific project rather than written as a reference page.
- **[Car Detection](Car_detection/car_detection.llms.md) is thin** at six cells, and sits alone in its own folder.
- The vision pages here overlap the task pages in [DL Tasks](https://bthek1.github.io/DL_tasks/); this site is the how, that one is the which, but the boundary is not always obvious.

------------------------------------------------------------------------

Back to top
