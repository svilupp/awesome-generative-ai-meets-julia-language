<!-- omit from toc -->
# Awesome Generative AI Meets Julia Programming Language[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Comprehensive guide to generative AI projects and resources in Julia.

[<img src="https://github.com/JuliaLang/julia/raw/master/doc/src/assets/logo.svg" align="right" width="100"/>](http://julialang.org)

Julia is a high-level, high-performance dynamic language for technical computing. 

<br>

<!-- omit from toc -->
## Contents
- [Generative AI Projects and Julia](#generative-ai-projects-and-julia)
- [Models](#models)
- [API SDKs](#api-sdks)
  - [Model Providers](#model-providers)
  - [Cloud Services Providers](#cloud-services-providers)
- [Packages](#packages)
- [JLL Packages](#jll-packages)
- [Benchmarks/Comparisons](#benchmarkscomparisons)
- [Applications/Products](#applicationsproducts)
- [Tutorials/learning](#tutorialslearning)
- [Noteworthy Mentions](#noteworthy-mentions)
  - [Local Deployments](#local-deployments)
  - [Generative AI - Previous Generation](#generative-ai---previous-generation)
- [Must-Know Python Projects](#must-know-python-projects)
- [Other Awesome Lists](#other-awesome-lists)

## Generative AI Projects and Julia

- [JuliaGenAI Organization](http://juliagenai.org/) - A Github organization and a community of Julia developers and researchers working on generative AI.

> [!IMPORTANT]
> **Google Summer of Code 2024 with JuliaGenAI**
>
> Leap into JuliaGenAI's Google Summer of Code: Unleash your potential, learn about practical applications, and make pivotal contributions in the exciting world of generative AI with Julia—a chance to grow, impact, and thrive in an expanding ecosystem. [Learn more](https://julialang.org/jsoc/gsoc/juliagenai/)

## Models

Build, train, and deploy Large language models in Julia.

- [Flux.jl](https://github.com/FluxML/Flux.jl) - Flux is a machine learning library for Julia that is flexible and allows building complex models. However, at the time of writing, I'm not aware of any Large Language Models (LLMs) that have been implemented and trained in Flux.
- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) - Transformers.jl is a Julia package that provides a high-level API for using pre-trained transformer models. It also allows to download any models from Hugging Face hub with `@hgf_str` macro string.
- [Pickle.jl](https://github.com/chengchingwen/Pickle.jl) - Great package for loading Pytorch weights into Julia (if you want to implement models yourself).
- [BytePairEncoding.jl](https://github.com/chengchingwen/BytePairEncoding.jl) - Pure Julia implementation of Byte Pair Encoding (BPE) algorithm. It's used by Transformers.jl to tokenize text.
- [Llama2.jl](https://github.com/cafaxo/Llama2.jl) - Llama2.jl provides simple code for inference and training of llama2-based language models based on [llama2.c](https://github.com/karpathy/llama2.c). It supports loading quantized weights in GGUF format (`q4_K_S` variant). Other similar projects: [LanguageModels.jl](https://github.com/rai-llc/LanguageModels.jl)
- [Llama.jl](https://github.com/marcom/Llama.jl/) - Julia interface to llama.cpp, a C/C++ library for running language models locally. Supports a wide range of models.


## API SDKs

### Model Providers

Access Generative AI models via official APIs.

- [OpenAI.jl](https://github.com/JuliaML/OpenAI.jl) - A community-maintained Julia wrapper to the OpenAI API. 

### Cloud Services Providers

Access Generative AI models via SDKs of popular cloud service providers.

- [GoogleCloud.jl](https://github.com/JuliaCloud/GoogleCloud.jl) - SDK for Google Cloud. There is an [open PR](https://github.com/JuliaCloud/GoogleCloud.jl/pull/57) to enable Vertex AI endpoints.

## Packages

- [ReplGPT.jl](https://github.com/ThatcherC/ReplGPT.jl) - Brings ChatGPT interface as a Julia REPL mode.
- [HelpGPT.jl](https://github.com/FedeClaudi/HelpGPT.jl) - Calls ChatGPT to explain any errors in Julia code.
- [GenGPT3.jl](https://github.com/probcomp/GenGPT3.jl) - A [Gen.jl](https://www.gen.dev/) generative function that wraps the OpenAI API.
- [GPTCodingTools.jl](https://github.com/svilupp/GPTCodingTools) - Code generation tool for Julia language with useful prompt templates and self-healing features (ala OpenAI Code Interpreter). It does work, but development has been abandoned. (Disclaimer: I'm the author of this package.)
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - Helps with everyday applications of Large Language Models in Julia by wrapping coming APIs, re-using prompts via templates, and enabling easy transition between different model providers (eg, OpenAI, Ollama). (Disclaimer: I'm the author of this package.)
- [LLMTextAnalysis.jl](https://github.com/svilupp/LLMTextAnalysis.jl) - Leverage Large Language Models to uncover, evaluate, and label themes/concepts/spectra in large document collections. (Disclaimer: I'm the author of this package.)

## JLL Packages

[JLLs](https://docs.binarybuilder.org/stable/jll/) are prebuilt libraries and executables to easily install and call non-Julia projects (eg, C/C++). Often they are the first step towards a Julia package with an idiomatic interface.

- [llama_cpp_jll.jl](https://juliahub.com/ui/Packages/General/llama_cpp_jll/) - JLL package for [llama.cpp](https://github.com/ggerganov/llama.cpp), the best interface for quantized llama2-style models.

## Benchmarks/Comparisons
- [Julia LLM Leaderboard](https://github.com/svilupp/Julia-LLM-Leaderboard) - Comparison of Julia language generation capabilities of various Large Language Models across a range of tasks. Visit if you want help choosing the right model for your application.

## Applications/Products

Applications and products that "work" with Julia language.

- [GitHub Copilot](https://github.com/features/copilot) - Excellent inline suggestions with the help of OpenAI models. It works extremely well with Julia language for repetitive tasks one line at a time, but larger code chunks are rarely correct.
- [Codium.ai](https://codium.ai/) - Great IDE or VSCode plugin for code analysis, suggestion and generation of test suites. Although the tests are written more in the style of Pytest rather than idiomatic Julia. It has a free tier.
- [Replit](https://replit.com/ai) - Replit's REPL is powered by an in-house model that supports Julia language.
- [Codeium](https://codeium.com/) - Free alternative to GitHub Copilot with extensions for most editors.

Julia-affiliated applications and products using LLMs

- [JuliaHub AskAI](https://juliahub.com/ui/AskAI) - AskAI is a [JuliaHub's](https://juliahub.com) RAG (Retrieval Augmented Generation) application that allows users to ask questions about the Julia language and its ecosystem. It is free, but you need to be logged in to JuliaHub to use it.
- [Genie UI Assistant](https://forem.julialang.org/pgimenez/introducing-genie-ui-assistant-the-ai-powered-ui-builder-for-genie-apps-3jpe) - Genie UI Assistant is a GPT-4 powered 
UI builder helping [Genie.jl's](https://github.com/GenieFramework/Genie.jl) users create UIs faster using natural language.
- [Comind](https://comind.me) - A social network, messaging, and LLM interface built in Julia.

## Tutorials/learning

- [Tutorial for using LLMs with Transformers.jl](https://info.juliahub.com/large-language-model-llm-tutorial-with-julias-transformers.jl) - A brief tutorial on how to use Transformers.jl to access LLMs from HuggingFace Hub.
- [Building a RAG Chatbot over DataFrames.jl Documentation - Hands-on Guide](https://forem.julialang.org/svilupp/building-a-rag-chatbot-over-dataframesjl-documentation-hands-on-guide-449m) - A hands-on guide on how to build a RAG chatbot over DataFrames.jl documentation using only minimal dependencies.
- [GenAI Mini-Tasks: Extracting Data from (.*)? Look No Further!](https://forem.julialang.org/svilupp/genai-mini-tasks-extracting-data-from-look-no-further-2m32) - A tutorial on structured data extraction. A part of a larger series of tutorials on small tasks that can be done with GenAI.

## Noteworthy Mentions

Some of the below projects are not necessarily Julia-specific, but noteworthy mentions in the generative AI space and interesting for Julia developers.

### Local Deployments

- [Ollama](https://github.com/jmorganca/ollama) - The best option for those looking to host a Large Language Model locally. Simply start the server and send the requests with [HTTP.jl](https://github.com/JuliaWeb/HTTP.jl).
- [LM Studio](https://lmstudio.ai/) - A desktop app for hosting and interacting with LLMs locally. It's a great option for those who want to use LLMs without coding. It's free for **personal use**.

### Generative AI - Previous Generation

- [GenerativeModels.jl](https://github.com/aicenter/GenerativeModels.jl) - Useful library to train more traditional generative models like VAEs. It's built on top of Flux.jl.

## Must-Know Python Projects

Python is on the leading edge of the generative AI revolution. Fortunately, we have [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) allowing us to easily call all the below Python packages.

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) - The most popular library for accessing LLMs and other models. It can be mostly used via Transformers.jl (see above).
- [LangChain](https://github.com/langchain-ai/langchain) - The best option for building applications on top of LLMs (eg, Chains, Agents). It has a lot of adapters for common models, databases, and other services.
- [Llama Index](https://github.com/run-llama/llama_index) - Similar to LangChain but with a focus on data-centered applications like RAG.
- [Instructor](https://github.com/jxnl/instructor) - Simple yet powerful structured extraction framework on top of OpenAI API. Excellent to understand the power of function calling API together with Pydantic.
- [Marvin](https://github.com/prefecthq/marvin) - Powerful building blocks to quickly build AI applications and expose them via a production-ready API.
- [Open Interpreter](https://github.com/KillianLucas/open-interpreter) - Let LLMs run code on your computer (eg, Python, JavaScript, Shell, and more). An open-source local alternative to OpenAI Code Interpreter.

## Other Awesome Lists

- [Awesome Generative AI](https://github.com/steven2358/awesome-generative-ai) - Great list for all things generative AI. An inspiration for this list!
