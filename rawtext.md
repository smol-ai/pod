# [AINews] Liquid Foundation Models: A New Transformers alternative + AINews Pod 2

<<<<<<< HEAD
**Adaptive computational operators are all you need.**
=======
# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**Apple's AI Announcements and Industry Reactions**

- Apple unveiled new AI features for iOS 18, including visual intelligence capabilities and improvements to Siri. [@swyx](https://twitter.com/swyx/status/1833231875537850659) noted that Apple has potentially "fixed Siri" and introduced a video understanding model, beating OpenAI to the first AI phone. The new features include mail and notification summaries, personal context understanding, and visual search integration.

- The new iPhone camera button is seen as prime real estate, with OpenAI/ChatGPT and Google search as secondary options to Apple's visual search. [@swyx](https://twitter.com/swyx/status/1833234781221622022) highlighted that the camera can now add events to the calendar, with processing done on-device and in the cloud.

- Some users expressed disappointment with Apple's recent innovations. [@bindureddy](https://twitter.com/bindureddy/status/1833248496948023753) mentioned that there hasn't been a compelling reason to upgrade iPhones in recent years, noting that Apple Intelligence seems similar to Google Lens, which was released years ago.

**AI Model Developments and Controversies**

- The AI community discussed the Reflection 70B model, with mixed reactions and controversies. [@BorisMPower](https://twitter.com/BorisMPower/status/1833187250420453716) stated that the model performs poorly, contrary to initial claims. [@corbtt](https://twitter.com/corbtt/status/1833209248236601602) announced an investigation into the model's performance, working with the creator to replicate the reported results.

- [@DrJimFan](https://twitter.com/DrJimFan/status/1833160432833716715) highlighted the ease of gaming LLM benchmarks, suggesting that MMLU or HumanEval numbers are no longer reliable indicators of model performance. He recommended using ELO points on LMSys Chatbot Arena and private LLM evaluation from trusted third parties for more accurate assessments.

- The AI research community discussed the importance of evaluation methods. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1833136159209263552) announced the open-sourcing of "Lighteval," an evaluation suite used internally at Hugging Face, to improve AI benchmarking.

**AI in Research and Innovation**

- A study comparing LLM-generated research ideas to those of human experts found that AI-generated ideas were judged as more novel. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833228667641561495) shared key insights from the paper, noting that LLM-generated ideas received higher novelty scores but were slightly less feasible than human ideas.

- [@omarsar0](https://twitter.com/omarsar0/status/1833234005917065274) discussed a new paper on in-context learning in LLMs, highlighting that ICL uses a combination of learning from in-context examples and retrieving internal knowledge.

- [@soumithchintala](https://twitter.com/soumithchintala/status/1833177895734267987) announced the release of RUMs, robot models that perform basic tasks reliably with 90% accuracy in unseen, new environments, potentially unlocking longer trajectory research.

**AI Tools and Applications**

- [@svpino](https://twitter.com/svpino/status/1833233962757722268) shared an example of AI's capability to turn complex documents into interactive graphs within seconds, emphasizing the rapid progress in this area.

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1833170410135056477) announced SVG support for FastHTML, allowing for the creation of Mermaid editors.

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833104751979794610) discussed DynamiqAGI, a comprehensive toolkit for addressing various GenAI use cases and building compliant GenAI applications on personal infrastructure.

**AI Ethics and Safety**

- [@fchollet](https://twitter.com/fchollet/status/1833171952070238240) argued that excessive anthropomorphism in machine learning and AI is responsible for misconceptions about the field.

- [@ylecun](https://twitter.com/ylecun/status/1833130597176205746) discussed the historical role of armed civilian militias in bringing down democratic governments and supporting tyrants, drawing parallels to current events.

**Memes and Humor**

- [@sama](https://twitter.com/sama/status/1833227974554042815) shared a humorous analogy: "if you strap a rocket to a dumpster, the dumpster can still get to orbit, and the trash fire will go out as it leaves the atmosphere," suggesting that while this contains important insights, it's better to launch nice satellites instead.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Reflection 70B: From Hype to Controversy**

- **Smh: Reflection was too good to be true - reference article** ([Score: 42, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fd2f7m/smh_reflection_was_too_good_to_be_true_reference/)): The performance of **Reflection 70B**, a recently lauded open-source AI model, has been **questioned** and the company behind it **accused of fraud**. According to a [VentureBeat article](https://venturebeat.com/ai/new-open-source-ai-leader-reflection-70bs-performance-questioned-accused-of-fraud/), concerns have been raised about the legitimacy of the model's reported capabilities and benchmarks. The situation has sparked debate within the AI community about the **verification of AI model performance claims**.

- **Out of the loop on this whole "Reflection" thing? You're not alone. Here's the best summary I could come up.** ([Score: 178, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fd75nm/out_of_the_loop_on_this_whole_reflection_thing/)): The post summarizes the **Reflection 70B controversy**, where **Matt Shumer** claimed to have created a revolutionary AI model using "**Reflection Tuning**" and **Llama 3.1**, surpassing established models like **ChatGPT**. Subsequent investigations revealed that the public API was likely a wrapper for **Claude 3.5 Sonnet**, while the released model weights were a poorly tuned **Llama 3 70B**, contradicting Shumer's claims and raising concerns about potential fraud and undisclosed conflicts of interest with **Glaive AI**.
  - **Matt Shumer's** claims about the **Reflection 70B** model were met with skepticism, with users questioning how it's possible to "accidentally" link to **Claude** while claiming it's your own model. Some speculate this could be a case of fraud or desperation in the face of a tightening AI funding landscape.
  - The incident drew comparisons to other controversial AI projects like the **Rabbit device** and "**Devin**". Users expressed growing skepticism towards **OpenAI** as well, questioning the company's claims about voice and video capabilities and noting key employee departures.
  - Discussions centered on potential motives behind Shumer's actions, with some attributing it to stupidity or narcissism rather than malice. Others speculated it could be an attempt to boost **Glaive AI** or secure venture capital funding through misleading claims.

- **Reflection and the Never-Ending Confusion Between FP16 and BF16** ([Score: 42, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fcjtpo/reflection_and_the_neverending_confusion_between/)): The post discusses a **technical issue** with the **Reflection 70B** model uploaded to **Hugging Face**, which is **underperforming** compared to the baseline **LLaMA 3.1 70B**. The author explains that this is likely due to an **incorrect conversion** from **BF16** (used in LLaMA 3.1) to **FP16** (used in Reflection), which causes significant information loss due to the incompatible formats (**5-bit exponent and 10-bit mantissa** for FP16 vs **8-bit exponent and 7-bit mantissa** for BF16). The post strongly advises against using **FP16** for neural networks or attempting to convert **BF16 weights to FP16**, as it can severely degrade model performance.
  - **BF16 to FP16 conversion** may not be as destructive as initially suggested. **llama.cpp** tests show the **perplexity difference** between BF16 and FP16 is **10x less** than FP16 to Q8, and most **GGUFs** on HuggingFace are likely based on FP16 conversion.
  - The discussion highlighted the importance of **Bayesian reasoning** when evaluating **Schumer's claims**, given previous misrepresentations about the base model, size, and open-source status. Some users emphasized the need to consider these factors alongside technical explanations.
  - Several users noted that most model **weights typically fall within [-1, 1]** range, making FP16 conversion less impactful. **Quantization** to **8 bits** or less per weight often results in negligible or reasonable accuracy loss, suggesting FP16 vs BF16 differences may be minimal in practice.


**Theme 2. AMD's UDNA: Unifying RDNA and CDNA to Challenge CUDA**


- **[AMD announces unified UDNA GPU architecture — bringing RDNA and CDNA together to take on Nvidia's CUDA ecosystem](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)** ([Score: 284, Comments: 90](https://reddit.com//r/LocalLLaMA/comments/1fcyap8/amd_announces_unified_udna_gpu_architecture/)): AMD unveiled its new **unified Data Center Next Architecture (UDNA)**, combining elements of **RDNA** and **CDNA** to create a single GPU architecture for both gaming and data center applications. This strategic move aims to challenge **Nvidia's CUDA** ecosystem dominance by offering a unified platform that supports **AI**, **HPC**, and **gaming** workloads, potentially simplifying development across different GPU types and increasing AMD's competitiveness in the GPU market.

**Theme 3. DeepSeek V2.5: Quietly Released Powerhouse Model**

- **[DeepSeek silently released their DeepSeek-Coder-V2-Instruct-0724, which ranks #2 on Aider LLM Leaderboard, and it beats DeepSeek V2.5 according to the leaderboard](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724)** ([Score: 183, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1fd6z0v/deepseek_silently_released_their/)): DeepSeek has quietly released **DeepSeek-Coder-V2-Instruct-0724**, a new coding model that has achieved the **#2 rank** on the **Aider LLM Leaderboard**. This model outperforms its predecessor, **DeepSeek V2.5**, according to the leaderboard rankings, marking a significant improvement in DeepSeek's coding capabilities.
  - **DeepSeek-Coder-V2** expands support from **86 to 338 programming languages** and extends context length from **16K to 128K**. The model requires **8x80GB cards** to run, with no lite version available for most users.
  - Users discussed version numbering confusion between DeepSeek's general and coding models. The new coder model (**0724**) outperforms **DeepSeek V2.5** on the **Aider LLM Leaderboard**, but V2.5 beats 0724 in most other benchmarks according to [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5).
  - Some users expressed interest in smaller, language-specific models for easier switching and interaction. DeepSeek typically takes about a month to open-source their models after initial release.

- **All of this drama has diverted our attention from a truly important open weights release: DeepSeek-V2.5** ([Score: 472, Comments: 95](https://reddit.com//r/LocalLLaMA/comments/1fclav6/all_of_this_drama_has_diverted_our_attention_from/)): The release of **DeepSeek-V2.5** has been overshadowed by recent AI industry drama, despite its potential significance as an **open GPT-4** equivalent. This new model, available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5), reportedly combines **general and coding capabilities** with upgraded **API and Web** features.
  - **DeepSeek-V2.5** received mixed reviews, with some users finding it **inferior to Mistral-Large** for creative writing and general tasks. The model requires **80GB*8 GPUs** to run, limiting its accessibility for local use.
  - Users reported issues running the model, including **errors in oobabooga** and problems with **cache quantization**. Some achieved limited success using **llama.cpp** with reduced context length, but performance was slow at **3-5 tokens per second**.
  - Despite concerns, some users found DeepSeek-V2.5 useful for adding variety to outputs and potentially solving coding problems. It's available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) and through a cost-effective [API](https://open-tone-changer.vercel.app/).


**Theme 4. Innovative Approaches to Model Efficiency and Deployment**

- **[Open Interpreter refunds all hardware orders for 01 Light AI device, makes it a phone app instead. App launches TODAY!](https://changes.openinterpreter.com/log/01-app)** ([Score: 42, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fczecj/open_interpreter_refunds_all_hardware_orders_for/)): Open Interpreter has **canceled** plans for its **01 Light AI hardware device**, opting instead to **launch a mobile app** that performs the same functions. This decision appears to be influenced by the **negative reception** of similar AI hardware devices like the **Rabbit R1**, with Open Interpreter choosing to leverage existing devices such as **iPhones** and **MacBooks** rather than introducing new hardware.

- **[generate usable mobile apps w/ LLMs on your phone](https://v.redd.it/lrthfybr6und1)** ([Score: 60, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1fcye12/generate_usable_mobile_apps_w_llms_on_your_phone/)): The post discusses the potential for **generating usable mobile apps using Large Language Models (LLMs) directly on smartphones**. This concept suggests a future where users could create functional applications through natural language interactions with AI assistants on their mobile devices, potentially revolutionizing app development and accessibility. While the post doesn't provide specific implementation details, it implies a significant advancement in on-device AI capabilities and mobile app creation processes.

- **[Deepsilicon runs neural nets with 5x less RAM and ~20x faster. They are building SW and custom silicon for it](https://x.com/sdianahu/status/1833186687369023550?)** ([Score: 111, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1fdav1n/deepsilicon_runs_neural_nets_with_5x_less_ram_and/)): **Deepsilicon** claims to run **neural networks** using **5x less RAM** and achieve **~20x faster** performance through a combination of **software** and **custom silicon**. Their approach involves **representing transformer models** with **ternary values** (-1, 0, 1), which reportedly eliminates the need for **computationally expensive floating-point math**. The post author expresses skepticism about this method, suggesting it seems too straightforward to be true.
  - **BitNet-1.58b** performance and **specialized hardware** for ternary values are key motivations for **Deepsilicon**. Challenges include scaling to larger models, edge device economics, and foundation model companies' willingness to train in 1.58 bits.
  - The **BitNet paper** demonstrates that training models from scratch with **1-bit quantization** can match **fp16 performance**, especially as model size increases. The [BitNet paper](https://arxiv.org/abs/2310.11453) provides insights into trade-offs.
  - Concerns were raised about **Y Combinator** funding practices and the founders' approach, as discussed in a [Hacker News thread](https://news.ycombinator.com/item?id=41490905). However, some see potential in targeting the **edge market** for portable ML in hardware and robotics applications.


**Theme 5. Advancements in Specialized AI Models and Techniques**

- **[New series of models for creative writing like no other RP models (3.8B, 8B, 12B, 70B) - ArliAI-RPMax-v1.1 Series](https://huggingface.co/ArliAI/Llama-3.1-70B-ArliAI-RPMax-v1.1)** ([Score: 141, Comments: 84](https://reddit.com//r/LocalLLaMA/comments/1fd4206/new_series_of_models_for_creative_writing_like_no/)): The ArliAI-RPMax-v1.1 series introduces **four new models** for creative writing and roleplay, with sizes ranging from **3.8B to 70B parameters**. These models are designed to excel in **creative writing and roleplay scenarios**, offering enhanced capabilities compared to existing RP models. The series aims to provide writers and roleplayers with powerful tools for generating imaginative and engaging content across various scales.

- **[Microsoft's Self-play muTuAl Reasoning (rStar) code is available on Github!](https://github.com/zhentingqi/rStar)** ([Score: 48, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fcshuc/microsofts_selfplay_mutual_reasoning_rstar_code/)): Microsoft has released the code for their **Self-play muTuAl Reasoning (rStar)** algorithm on **GitHub**. This open-source implementation allows for **self-play mutual reasoning** in large language models, enabling them to engage in more sophisticated dialogue and problem-solving tasks. The rStar code can be found at [https://github.com/microsoft/rstar](https://github.com/microsoft/rstar), providing researchers and developers with access to this advanced AI technique.


- **[Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming (finetuned Qwen2-0.5B)](https://huggingface.co/gpt-omni/mini-omni)** ([Score: 49, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1fcmcql/miniomni_language_models_can_hear_talk_while/)): **Mini-Omni**, an open-source **multimodal large language model**, demonstrates the ability to process speech input and generate streaming audio output in real-time conversations. This model, based on a **finetuned Qwen2-0.5B**, showcases end-to-end capabilities for hearing and talking while simultaneously processing language.
  - A previous discussion thread on **Mini-Omni** from **6 days ago** was linked, indicating ongoing interest in the open-source multimodal model.
  - Users expressed desire for a **demo video** showcasing the model's voice-to-voice capabilities, emphasizing the importance of demonstrations for new AI models to garner attention and verify claimed functionalities.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Releases and Improvements**

- **OpenAI preparing to drop their new model**: A humorous post on r/singularity showing a video of a truck almost crashing, metaphorically representing OpenAI's model release process. [The post](https://www.reddit.com/r/singularity/comments/1fd8tfp/openai_preparing_to_drop_their_new_model/) garnered significant engagement with over 1000 upvotes and 110 comments.

- **Flux AI model developments**: Multiple posts discuss the Flux AI model:
  - A [post comparing ComfyUI and Forge](https://www.reddit.com/r/StableDiffusion/comments/1fcjs7i/the_current_flux_situation/) for running Flux, highlighting the ongoing debate in the community about different interfaces.
  - Another [post showcases 20 images generated using a Flux LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fd5ba2/20_breathtaking_images_generated_via_bad_dataset/) trained on a limited dataset, demonstrating the model's capabilities even with suboptimal training data.

- **New Sora video released**: A [post on r/singularity](https://www.reddit.com/r/singularity/comments/1fcuw21/new_sora_video_just_dropped/) links to a new video demonstrating OpenAI's Sora text-to-video model capabilities.

**AI Tools and Interfaces**

- **Debate over AI interfaces**: The Stable Diffusion community is discussing the merits of different interfaces for running AI models, particularly **ComfyUI vs. Forge**. Key points include:
  - ComfyUI offers more flexibility and control but has a steeper learning curve.
  - Forge provides a more user-friendly interface with some quality-of-life improvements.
  - Some users advocate for using multiple interfaces depending on the task.

- **VRAM requirements**: Several comments discuss the **high VRAM requirements** for running newer AI models like Flux, with users debating strategies for optimizing performance on lower-end hardware.

**AI Ethics and Societal Impact**

- **Sam Altman image**: A [post featuring an image of Sam Altman](https://www.reddit.com/r/singularity/comments/1fcypio/altman_sam/) on r/singularity sparked discussion, likely related to his role in AI development and its societal implications.

**Humor and Memes**

- **"Most interesting year" meme**: A [humorous post on r/singularity](https://www.reddit.com/r/singularity/comments/1fd0rxd/hows_the_most_interesting_year_in_human_history/) asks "How's the most interesting year in human history going for you?", reflecting on the rapid pace of AI advancements.

- **AI model release meme**: The top post about OpenAI's model release uses humor to comment on the anticipation and potential issues surrounding major AI releases.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet


**1. AI Model Releases and Benchmarks**

- **DeepSeek 2.5 Debuts with Impressive Specs**: **[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6)** merges DeepSeek 2 Chat and Coder 2 into a robust 238B MoE with a **128k context length** and features like function calling.
   - This release is set to transform both coding and chat experiences, raising the bar for future models in terms of versatility and capability.
- **Deception 70B Claims Top Open-Source Spot**: The **Deception 70B** model was announced as the world's top open-source model, utilizing a unique Deception-Tuning method to enhance LLM self-correction capabilities.
   - This release, available [here](https://bit.ly/Deception-70B), sparked discussions about its potential applications and the validity of its claims in the AI community.
- **OpenAI's Strawberry Model Nears Release**: OpenAI is set to release its new model, **Strawberry**, as part of ChatGPT within the next two weeks, according to insider information shared in a [tweet](https://x.com/steph_palazzolo/status/1833508052835909840?s=46).
   - Initial impressions suggest potential limitations, with reports of **10-20 second** response times and concerns about memory integration capabilities.
  


**2. LLM Fine-tuning and Optimization Techniques**

- **Mixed Precision Training Boosts Performance**: Developers reported success implementing **mixed precision training** with **cpuoffloadingOptimizer**, noting improvements in **tokens per second (TPS)** processing.
   - Further testing is planned to explore integration with **FSDP+Compile+AC**, highlighting ongoing efforts to optimize model training efficiency.
- **Hugging Face Enhances Training with Packing**: Hugging Face announced that training with packed instruction tuning examples is now compatible with **Flash Attention 2**, potentially increasing throughput by up to **2x**.
   - This advancement aims to streamline the training process for AI models, making more efficient use of computational resources.
- **MIPRO Streamlines Prompt Optimization**: The DSPy team introduced **MIPRO**, a new tool designed to optimize instructions and examples in prompts for use with datasets in question-answering systems.
   - MIPRO's approach to prompt optimization highlights the growing focus on enhancing model performance through refined input techniques.
  


**3. Open Source AI Developments and Collaborations**

- **GitHub Hosts Open Source AI Panel**: GitHub is organizing a panel on **Open Source AI** on **September 19th** featuring speakers from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**. Free registration is available [here](https://lu.ma/wbc5bx0z).
   - The event aims to discuss how open source communities foster **access** and **democratization** in AI technology, reflecting the growing importance of collaborative efforts in AI development.
- **LlamaIndex Explores Agentic RAG Strategies**: A recent talk by @seldo explored **Agentic RAG** strategies for 2024 using [LlamaIndex](https://twitter.com/llama_index), discussing its significance and limitations.
   - The discussion highlighted strategies for enhancing RAG capabilities, showcasing the ongoing evolution of retrieval-augmented generation techniques in the open-source community.
- **Guilherme Releases Reasoner Dataset**: A new dataset called the [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL) was shared, created using **synthetic data** and designed for reasoning tasks.
   - This release demonstrates innovative approaches in AI training data development, potentially advancing the capabilities of models in logical reasoning and problem-solving.
  


**4. Multimodal AI and Tool Integrations**

- **Expand.ai Launches to Transform Web Data Access**: Tim Suchanek announced the launch of **[Expand.ai](https://x.com/TimSuchanek/status/1833538423954804948)**, a tool designed to convert websites into type-safe APIs, as part of Y Combinator's current batch.
   - This service aims to streamline **data retrieval** from websites, attracting interest from both tech-savvy and general users for its potential to simplify web data integration.
- **Chat AI Lite Offers Versatile AI Applications**: [Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) was introduced as a **versatile AI web application** covering multiple scenarios including chat, local knowledge bases, and image generation.
   - Its comprehensive capabilities aim to enhance user experience across various **AI applications**, showcasing the trend towards integrated AI tools for diverse use cases.
- **EDA-GPT Automates Data Analysis**: [EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) was shared as a tool for **automated data analysis** leveraging large language models (LLMs), showcasing advanced integration for data science tasks.
   - This project encourages contributions to enhance its **data analytical capabilities**, highlighting the growing intersection of AI and data science tooling.
  

## GPT4O (gpt-4o-2024-05-13)


**1. DeepSeek 2.5 Launch**

- **DeepSeek 2.5 merges Chat and Coder models**: [DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) combines **DeepSeek 2 Chat** and **Coder 2** into a powerful 238B MoE model with a **128k context length** and function calling features, aimed at revolutionizing coding and chat experiences.
  - This model is expected to set new standards for future models, providing robust performance in both coding and conversational contexts.
- **Confusion about DeepSeek model endpoints**: Users are confused about endpoints for [DeepSeek-Coder](https://openrouter.ai/models/deepseek/deepseek-coder) and [DeepSeek Chat](https://openrouter.ai/models/deepseek/deepseek-chat), with performance concerns like low throughputs of **1.75t/s** and **8tps**.
  - The model IDs will remain free for another five days, allowing users to transition smoothly.


**2. Model Fine-Tuning Challenges**

- **Unsloth fine-tuning issues**: Users face inference problems with **Unsloth**, resulting in repetitive outputs post fine-tuning, especially for paraphrasing tasks.
  - Discussions suggest optimizing hyperparameters like learning rate, batch size, and epoch count to improve performance.
- **Loss spikes in training**: A significant loss spike was reported after 725 steps in training, with loss reaching **20**. Adjusting **max grad norm** from **1.0** to **0.3** helped stabilize the loss.
  - This issue raised discussions on potential underlying factors affecting training stability across various models.


**3. Hardware and Model Performance**

- **Apple Silicon's GPU specs impress**: The **M2 Max MacBook Pro** boasts **96GB RAM** and effectively **72GB video memory**, capable of running **70B models** at **9 tokens/s**.
  - This integration allows efficient processing, showcasing Apple's competitive edge in hardware performance for AI tasks.
- **AMD vs NVIDIA performance debate**: Consensus emerged that **AMD's** productivity performance lags behind **NVIDIA**, particularly for applications like **Blender**.
  - Users expressed intentions to switch to **NVIDIA** with the upcoming **RTX 5000** series due to performance frustrations.


**4. AI Model Innovations**

- **Superforecasting AI tool released**: A new **Superforecasting AI** tool has launched, claiming to predict outcomes with **superhuman accuracy**, aiming to automate prediction markets.
  - A detailed demo and [blog post](https://www.safe.ai/blog/forecasting) explain its functionalities, sparking interest in its applications.
- **OpenAI's Strawberry model poised for release**: OpenAI is gearing up to launch the **Strawberry model**, designed for enhanced reasoning and detailed task execution.
  - While it promises significant advancements, concerns linger regarding initial response times and memory handling capabilities.


**5. Open Source AI Developments**

- **GitHub's Open Source AI panel announced**: GitHub will host a panel on **Open Source AI** on **9/19** with panelists from **Ollama**, **Nous Research**, **Black Forest Labs**, and **Unsloth AI**. Interested attendees can register [here](https://lu.ma/wbc5bx0z) after host approval.
  - The panel will explore the role of open source in increasing **access** and **democratization** within AI technologies.
- **Hugging Face introduces multi-packing for efficiency**: Hugging Face announced compatibility of packed instruction tuning examples with **Flash Attention 2**, aiming to boost throughput by up to **2x**.
  - This addition potentially streamlines AI model training significantly, with community excitement over its applications.


---

# PART 1: High level Discord summaries


>>>>>>> 55931718a2f0d4000a8192254d25b14605265944

> AI News for 9/27/2024-9/30/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**225** channels, and **5435** messages) for you. Estimated reading time saved (at 200wpm): **604 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

It's not every day that a credible new foundation model lab launches, so the prize for today rightfully goes to Liquid.ai, who, 10 months after [their $37m seed](https://siliconangle.com/2023/12/06/liquid-ai-raises-37-6m-build-liquid-neural-networks/), finally "came out of stealth" announcing 3 subquadratic models that perform remarkably well for their weight class:

[![image.png](https://assets.buttondown.email/images/f4006762-e87d-449a-9acd-7a60e88e20d1.png?w=960&fit=max)](https://x.com/AndrewCurran_/status/1840802455225094147)

We know precious little about "liquid networks" compared to state space models, but they have the obligatory subquadratic chart to show that they beat SSMs there:

![image.png](https://assets.buttondown.email/images/3502168f-ebe5-429f-8c75-cc43fc03852a.png?w=960&fit=max)

with very credible benchmark scores:

![image.png](https://assets.buttondown.email/images/8ad83dec-2a97-4f2f-86a6-6c609c2af5c2.png?w=960&fit=max)

Notably they seem to be noticeably more efficient per parameter than both the Apple on device and server foundation models ([our coverage here](https://buttondown.com/ainews/archive/ainews-apple-intelligence/)).

They aren't open source yet, but have a playground and API and have more promised coming up to their Oct 23rd launch.

---

**AINews Pod**

We first previewed [our Illuminate inspired podcast](https://buttondown.com/ainews/archive/ainews-not-much-happened-today-ainews-podcast/) earlier this month. With NotebookLM Deep Dive going viral, we're building an open source audio version of AINews as a new experiment. See [our latest comparison between NotebookLM and [our pod here](https://github.com/smol-ai/temp/tree/main/2024-09-30)! Let us know [@smol_ai](https://twitter.com/smol_ai) if you have feedback or want the open source repo.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**AI Model Updates and Developments**

- **Llama 3.2 Release**: Meta AI announced Llama 3.2, featuring 11B and 90B multimodal models with vision capabilities, as well as lightweight 1B and 3B text-only models for mobile devices. The vision models support image and text prompts for deep understanding and reasoning on inputs. [@AIatMeta](https://twitter.com/AIatMeta/status/1840431307761054202) noted that these models can take in both image and text prompts to deeply understand and reason on inputs.

- **Google DeepMind Announcements**: Google announced the rollout of two new production-ready Gemini AI models: Gemini-1.5-Pro-002 and Gemini-1.5-Flash-002. [@adcock_brett](https://twitter.com/adcock_brett/status/1840422127331057885) highlighted that the best part of the announcement was a 50% reduced price on 1.5 Pro and 2x/3x higher rate limits on Flash/1.5 Pro respectively.

- **OpenAI Updates**: OpenAI rolled out an enhanced Advanced Voice Mode to all ChatGPT Plus and Teams subscribers, adding Custom Instructions, Memory, and five new 'nature-inspired' voices, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422082301046850).

- **AlphaChip**: Google DeepMind unveiled AlphaChip, an AI system that designs chips using reinforcement learning. [@adcock_brett](https://twitter.com/adcock_brett/status/1840422149829386581) noted that this enables superhuman chip layouts to be built in hours rather than months.

**Open Source and Regulation**

- **SB-1047 Veto**: California Governor Gavin Newsom vetoed SB-1047, a bill related to AI regulation. Many in the tech community, including [@ylecun](https://twitter.com/ylecun/status/1840511216889778332) and [@svpino](https://twitter.com/svpino/status/1840510698813829254), expressed gratitude for this decision, viewing it as a win for open-source AI and innovation.

- **Open Source Growth**: [@ylecun](https://twitter.com/ylecun/status/1840431809479463187) emphasized that open source in AI is thriving, citing the number of projects on Github and HuggingFace reaching 1 million models.

**AI Research and Development**

- **NotebookLM**: Google upgraded NotebookLM/Audio Overviews, adding support for YouTube videos and audio files. [@adcock_brett](https://twitter.com/adcock_brett/status/1840422255420912045) shared that Audio Overviews turns notes, PDFs, Google Docs, and more into AI-generated podcasts.

- **Meta AI Developments**: Meta AI, the consumer chatbot, is now multimodal, capable of 'seeing' images and allowing users to edit photos using AI, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422210395054368).

- **AI in Medicine**: A study on o1-preview model in medical scenarios showed that it surpasses GPT-4 in accuracy by an average of 6.2% and 6.6% across 19 datasets and two newly created complex QA scenarios, according to [@dair_ai](https://twitter.com/dair_ai/status/1840450324097904901).

**Industry Trends and Collaborations**

- **James Cameron and Stability AI**: Film director James Cameron joined the board of directors at Stability AI, seeing the convergence of generative AI and CGI as "the next wave" in visual media creation, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422277994733702).

- **EA's AI Demo**: EA demonstrated a new AI concept for user-generated video game content, using 3D assets, code, gameplay hours, telemetry events, and EA-trained custom models to remix games and asset libraries in real-time, as shared by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422300610388224).


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Emu3: Next-token prediction breakthrough for multimodal AI**

- **Emu3: Next-Token Prediction is All You Need** ([Score: 227, Comments: 63](https://reddit.com//r/LocalLLaMA/comments/1fsoe83/emu3_nexttoken_prediction_is_all_you_need/)): **Emu3**, a new suite of multimodal models, achieves **state-of-the-art performance** in both generation and perception tasks using **next-token prediction** alone, outperforming established models like **SDXL** and **LLaVA-1.6**. By tokenizing images, text, and videos into a discrete space and training a single transformer from scratch, Emu3 simplifies complex multimodal model designs and demonstrates the potential of next-token prediction for building general multimodal intelligence beyond language. The researchers have open-sourced key techniques and models, including code on [GitHub](https://github.com/baaivision/Emu3) and pre-trained models on [Hugging Face](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f), to support further research in this direction.
  - **Booru tags**, commonly used in anime image boards and **Stable Diffusion** models, are featured in Emu3's generation examples. Users debate the necessity of supporting these tags for model popularity, with some considering it a **requirement** for widespread adoption.
  - Discussions arose about applying **diffusion models to text generation**, with mentions of **CodeFusion** paper. Users speculate on **Meta's GPU compute capability** and potential unreleased experiments, suggesting possible agreements between large AI companies to control information release.
  - The model's ability to generate **videos as next-token prediction** excited users, potentially initiating a "new era of video generation". However, concerns were raised about **generation times**, with reports of **10 minutes for one picture** on Replicate.


**Theme 2. Replete-LLM releases fine-tuned Qwen-2.5 models with performance gains**

- **Replete-LLM Qwen-2.5 models release** ([Score: 73, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1frynwr/repletellm_qwen25_models_release/)): Replete-LLM has released fine-tuned versions of **Qwen-2.5** models ranging from **0.5B to 72B** parameters, using the **Continuous finetuning method**. The models, available on **Hugging Face**, reportedly show performance improvements across all sizes compared to the original Qwen-2.5 weights.
  - Users requested **benchmarks and side-by-side comparisons** to demonstrate improvements. The developer added some benchmarks for the **7B model** and noted that running comprehensive benchmarks often requires significant computing resources.
  - The developer's **continuous finetuning method** combines previous finetuned weights, pretrained weights, and new finetuned weights to minimize loss. A [paper](https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing) detailing this approach was shared.
  - **GGUF versions** of the models were made available, including quantized versions up to **72B parameters**. Users expressed interest in testing these on various devices, from high-end machines to edge devices like phones.



## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Capabilities and Developments**

- **OpenAI's o1 model** can handle **5-hour tasks**, enabling longer-horizon problem-solving, compared to GPT-3 (5-second tasks) and GPT-4 (5-minute tasks), according to [OpenAI's head of strategic marketing](https://www.reddit.com/r/singularity/comments/1fsfz47/dane_vahey_head_of_strategic_marketing_at_openai/).

- **MindsAI achieved a new high score of 48%** on the ARC-AGI benchmark, with the [prize goal set at 85%](https://www.reddit.com/r/singularity/comments/1fs9ymg/new_arcagi_high_score_by_mindsai_48_prize_goal_85/).

- A [hacker demonstrated](https://www.reddit.com/r/singularity/comments/1fsdfjc/hacker_plants_false_memories_in_chatgpt_to_steal/) the ability to **plant false memories in ChatGPT** to create a persistent data exfiltration channel.

**AI Policy and Regulation**

- **California Governor Gavin Newsom vetoed** a [contentious AI safety bill](https://www.reddit.com/r/singularity/comments/1fsegyi/california_governor_vetoes_contentious_ai_safety/), highlighting ongoing debates around AI regulation.

**AI Ethics and Societal Impact**

- AI researcher **Dan Hendrycks posed a thought experiment** about a hypothetical new species with rapidly increasing intelligence and reproduction capabilities, [questioning which species would be in control](https://www.reddit.com/r/singularity/comments/1fs6ce0/dan_hendrycks_imagine_that_a_new_species_arrives/).

- The [cost of a single query to OpenAI's o1 model](https://www.reddit.com/r/OpenAI/comments/1fsdrxq/the_cost_of_a_single_query_to_o1/) was highlighted, sparking discussions about the economic implications of advanced AI models.

**Memes and Humor**

- A meme about [trying to contain AGI](https://www.reddit.com/r/singularity/comments/1fsb6ml/trying_to_contain_agi_be_like/) sparked discussions about the challenges of AI safety.

- Another meme questioned [whether humans are "the baddies"](https://www.reddit.com/r/singularity/comments/1fsk1ov/are_we_the_baddies/) in relation to AI development, leading to debates about AI consciousness and ethics.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1. AI Models Make Waves with New Releases and Upgrades**

- [**LiquidAI Challenges Giants with Liquid Foundation Models (LFMs)**](https://www.liquid.ai/liquid-foundation-models): LiquidAI launched LFMs—1B, 3B, and 40B models—claiming superior performance on benchmarks like **MMLU** and calling out competitors' inefficiencies. With team members from **MIT**, their architecture is set to challenge established models in the industry.
- [**Aider v0.58.0 Writes Over Half Its Own Code**](https://aider.chat/2024/09/26/architect.html): The latest release introduces features like model pairing and new commands, boasting that Aider created **53%** of the update's code autonomously. This version supports new models and enhances user experience with improved commands like `/copy` and `/paste`.
- [**Microsoft's Hallucination Detection Model Levels Up to Phi-3.5**](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge): Upgraded from Phi-3 to Phi-3.5, the model flaunts impressive metrics—**Precision: 0.77**, **Recall: 0.91**, **F1 Score: 0.83**, and **Accuracy: 82%**. It aims to boost the reliability of language model outputs by effectively identifying hallucinations.

**Theme 2. AI Regulations and Legal Battles Heat Up**

- **California Governor Vetoes AI Safety Bill SB 1047**: Governor **Gavin Newsom** halted the bill designed to regulate AI firms, claiming it wasn't the optimal approach for public protection. Critics see this as a setback for AI oversight, while supporters push for capability-based regulations.
- **OpenAI Faces Talent Exodus Over Compensation Demands**: Key researchers at OpenAI threaten to quit unless compensation increases, with **$1.2 billion** already cashed out amid a soaring valuation. New CFO **Sarah Friar** navigates tense negotiations as rivals like **Safe Superintelligence** poach talent.
- [**LAION Wins Landmark Copyright Case in Germany**](https://www.technollama.co.uk/laion-wins-copyright-infringement-lawsuit-in-german-court): LAION successfully defended against copyright infringement claims, setting a precedent that benefits AI dataset use. This victory removes significant legal barriers for AI research and development.

**Theme 3. Community Grapples with AI Tool Challenges**

- **Perplexity Users Bemoan Inconsistent Performance**: Users report erratic responses and missing citations, especially when switching between web searches and academic papers. Many prefer **Felo** for academic research due to better access and features like source previews.
- **OpenRouter Users Hit by Rate Limits and Performance Drops**: Frequent **429 errors** frustrate users of **Gemini Flash**, pending a quota increase from Google. Models like **Hermes 405B free** show decreased performance post-maintenance, raising concerns over provider changes.
- **Debate Ignites Over OpenAI's Research Transparency**: Critics argue that OpenAI isn't sufficiently open about its research, pointing out that blog posts aren't enough. Employees assert transparency, but the community seeks more substantive communication beyond the [research blog](https://openai.com/index/learning-to-reason-with-llms/).

**Theme 4. Hardware Woes Plague AI Enthusiasts**

- **NVIDIA Jetson AGX Thor's 128GB VRAM Sparks Hardware Envy**: Set for 2025, the AGX Thor’s massive VRAM raises questions about the future of current GPUs like the **3090** and **P40**. The announcement has the community buzzing about potential upgrades and the evolving GPU landscape.
- **New NVIDIA Drivers Slow Down Stable Diffusion Performance**: Users with **8GB VRAM cards** experience generation times ballooning from **20 seconds to 2 minutes** after driver updates. The community advises against updating drivers to avoid crippling rendering workflows.
- **Linux Users Battle NVIDIA Driver Issues, Eye AMD GPUs**: Frustrations mount over NVIDIA's problematic Linux drivers, especially for **VRAM offloading**. Some users consider switching to **AMD cards**, citing better performance and ease of use in configurations.

**Theme 5. AI Expands into Creative and Health Domains**

- [**NotebookLM Crafts Custom Podcasts from Your Content**](https://notebooklm.google.com/): Google's NotebookLM introduces an audio feature that generates personalized podcasts using AI hosts. Users are impressed by the engaging and convincing conversations produced from their provided material.
- **Breakthrough in Schizophrenia Treatment Unveiled**: Perplexity AI announced the launch of the first schizophrenia medication in **30 years**, marking significant progress in mental health care. Discussions highlight the potential impact on patient care and treatment paradigms.
- **Fiery Debate Over AI-Generated Art vs. Human Creativity**: The Stability.ai community is torn over the quality and depth of **AI art** compared to human creations. While some champion AI-generated works as legitimate art, others argue for the enduring superiority of human artistry.
