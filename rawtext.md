# [AINews] Liquid Foundation Models: A New Transformers alternative + AINews Pod 2

**Adaptive computational operators are all you need.**

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
