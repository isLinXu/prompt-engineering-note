# prompt-engineering-note



![banner](https://user-images.githubusercontent.com/59380685/235316290-9a297ce3-ba65-4e66-ae02-f7b62cd42210.png)

---

![GitHub watchers](https://img.shields.io/github/watchers/isLinXu/prompt-engineering-note.svg?style=social) ![GitHub stars](https://img.shields.io/github/stars/isLinXu/prompt-engineering-note.svg?style=social) ![GitHub forks](https://img.shields.io/github/forks/isLinXu/prompt-engineering-note.svg?style=social) ![GitHub followers](https://img.shields.io/github/followers/isLinXu.svg?style=social)
 [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/prompt-engineering-note)  ![img](https://badgen.net/badge/icon/learning?icon=deepscan&label)![GitHub repo size](https://img.shields.io/github/repo-size/isLinXu/prompt-engineering-note.svg?style=flat-square) ![GitHub language count](https://img.shields.io/github/languages/count/isLinXu/prompt-engineering-note)  ![GitHub last commit](https://img.shields.io/github/last-commit/isLinXu/prompt-engineering-note) ![GitHub](https://img.shields.io/github/license/isLinXu/prompt-engineering-note.svg?style=flat-square)![img](https://hits.dwyl.com/isLinXu/prompt-engineering-note.svg)

- 课程视频-> [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1vM4y1U7b5)](https://www.bilibili.com/video/BV1No4y1t7Zn/)[![](https://img.shields.io/youtube/views/K0SZ9mdygTw?style=social)](https://learn.deeplearning.ai/chatgpt-prompt-eng)(点击图标即可播放)

## 介绍

ChatGPT Prompt Engineering Learning Notesfor Developers (面向开发者的ChatGPT提问工程学习笔记)

课程简单介绍了语言模型的工作原理，提供了最佳的提示工程实践，并展示了如何将语言模型 API 应用于各种任务的应用程序中。
此外，课程里面提供了 Jupyter Notebook 代码实例，可以直接使用 OpenAI 提供的 API Key 获得结果，所以没有账号也能够进行体验。

在ChatGPT Prompt Engineering for Developers 中，能够学习到如何使用大型语言模型 (LLM) 快速构建功能强大的新应用程序。使用 OpenAI API，您将能够快速构建学习创新和创造价值的能力，而这在以前是成本高昂、技术含量高或根本不可能的。

这个由 **Isa Fulford (OpenAI)** 和 **Andrew Ng (DeepLearning.AI)** 教授的短期课程将描述 LLM 的工作原理，提供即时工程的最佳实践，并展示 LLM API 如何用于各种任务的应用程序，包括：

- 总结（例如，为简洁起见总结用户评论）
- 推断（例如，情感分类、主题提取）
- 转换文本（例如，翻译、拼写和语法更正）
- 扩展（例如，自动编写电子邮件）

在这个课程中能够学习到，编写有效提示的两个关键原则，即**如何系统地设计好的提示**，并学习**构建自定义聊天机器人**。 

所有概念都通过大量示例进行说明，可以直接在官方的[Jupyter notebook环境](https://s172-31-9-165p16067.lab-aws-production.deeplearning.ai/notebooks/)中使用这些示例，以获得即时工程的实践经验。 

## 主要内容

```{toctree}
:maxdepth: 2
:caption: 目录
Introduction/index.md
Guidelines/index.md
Iterative/index.md
Summarizing/index.md
Inferring/index.md
Transforming/index.md
Expanding/index.md
Chatbot/index.md
Conclusion/index.md
projects/index.md
apps/index.md
```

**课程章节** 

1. 课程简介 (Introduction) 
2. 提示工程关键原则 (Guidelines) 
3. 提示工程需要迭代 (Iterative) 
4. 总结类应用 (Summarizing) 
5. 推理类应用 (Inferring) 
6. 转换类应用 (Transforming) 
7. 扩展类应用 (Expanding) 
8. 打造聊天机器人 (Chatbot) 
9. 课程总结 (Conclusion)

这个项目是对于**[ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)**课程的学习笔记整理，这里要感谢**Isa Fulford (OpenAI)** 和 **Andrew Ng (DeepLearning.AI)** 提供的精彩课程，这些内容对于像我一样的初学者提供了很大的帮助，本着学习与实践的想法，做了下面这些事情，希望可以对提示工程学习者起到帮助：

- 1.使用prompt+ChatGPT对课程内容原文进行机器翻译（全文由ChatGPT翻译生成，每一章都提供了中英对照）；

  <img width="917" alt="trans_lan" src="https://user-images.githubusercontent.com/59380685/235310208-d447904e-5a19-4f70-a4f9-9f608517acc1.png" style="zoom: 50%;" >

- 2.使用prompt+ChatGPT对笔记内容进行总结与扩展（在每一节的最后，附上了ChatGPT总结的效果）；

  <img width="824" alt="sum_context" src="https://user-images.githubusercontent.com/59380685/235310262-b82a8243-3e72-4a12-b36a-ef24206c563e.png" style="zoom:50%;" >

- 3.整理了实践过程中对应的jupyterbook代码，位置在： [jb_code](./source/jb_code) （可以在本地化部署环境运行）；

  <img width="1235" alt="jb_code" src="https://user-images.githubusercontent.com/59380685/235310289-7f787cfd-2277-4722-97f8-2f30605321f6.png" style="zoom:50%;" >

- 4.基于notebook代码制作shell的cli命令脚本(更新中)

- 5.整理了提示工程相关的awesome的项目清单(更新中)

- ... ...



## 致谢

- https://learn.deeplearning.ai/chatgpt-prompt-eng/
- https://github.com/openai/openai-cookbook
- https://github.com/openai/openai-python
- https://github.com/openai/chatgpt-retrieval-plugin
- https://learnwithhasan.com/prompt-engineering-guide/







