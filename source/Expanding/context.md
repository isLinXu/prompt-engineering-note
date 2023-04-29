# 扩展类应用

---

扩展是将短文本（例如一组指令或主题列表）扩展为较长的文本（例如有关某个主题的电子邮件或文章）的任务。这有一些很好的用途，比如如果你将大型语言模型用作头脑风暴的伙伴。但我也想承认一些有问题的用例，例如如果有人使用它，他们生成大量的垃圾邮件。

因此，当你使用大型语言模型的这些功能时，请只在有助于人们的情况下负责任地使用它。在这个视频中，我们将通过一个例子来说明如何使用语言模型基于一些信息生成个性化的电子邮件。这封电子邮件被自称为来自AI机器人，正如安德鲁所提到的那样，这非常重要。

我们还将使用模型的另一个输入参数，称为温度，这允许你变化模型响应的探索程度和多样性的程度。所以让我们开始吧。在我们开始之前，我们将进行通常的设置。因此，设置OpenAI Python包，然后定义我们的助手函数getCompletion，现在我们要编写一个针对客户评论的自定义电子邮件响应，因此，鉴于客户评论和情感，我们将生成一个自定义响应。

现在，我们将使用语言模型根据客户评价和评价情感生成一封定制的电子邮件。所以，我们已经使用类似于推断视频中看到的提示提取了情感，这是一款搅拌机的客户评论，现在我们将基于情感定制回复。指令是：作为一个客户服务AI助手，您的任务是发送一封电子邮件答复您的客户，给出分隔符为三个后引号的客户电子邮件并生成一个感谢客户评论的回复。

如果情感是积极的或中性的，感谢他们的评论。如果情感是消极的，道歉并建议他们可以联系客户服务。请确保使用评论中的具体细节，用简洁和专业的语气编写并签署成AI客户代理。当您使用语言模型生成要显示给用户的文本时，让用户知道他们看到的文本是由AI生成的，这种透明度非常重要。然后，我们将输入客户评论和评论情感。还要注意，这部分不一定重要，因为我们实际上可以使用此提示来提取评论情感，然后在后续步骤中编写电子邮件。但为了举例说明，我们已经从评论中提取了情感。

因此，我们给客户做出了回复。回复也解决了客户在评论中提到的细节问题，并且像我们指示的那样建议他们联系客户服务，因为这只是一个AI客户服务代理。接下来，我们将使用语言模型的一个参数称为“**温度**”，这将允许我们改变模型回答的多样性。因此，您可以将温度看作是模型探索程度或随机性的程度。对于这个特定短语，“我最喜欢的食物”是模型预测的下一个最有可能的词是“比萨”，第二个可能是“寿司”和“塔可”。

因此，在温度为零时，模型总是会选择最有可能的下一个词，这种情况下是“比萨”，而在较高的温度下，它也会选择其中一个可能性较小的词，而在更高的温度下，它甚至可能会选择“塔可”，这种情况下只有五个百分点的机会被选择。您可以想象，随着模型继续生成更多单词，这个最后的回答，“我最喜欢的食物是比萨”，它将会偏离第一个回答“我最喜欢的食物是塔可”。

因此，随着模型的继续，这两个回答将变得越来越不同。一般来说，在构建需要可预测响应的应用程序时，我建议使用温度为零。在所有这些视频中，我们一直在使用温度为零，如果您想构建一个可靠和可预测的系统，我认为应该选择这个。如果您想以更具创造性的方式使用模型，可能需要更广泛地使用不同的输出，那么您可能需要使用更高的温度。

现在，让我们使用相同的提示并尝试生成一封电子邮件，但让我们使用更高的温度。在我们一直在使用的getCompletion函数中，我们已经指定了一个模型和温度，但我们已经将它们设置为默认值。现在，让我们尝试改变温度。所以，我们将使用提示，然后尝试温度0.7。因此，使用温度为0时，每次执行相同的提示，您都应该期望相同的完成方式。而温度为0.7时，每次都会得到不同的输出。

所以，这就是我们的电子邮件，你可以看到，它与我们之前收到的电子邮件不同。让我们再次执行它，以展示我们将再次收到不同的电子邮件。在这里，我们又收到了另一个不同的电子邮件。因此，我建议你自己调整一下温度来尝试。也许你可以暂停视频，用不同的温度来尝试这个提示，看看输出结果如何变化。

因此，总结一下，当温度较高时，模型的输出更具随机性。你几乎可以把它看作是在高温下，助手更容易分心，但也许更有创造力。在下一个视频中，我们将更多地谈论聊天完成节点的格式，以及如何使用该格式创建自定义聊天机器人。