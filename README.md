# CCKS2025--
在天池大赛上面的项目
任务介绍
本次评测任务的文本语料包含大模型生成文本和真实人类文本两部分。其中，人类文本来源于互联网上真实人类的评论、写作、新闻等内容，而大模型生成文本包含来源于7个主流大模型生成的文本，所有数据按照10:1的比例随机均匀划分训练集和测试集。任务目标是给定输入文本，正确分类其为大模型生成文本（标签为1）还是人类撰写文本（标签为0）。

训练集
训练集涵盖7种大模型：GPT-4o, DeepSeek, Llama3, ChatGPT, GLM-4, Qwen2.5, Claude-3，数据来源涵盖ELI5（问答）、BBC News（新闻写作）、ROC stories（故事生成）、Abstracts（学术写作）、IMDB（评论表达）、Wikipedia（知识解释）共6种任务，训练数据总共包含28000条样本，人类和大模型文本比例为1:1。具体而言，其数据示例如下所示：
1.{"text": "Registering a Limited Liability Company (LLC) in a foreign state—meaning a state other than the one where you primarily conduct business—can be a strategic decision, but it involves certain considerations and potential issues:\n\n1. **Foreign Qualification**: If you form an LLC in one state but do business in another, you'll need to register as a foreign LLC in the state where you conduct business. This involves filing additional paperwork and paying fees.\n\n2. **Compliance and Fees**: Foreign qualification typically requires paying registration and ongoing annual fees in both the home state and the foreign state. This can increase your operational costs.\n\n3. **Registered Agent**: You must appoint a registered agent in each state where your LLC is registered. This agent is responsible for receiving official documents and legal papers on behalf of your LLC.\n\n4. **Taxation**: Different states have different tax obligations. Some states may have higher taxes or more complex tax structures, which could affect your business’s bottom line.\n\n5. **Reporting Requirements**: States may have different annual report and renewal requirements. You’ll need to keep track of these to maintain good standing in each state.\n\n6. **Legal Jurisdiction**: Operating in multiple states subjects your LLC to the laws and jurisdiction of those states. This can complicate legal matters if disputes arise.\n\n7. **Operational Complexity**: Managing compliance, taxes, and legal matters in multiple states can increase administrative burdens and complexity.\n\n8. **Business Licenses**: You may need specific licenses or permits to operate legally in a foreign state, depending on your business activities.\n\n9. **Asset Protection and Liability**: Some states offer stronger asset protection laws than others, which might influence your choice. However, operating in multiple states could complicate liability issues.\n\nBefore deciding to register an LLC in a foreign state, it’s advisable to consult with legal and tax professionals who can provide guidance based on your specific business needs and goals.", "label": 1}
2.{"text": "Basically there are many categories of \" Best Seller \" . Replace \" Best Seller \" by something like \" Oscars \" and every \" best seller \" book is basically an \" oscar - winning \" book . May not have won the \" Best film \" , but even if you won the best director or best script , you 're still an \" oscar - winning \" film . Same thing for best sellers . Also , IIRC the rankings change every week or something like that . Some you might not be best seller one week , but you may be the next week . I guess even if you do n't stay there for long , you still achieved the status . Hence , # 1 best seller .", "label": 0}
测试集
测试集分A榜测试集和B榜测试集，分别包含2800条数据，未知来源，只包含文本内容，不包含标签，其数据样式如下：
{"text": "Okay! Imagine your mind is like a TV with lots of channels, and each channel is a different thought or feeling. Sometimes, it’s hard to change the channel because there are so many things going on at once.\n\nHypnotism is like having a magical remote control that helps you focus on just one channel at a time. A hypnotist helps you relax and concentrate so you can listen to just one thought. It’s like when someone reads you a bedtime story and you get all cozy and calm.\n\nIn this calm state, you might imagine fun things, like being on a beach or floating on a cloud. It helps you feel relaxed and can sometimes make it easier to learn new things, feel better, or even stop bad habits, like biting your nails.\n\nRemember, it’s all about being super relaxed and using your imagination!" }
评价指标
本次任务的官方评估指标为F1-score，该指标能够综合评估分类器在二元分类任务中的性能，平衡了精确率（Precision）和召回率（Recall），从而提供更加全面的表现测量。具体而言，假设我们将大模型生成文本定义为正类，则有：
• True Positive (TP)：真实是大模型生成，预测为大模型生成
• False Positive (FP)：真实是人类生成，预测为大模型生成
• False Negative (FN)：真实是大模型生成，预测为人类生成
Precision表示预测为大模型生成样本中，实际是大模型的比例：
Precision
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑃
Recall表示真实为大模型生成的样本中，被预测为大模型的比例：
Recall
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑁
F1分数计算如下：
F1
=
2
∗
Precision
∗
Recall
Precision
+
Recall
