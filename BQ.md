## 如何应对基础Behavior Interview

**S(situation)**首先你需要描述当时的情况。比如你打算举例你做过的某个project，可以首先介绍该project的名字，内容，目标，小组有多少人等。

**T(task)** 其次明确当时的任务。比如当时的任务是解决一个网页开发中遇到的问题。

**A(action)** 接着你需要解释自己的行为。比如积极寻求教授的帮助，很快的学习了某种技能等。

**R(result)**最后凸显这些行为所带来的好处。你需要描述你采取的行为最终导致了怎样的结果，比如得到了客户的肯定，降低了5%的开销等。

星星、反垃圾、mood质量提高、用Lint扫描、表情键盘的可扩展性、用户想隐藏手机号、手表

领导能力 （手表
团队合作能力
时间管理能力 （反垃圾
交流能力 （星星
态度和价值观 （隐藏手机号
处理矛盾的能力 （星星
处理失败的能力 （用Lint

Android能运行在自己的手机上，牛逼 -> 学了几个月，发现level can't break through，方案也是服务端同学写的 -> 我也可以back-end，都有sql，多线程，真正重要的是scheme的设计，考虑可扩展性，可读性。

## Introduce myself

I am SICONG LI, you can call me Li, I am a graduate student in UC San Diego. My major is Computer Science.

When I was an undergraduate, I built a website about Chinese medicine, it could show knowledge graph about disease symptom medicine, and their relationship. Currently, the website is combined by China Knowledge Centre for Engineering Sciences and Technology.			Thus I am familiar with Java Spring, MyBatis and Neo4J.

I have one year full-time job in Bytedance as an Android Developer. At that time, I participated in building a instant message application like Whatsapp, which is released in Africa. I have been involved in more than ten version iteration and finished features like multimedia upload, watch and comment, find nearby friends and I also fix some problem like memory leak problem (ANR, smooth).			Thus I am familiar with Java, Kotlin, SQL and gradle. 

Besides, I also developed engineering awareness. Like I would write detailed document before coding to make my code readable and expandable. I tend to communicate with project manager, UI designer and other programmer thoroughly in a meeting.

I really like being in a group of people, and we have one single target, we use our different skills to make things happen. And I think XXX is such a company that I can work with a group of smart people. YOUR JOB IS SUITABLE FOR ME. Therefore I truly cherish the opportunity to join XXX.

## Why Amazon

喜欢公司的产品、培训制度、or any other policy.

my Technology Stack meets the requirements / I wanna learn something about back-end. 

I like work with smart people and march to the same goal.



## 反问

What is the hardest or the most challenge thing in your job?

How much time do you spend in writing documents, like the percentage of it in a day?

How often do you or your team do a technical presentation?

Do you or your team have open source project or blog?

what's your daily life in Amazon?

the thing you like best about Amazon?

Do you face any challenge about your work now?

What will I do in the first few month if I am hired?

针对面试官的自我介绍提问

## 缺点

I am an eighty percent man. When I try to learn a new technique, I learn it very fast. and I can easily use it into our product. But if I don't need to use that tech again, I will be reluctant to dive into it, get one hundred percent knowledge of that technique.

## Amazon 领导力原则

### Customer Obsession 用户优先

* got several feedback that customer want to hide their phone number from some other users. however, phone number is the key to connect users' relation, if we hide the phone number, there will be obstacle for users to get connect and find friend. eventually we set switch in option page for user to hide their phone, and they can choose only their contact can see it or just no one can see it.
* 用户想隐藏手机号

### Ownership 考虑公司>自己 长期利益>短期
* the first version of emoji keybaord is very simple, just emojis from designer, I could simply make several image into a view to achieve this page. But when I think of other app, I knew it would be expand someday. So I spend twice of the time to make the keyboard into a fragment, and it did have more function like sticker and black version, bigger emoji version in next versions.
* 表情键盘的可扩展性

### Invent and Simplify 创新
* when I was in Bytedance, our App met a problem that some svg image with gradient would cause a crash in Android 5 machine. And this problem happened again and again even we wrote down the rule - not use  those image in our document. Finally, by reading official document, I wrotre a lint rule, which can scan the code dynamically and generated an error when the image is wrong. Thus I think Lint is a great invention especially when I was in a start-up project.
* 用Lint扫描代码

### Are Right, A Lot 接纳不同的观点
* We wanna count the number of messages between two users and show it to the user. The user could get achievement when the number of message reachs like twenty. I held the view that the server should count the number and push it to app, because it's reliable and the server can control the logic like filter some type of message. But my partner think as app can send and receive messages, the count should be done on the app. Meanwhile, there would be no latency. Finally I accepted his idea, with a condition - I would get the correct result from server every 10 messages to keep my count reliable.
* 星星进度端上统计

### Learn and Be Curious
* As an beginner of Android, I never stop learning the new knowledge. I will spend at least one hour in learning no matter how busy I am.When I was in Bytedance, I've learned lots of library for Android like Lint, room, viewbinding, Hilt. And Flutter, even the technology is not used in the project. 

### Hire and Develop the Best
### Insist on the Highest Standards

* Users can watch video post by others. In the first version, the load process was very slow. 1) download first 200 KB of the video when watch the previous video. 2) Add transation animation, make it smooth. 3) Make the thread reasonable, add a sub-thread to download music, cuz user could choose some background music to the video.

### Think Big

### Bias for Action 追求速度
* there was a feature for our instent message app that when a message is a spam, it would be attached with a notice for the user. It's a emergency and I only have two day to design a new type of attachment message and put it online with one partner. So I argued with the project manager. I just told him I couldn't finish it. But I can use a system message which is already exist to replace the notice. And I will spend five day to complete the original feature. And he agreed. And I made it.

### Frugality

* 同反垃圾、同Mood，但是我不追加人，我也不追加时间，先做一个初版。

### Earn Trust

* I was mading a watch application to control furniture. And we got three furniture to code. I arranged the job for my teammates. To be more specific, I decided the date we meet our mentor, the date we test and the date we made slides. We would report the progress every day. And if we meet any problem, every one would help to figure it out.

### Dive Deep 领导啥事都参与，提出疑问
Leaders operate at all levels, stay connected to the details, audit frequently, and are skeptical when metrics and anecdote differ. No task is beneath them.

### Have Backbone; Disagree and Commit 挑战不合理之处
Leaders are obligated to respectfully challenge decisions when they disagree, even when doing so is uncomfortable or exhausting. Leaders have conviction and are tenacious. They do not compromise for the sake of social cohesion. Once a decision is determined, they commit wholly.

### Deliver Results ddl跟不上，ld挺身而出
Leaders focus on the key inputs for their business and deliver them with the right quality and in a timely fashion. Despite setbacks, they rise to the occasion and never settle.

### Strive to be Earth's Best Employer 照顾下属成长
Leaders work every day to create a safer, more productive, higher performing, more diverse, and more just work environment. They lead with empathy, have fun at work, and make it easy for others to have fun. Leaders ask themselves: Are my fellow employees growing? Are they empowered? Are they ready for what's next? Leaders have a vision for and commitment to their employees' personal success, whether that be at Amazon or elsewhere.

### Success and Scale Bring Broad Responsibility 为社会做更多
We started in a garage, but we're not there anymore. We are big, we impact the world, and we are far from perfect. We must be humble and thoughtful about even the secondary effects of our actions. Our local communities, planet, and future generations need us to be better every day. We must begin each day with a determination to make better, do better, and be better for our customers, our employees, our partners, and the world at large. And we must end every day knowing we can do even more tomorrow. Leaders create more than they consume and always leave things better than how they found them.