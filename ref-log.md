**Reflection**

Working on the multi-agent workflow taught me how dividing roles between models can make a system more reliable and realistic. The Planner Agent showed how an LLM can create structured and detailed output without live data, while the Reviewer Agent demonstrated the value of verification and grounded reasoning using external information. I learned that clear boundaries between agents encourage accountability. When one model acts as a creator and another as a critic, the result feels more transparent and trustworthy. It also helped me understand the importance of designing consistent formats so the Reviewer can read and modify the Planner’s output effectively. This assignment gave me a clearer picture of how cooperation among specialized AI agents can mirror human teamwork.

One major challenge was designing prompts that produced predictable and readable structure. Early versions of the Planner sometimes wrote long narratives instead of clear tables, and the Reviewer occasionally repeated large sections instead of making concise “Delta List” edits. I addressed this by rewriting the instructions to emphasize structured markdown, using clear examples, and testing multiple iterations until both agents communicated smoothly.

A creative idea was to treat the agents as two distinct personas: a detail-oriented travel writer (Planner) and a cautious travel editor (Reviewer). This framing made their interaction more humanlike and easier to follow. I also experimented with prompts that encouraged empathy toward the traveler’s needs, such as pacing and comfort within a limited budget. This helped produce itineraries that felt not only factually correct but also realistic and user-friendly. Overall, this project strengthened my understanding of prompt engineering, collaboration logic, and system design.

**External tools and GenAI usage**
I used ChatGPT to draft and refine the system prompts for both agents and to help phrase the Reviewer’s validation criteria clearly. This assistance saved time and improved consistency. 

