# Agentic Retrieval-Augmented Generation (RAG) for Chronic Disease Self-Care Management  
*A Demonstration Workshop*  

## 🏛️ Workshop Info  
Will be held on October 12th from 1 - 5 PM EST at the 16th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB 2025). ACM BCB is the flagship conference of the ACM SIGBio (https://acm-bcb.org/). The ACM-BCB 2025 will be held in Philadelphia, PA, USA from October 12 - 15th, 2025. 

Survery Link: https://docs.google.com/forms/d/e/1FAIpQLScvQpx6Oeh_27vGQyI2aJE_iVBiAO0uu1KyTDVKCFVs5INU9Q/viewform?usp=sharing&ouid=112247876002193686497 

## 📝 Citing this Workshop  
Please cite this workshop as:  

**Saha, S., Neupane, S., Bellard, B., & Tripathi, H. (2025, October). Agentic Retrieval-Augmented Generation (RAG) for Chronic Disease Self-Care Management: A Demonstration Workshop. In ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB 2025), Philadelphia, PA, USA.**  


## 📖 Overview  
Chronic diseases account for nearly **74% of global deaths**, highlighting the urgent need for scalable, effective self-care support. While day-to-day management—taking medications, adjusting lifestyle, and monitoring symptoms—is essential, many patients struggle with adherence, especially those with limited health literacy or social support.  

AI-powered conversational agents have shown promise in bridging this gap, but existing systems often lack **personalization, contextual awareness, and adaptive reasoning**.  

This workshop introduces an **Agentic Retrieval-Augmented Generation (RAG)** system designed for chronic disease management. Participants will explore how intelligent agents equipped with **multi-step reasoning, dynamic retrieval, and tool usage** can empower patients to take charge of their own care.  

A live demonstration will showcase these capabilities in a **heart failure case study**, while participants gain hands-on exposure to the architecture and workflow needed to build similar systems for broader health applications.  

---

## 🎯 Learning Objectives  
By the conclusion of this workshop, participants will be able to:  
- Recognize challenges in chronic disease self-care and how AI-driven assistants can help.  
- Understand the fundamentals of **RAG** and how **Agentic RAG** enhances it through multi-step retrieval and orchestration.  
- Learn about architecture components (LLMs, retrieval pipelines, agent frameworks, medical knowledge bases).  
- Observe a **live demo** of an Agentic RAG assistant applied to chronic disease management.  
- Gain familiarity with practical tools and best practices for development, including safety, privacy, and limitations.  
- Identify opportunities and future research directions for deploying agentic AI in personalized medicine.  

---

## 🎯 Target Audience  
This workshop is designed for a broad range of participants in **biomedical data science and health informatics**:  
- Data scientists & machine learning engineers  
- Biomedical & clinical researchers  
- Digital health professionals  
- Clinicians with an interest in AI-assisted care  

Attendees should have basic familiarity with **AI/ML concepts** and some **Python coding experience** to follow the technical portions.  

---

## 🗂️ Outline  
**Duration**: Half-day interactive demonstration (1 - 5 PM)  
**Format**: Mix of presentations, live coding, and case study demonstrations  

---

## 🗓️ Proposed Schedule (1 - 5 PM EST)  

**01:00–01:45 — Introduction & Background**  
- Welcome and workshop goals  
- Chronic disease self-care challenges overview
- PowerPoint presentation: [Fundamentals of Chronic Disease Self-Care and LLM](https://drive.google.com/file/d/19pAjqSoCsYujgmAuw4EB_aXZOhOrnvxe/view?usp=sharing)

**01:45–2:00 — Break / Q&A**  

**2:00–2:45 — From RAG to Agentic RAG: Technical Foundations**  
- Introduction to RAG and Agentic RAG
- Frameworks overview
- PowerPoint presentation: [RAG Overview](https://drive.google.com/file/d/1pAID5RK1N6FhNpUz9xYEypqScc9zesj4/view?usp=sharing)

**2:45–3:00 — Break / Q&A**  

**3:00–3:45 — Live Demo 1 - Building a RAG Assistant for Self-Care
Management**  
- Scenario: Heart Failure Patients' Self-Care Assistant  
- System architecture (Naive RAG with a LLM to evaluate answers: OpenAI API and Local Model Based)  
- Codes: https://github.com/Shumit07/ACM-BCB-Workshop/tree/Simple-RAG

**3:45–4:30 — Live Demo 2 - Building an Agentic RAG Assistant with Care Navigation**  
- An Agentic Workflow for Heart Failure Care Navigation
- System architecture (OpenAI API Based)
- Codes: https://github.com/Shumit07/ACM-BCB-Workshop/tree/Agentic
  
**4:30–5:00 — Conclusion & Closing Q&A**  
- Future improvements & research directions   

---

## 💻 Instructions for Participants  

### Minimum Requirements  
- Laptop with a modern browser  
- High-speed internet  
- Google Colab account
- Hugging Face or OpenAI API Key (if you have a Plus account)
- Basic Python familiarity 

### Setup & Resources  
Before the workshop, participants will receive:  
- A **GitHub repository** with example code, notebooks, and sample data  
- A **Google Colab notebook** for interactive demos  

Steps:  
1. **Access notebooks**: Open in Google Colab (preferred) or download locally  
2. **Download data**: Sample datasets included in repo (synthetic patient records, knowledge bases)  
3. **Install dependencies**: Via `requirements.txt` or provided Colab setup cells  
4. **API keys (if needed)**:  If you have an OpenAI Plus account, please get the OpenAI key as follows. 
5. **Test environment**: Run a provided snippet to confirm everything works before the session  

Participants not running code can still benefit by **observing demos** and interacting with the shared interface.  

---

## 👥 Organizers  

- **Shumit Saha** — Assistant Professor of Biomedical Data Science, Meharry Medical College, USA.  
  Research: AI, NLP, and multimodal data for chronic disease management.  

- **Subash Neupane** — Assistant Professor of Computer Science & Data Science, Meharry Medical College, USA.  
  Research: Grounding Generative AI in verifiable medical knowledge using Neuro-Symbolic AI.  

- **Broderick Bellard** — MSc candidate in Data Science, Meharry Medical College, USA.  
  Research: AI-driven healthcare, drug repurposing, equitable and explainable ML.  

- **Himanshu Tripathi** — PhD candidate in Computer Science, University of Alabama, USA.
  Research: Privacy-preserving ML.  

---

## 🙏 Acknowledgement  
This workshop was supported by the **NIH R25 program**. The organizers thank the **Meharry School of Applied Computational Sciences (SACS)** for providing logistical support.  

---

## 📚 References  
1. S. A. Anisha, A. Sen, and C. Bain. *Evaluating the Potential and Pitfalls of AI-Powered Conversational Agents as Humanlike Virtual Health Carers in the Remote Management of Noncommunicable Diseases: Scoping Review.* **Journal of Medical Internet Research**, 26(1): e56114 (2024).  
