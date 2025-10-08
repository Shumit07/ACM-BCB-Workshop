# HeartWise Agentic Patient Support System

## 🚀 Overview

This project showcases an advanced, agentic chatbot system named **HeartWise**. It is designed to assist patients by not only providing information but also by autonomously taking action in high-risk scenarios. When a high-risk patient expresses a need for help (e.g., "take action," "help me"), the system initiates a multi-step protocol to ensure the patient receives the appropriate level of care.

The core of this system lies in its ability to **perceive**, **reason**, and **act**. It perceives the user's intent, reasons about the best course of action based on the patient's specific context (like whether they have a primary care physician), and then acts by executing a series of tool calls—summarizing clinical data, locating emergency rooms, and dispatching notifications.

---

## 🤖 The Agentic Workflow

The agentic capability is triggered when a user with a 'high-risk' status uses an action-oriented prompt. The system then follows a dynamic, conditional workflow to achieve its goal of ensuring patient safety.

### Workflow Diagram

```mermaid
graph TD
    A[Start: User issues 'Take Action' prompt] --> B{Agent: agentic_chatbot.py};
    B --> C{Risk Status Check};
    C -->|'low' or 'moderate'| D[Provide reassuring canned response];
    C -->|'high'| E[Initiate High-Risk Protocol: High_Risk_Patient_Action(id)];
    
    subgraph High-Risk Protocol
    E --> F[1. Fetch Patient Data from Excel];
    F --> G[2. Call Clinical Analyst Tool: OpenAI_Summarization];
    G --> H[3. OpenAI API: Generate Clinical Summary];
    H --> I{4. Reason: Does Patient have a PCP?};
    
    I -->|Yes| J[Action 1: Notify PCP];
    J --> K[Tool: Send_Email.Email() to PCP];
    K --> L[Format Summary & Send];
    
    I -->|No| M[Action 2: Notify Patient with ERs];
    M --> N[Tool: Find_ER.FindER(location)];
    N --> O[Sub-Tool: Geocoding API to get coordinates];
    O --> P[Sub-Tool: Overpass API to find hospitals];
    P --> Q[Format Top 5 ERs];
    Q --> R[Tool: Send_Email.Email() to Patient];
    R --> S[Format Summary & ER List & Send];
    end

    L --> T{End: Respond to User};
    S --> T;
    D --> T;
    T --> |"Protocol initiated: An urgent notification..."| A;
