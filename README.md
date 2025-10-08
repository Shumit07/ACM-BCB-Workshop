# HeartWise Agentic Patient Support System

## 🚀 Agentic Capabilities Overview

This project is an **agentic AI system** designed to function as an autonomous support agent for high-risk cardiac patients. Unlike a standard chatbot that only answers questions, HeartWise perceives a user's need for direct intervention, reasons about the best course of action, and uses a suite of software tools to execute a multi-step plan to ensure patient safety.

The agent's core purpose is to bridge the gap between conversation and real-world action. When a high-risk patient signals distress, the agent autonomously performs the following functions:

* **Intent Recognition**: It actively listens for trigger phrases (e.g., "take action," "help me") to understand when a situation requires intervention beyond providing information.
* **Clinical Data Summarization**: It uses an external Large Language Model (LLM) to process complex patient data, including chat logs and existing notes, and synthesizes it into a concise clinical summary suitable for a medical professional.
* **Conditional Reasoning**: The agent's primary reasoning task is to analyze the summarized patient data to determine if a Primary Care Physician (PCP) is on file. This single decision point dictates its entire subsequent action plan.
* **Geospatial Tool Use**: If a PCP is not available, the agent autonomously uses a location intelligence toolset. It can convert a patient's address into geographic coordinates and then query an external mapping service (Overpass API) to find and rank nearby emergency rooms by distance.
* **Action Execution & Communication**: Based on its reasoning, the agent executes a final action by calling a communication tool. It either sends the clinical summary to the PCP on file or sends the summary and a list of nearby ERs directly to the patient.
* **Task Confirmation**: After completing its task, the agent reports its actions back to the user, providing a clear confirmation of the steps it has taken on their behalf.

---

## 🤖 The Agentic Workflow

The agentic capability is triggered when a user with a 'high-risk' status uses an action-oriented prompt. The system then follows a dynamic, conditional workflow to achieve its goal of ensuring patient safety.

### Workflow Diagram

### Agentic Workflow Diagram

```text
[START: User issues 'Take Action' prompt]
    │
    ▼
[Agent: agentic_chatbot.py]
    │
    └─── 1. Checks patient risk status
         ├── If 'low' or 'moderate' risk ──> [Provide reassuring canned response] ──> [STOP]
         │
         └── If 'high' risk ──> [Initiate High-Risk Protocol]
              │
              ├── 2. Fetches Patient Data from Excel file
              │
              ├── 3. Calls Tool: OpenAI_Summarization.Summary_Email()
              │    │
              │    ├── Uses OpenAI API to generate a clinical summary
              │    └── Analyzes data to check if a PCP is on file
              │
              └── 4. Reasons: Does the patient have a PCP?
                   ├── YES ──> [Action Path A: Notify PCP]
                   │    │
                   │    └── Calls Tool: Send_Email.Email()
                   │        ├── Target: PCP's Email
                   │        └── Content: Urgent message with clinical summary
                   │
                   └── NO ──> [Action Path B: Notify Patient with ERs]
                        │
                        ├── Calls Tool: Find_ER.FindER()
                        │   ├── Sub-Tool: Geocoding API (for coordinates)
                        │   └── Sub-Tool: Overpass API (for nearby hospitals)
                        │
                        └── Calls Tool: Send_Email.Email()
                            ├── Target: Patient's Email
                            └── Content: Urgent message with clinical summary & ER list
    │
    ▼
[END: Agent responds to user with confirmation of the action taken]
```
