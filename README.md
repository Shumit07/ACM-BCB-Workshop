# HeartWise Agentic Patient Support System

## 🚀 Agentic Capabilities Overview

This project is an **agentic AI system** designed to function as an autonomous support agent for high-risk cardiac patients. Unlike a standard chatbot that only answers questions, HeartWise perceives a user's need for direct intervention, reasons about the best course of action, and uses a suite of software tools to execute a multi-step plan to ensure patient safety.

The agent's core purpose is to bridge the gap between conversation and real-world action. When a high-risk patient signals distress, the agent autonomously performs the following functions:

  * 💡 **Intent Recognition**: It actively understands a user's request to know when a situation requires intervention beyond just providing information.
  * 📋 **Clinical Data Summarization**: It uses an external Large Language Model (LLM) to process complex patient data, including chat logs and existing notes, and synthesizes it into a concise clinical summary suitable for a medical professional.
  * 🤔 **Conditional Reasoning**: The agent's primary reasoning task is to analyze the summarized patient data to determine if a Primary Care Physician (PCP) is on file. This single decision point dictates its entire subsequent action plan.
  * 🗺️ **Geospatial Tool Use**: If a PCP is not available, the agent autonomously uses a location intelligence toolset. It can convert a patient's address into geographic coordinates and then query an external mapping service (Overpass API) to find and rank nearby emergency rooms by distance.
  * 📧 **Action Execution & Communication**: Based on its reasoning, the agent executes a final action by calling a communication tool. It either sends the clinical summary to the PCP on file or sends the summary and a list of nearby ERs directly to the patient.
  * ✅ **Task Confirmation**: After completing its task, the agent reports its actions back to the user, providing a clear confirmation of the steps it has taken on their behalf.

-----

## ⚙️ The Agentic Workflow

The agentic capability is triggered when a user with a 'high-risk' status uses an action-oriented prompt. The system then follows a dynamic, conditional workflow to achieve its goal of ensuring patient safety.

#### Prerequisites for HeartWise Emergency Response System

<ol>
  <li> OpenAI API Key - For clinical summary generation via GPT models </li>
  <li> Google Account with App Password - For automated email sending via Gmail SMTP </li>
  <li> Geolocation API Access - Nominatim geocoding service (free, no key required) </li>
  <li> Overpass API Access - OpenStreetMap hospital data (free, no key required) </li>
</ol>

### 📊 Agentic Workflow Diagram

```text
[START: Patient expresses distress/need for help] 🆘
    │
    ▼
[SYSTEM: Emergency Response Protocol Activated]
    │
    └─── 1. Risk Assessment Check 📊
         │
         ├── Low/Moderate Risk ──> [Supportive Guidance] ──> [STOP] 🛑
         │   └── "Continue care plan"
         │
         └── HIGH RISK ──> [Emergency Protocol Initiated] 🚨
              │
              ▼
         2. Patient Data Aggregation 📋
              ├── Medical history retrieved
              ├── Recent symptoms analyzed
              ├── Chat logs compiled
              └── Contact information verified
              │
              ▼
         3. Clinical Summary Generation 🏥
              ├── AI analyzes complete patient profile
              ├── Creates medical-grade summary
              └── Formats for rapid clinical review
              │
              ▼
         4. Care Coordination Decision Point 🔀
              │
              ├── PATH A: Has Primary Care Physician ──────┐
              │                                            │
              └── PATH B: No PCP on File ───────┐          │
                   │                            │          │
                   ▼                            │          ▼
         5B. Emergency Room Locator             │    5A. PCP Notification 👨‍⚕️
              ├── Geocoding activated           │         ├── Urgent email sent
              ├── Hospital search (expanding)   │         ├── Clinical summary provided for help
              ├── 5 nearest ERs identified      │              │
              └── Distance calculation          │              │
                   │                            │              │
                   ▼                            │              ▼
         6B. Patient Direct Notification        │    6A. Confirmation to Patient ✅
              ├── Email with clinical summary   │         └── "Your doctor has been notified"
              ├── ER list with distances        │
              ├── Interactive map provided      │
              └── "Seek immediate care"         │
                   │                            │
                   └────────────┬───────────────┘
                                │
                                ▼
[END: Patient receives confirmation + next steps + visual aids] ✅
```

### 📊 Agentic Codeflow Diagram

```text
[START: User issues 'Take Action' prompt]
    │
    ▼
[Agent: agentic_chatbot.py]
    │
    └─── 1. Checks patient risk status
         ├── If 'low' or 'moderate' risk ──> [Provide reassuring canned response] ──> [STOP] 🛑
         │
         └── If 'high' risk ──> [Initiate High-Risk Protocol] 🚀
              │
              ├── 2. Fetches Patient Data from Excel file 📁
              │
              ├── 3. Calls Tool: OpenAI_Summarization.Summary_Email() 🧠
              │    │
              │    ├── Uses OpenAI API to generate a clinical summary
              │    └── Analyzes data to check if a PCP is on file
              │
              └── 4. Reasons: Does the patient have a PCP? 🤔
                   ├── YES ──> [Action Path A: Notify PCP] 👨‍⚕️
                   │    │
                   │    └── Calls Tool: Send_Email.Email() 📧
                   │        ├── Target: PCP's Email
                   │        └── Content: Urgent message with clinical summary
                   │
                   └── NO ──> [Action Path B: Notify Patient with ERs] 🏥
                        │
                        ├── Calls Tool: Find_ER.FindER() 🗺️
                        │   ├── Sub-Tool: Geocoding API (for coordinates)
                        │   └── Sub-Tool: Overpass API (for nearby hospitals)
                        │
                        └── Calls Tool: Send_Email.Email() 📧
                            ├── Target: Patient's Email
                            └── Content: Urgent message with clinical summary & ER list
    │
    ▼
[END: Agent responds to user with confirmation of the action taken] ✅
```
