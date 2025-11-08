## 🚀 Local Agent Setup (Windows)

This guide provides the necessary steps to clone the repository and set up the Python environment to run the agent locally on a Windows system.

---

### Prerequisites

* **Git** installed on your system.
* **Python 3** (preferably 3.8+) installed and added to your system's PATH.

---

### Installation Steps

1.  **Clone the Repository**
    Open your terminal (e.g., Command Prompt, PowerShell, or Git Bash) and execute the following command to download the project files:
    ```bash
    git clone [repository-url]
    ```
2.  **Navigate to the Project Directory**
    Change the current working directory to the project folder:
    ```bash
    cd ./repo-name
    ```
    (Replace `repo-name` with the actual name of the cloned directory.)

3.  **Create a Virtual Environment**
    Create a dedicated virtual environment named `venv` to isolate project dependencies:
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment**
    Activate the new environment. Use the command appropriate for your terminal:
    * **PowerShell:**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    * **Command Prompt:**
        ```cmd
        .\venv\Scripts\activate.bat
        ```

5.  **Install Dependencies**
    Install all required project packages listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Agent**
    Start the agent monitor script:
    ```powershell
    .\monitor.ps1
    ```
    