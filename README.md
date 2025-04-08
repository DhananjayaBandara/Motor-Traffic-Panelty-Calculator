# Motor-Traffic-Panelty-Calculator

## Overview
The **Motor-Traffic-Panelty-Calculator** is an AI-powered system designed to identify traffic violations and calculate fines in Sri Lanka. It leverages Natural Language Processing (NLP) techniques such as Named Entity Recognition (NER) and text classification to process legal documents and traffic regulations. The system is built using Python (Flask/Django) for the backend and React.js for the frontend, with a centralized legal database to ensure accurate and efficient automation in traffic law enforcement and digital governance.

---

## Features
- **Traffic Violation Detection**: Automatically detects traffic violations based on predefined legal rules and regulations.
- **Fine Calculation**: Calculates fines for violations using a centralized database of penalties.
- **NLP Integration**: Uses NLP techniques like NER and text classification to extract and process legal data.
- **Legal Database**: Centralized repository of traffic laws and penalties for accurate enforcement.
- **Automation**: Supports automation in traffic law enforcement, reducing manual intervention.
- **Digital Governance**: Facilitates digital transformation in traffic management and governance.

---

## Technology Stack
### Backend
- **Python**: Core programming language.
- **Flask/Django**: Frameworks for building RESTful APIs and backend logic.
- **NLP Libraries**: Libraries like SpaCy or NLTK for text processing and classification.

### Frontend
- **React.js**: Framework for building a dynamic and responsive user interface.

### Database
- **Centralized Legal Database**: Stores traffic laws, penalties, and violation data.

### Other Tools
- **CSV/HTML Parsing**: Extracts and processes legal data from structured files.
- **AI Models**: Machine learning models for text classification and entity recognition.

---

## Folder Structure
```
Motor-Traffic-Panelty-Calculator/
│
├── principle_docs/HTML/          # Legal documents in HTML/CSV format
├── violationDetectorForPrincipleEnactment/  
│   ├── IndexFines.csv            # Mapping of violations to fines
│
├── backend/                      # Backend source code (Flask/Django)
│   ├── app/                      # Core application logic
│   ├── models/                   # Database models
│   ├── routes/                   # API endpoints
│
├── frontend/                     # Frontend source code (React.js)
│   ├── components/               # React components
│   ├── pages/                    # Application pages
│
├── tests/                        # Unit and integration tests
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn
- A virtual environment tool (e.g., `venv` or `virtualenv`)

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/DhananjayaBandara/Motor-Traffic-Panelty-Calculator.git
   cd Motor-Traffic-Panelty-Calculator
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backend server:
   ```bash
   python manage.py runserver  # For Django
   flask run                  # For Flask
   ```

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

---

## Usage
1. **Upload Legal Documents**: Upload traffic law documents in CSV or HTML format to the system.
2. **Violation Detection**: The system processes the uploaded documents and identifies violations using NLP techniques.
3. **Fine Calculation**: Based on the detected violations, the system calculates fines using the centralized legal database.
4. **API Integration**: Use the RESTful APIs to integrate the system with external applications or services.

---

## Key Files
- **`principle_docs/HTML/`**: Contains legal documents in CSV and HTML formats.
- **`violationDetectorForPrincipleEnactment/IndexFines.csv`**: Maps traffic violations to their respective fines.
- **`backend/routes/`**: Defines API endpoints for processing violations and calculating fines.
- **`frontend/components/`**: Contains React components for the user interface.

---

## Testing
1. Run backend tests:
   ```bash
   python manage.py test  # Django
   pytest                 # Flask
   ```
2. Run frontend tests:
   ```bash
   npm test
   ```

---

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## Contact
For questions or support, please contact:
- **Email**: prasannadananjaya7@gmail.com
- **GitHub Issues**: [GitHub Issues](https://github.com/DhananjayaBandara/Motor-Traffic-Panelty-Calculator/issues)
```