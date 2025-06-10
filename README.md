# POC-GraphRAG

A Proof of Concept (POC) project implementing a Graph-based Retrieval-Augmented Generation (GraphRAG) system, combining a backend server for processing and a frontend AI chatbot interface.

## Overview

This project demonstrates a GraphRAG implementation, leveraging a backend server to handle data processing and a frontend built with modern web technologies for user interaction. The backend is implemented in Python, while the frontend is a chatbot interface built using a JavaScript-based framework.

## Prerequisites

To run this project, ensure you have the following installed:

- **Backend**:
  - Python 3.8+
  - Dependencies listed in `graphrag/requirements.txt`

- **Frontend**:
  - Node.js (version 16 or higher)
  - pnpm (Node package manager)

## Getting Started

Follow the instructions below to set up and run the backend and frontend components.

### Running the Backend

1. Navigate to the `graphrag` directory from the project root:
   ```bash
   cd graphrag
   ```
2. Install python dependencies from `./graphrag/requirements.txt`
   
3. Start the backend server:
   ```bash
   python ./graphrag/api/server.py
   ```

The backend server will start and be ready to handle requests.

### Running the Frontend

1. Navigate to the `ai-chatbot` directory from the project root:
   ```bash
   cd ai-chatbot
   ```

2. Install the frontend dependencies:
   ```bash
   pnpm install
   ```

3. Start the development server:
   ```bash
   pnpm dev
   ```

The frontend will be available at the URL provided in the terminal (typically `http://localhost:5173`).

## Project Structure

- **`graphrag/`**: Contains the backend code, including the GraphRAG implementation and API server.
- **`ai-chatbot/`**: Contains the frontend code for the AI chatbot interface.
- Other directories and files may include configuration, data, or utility scripts.

## Usage

- **Backend**: The backend server processes graph-based queries and serves data to the frontend. Ensure it is running before starting the frontend.
- **Frontend**: The chatbot interface allows users to interact with the GraphRAG system. Open the provided URL in a browser to access the chatbot.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the maintainers.
