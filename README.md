# IntelliInspect: Real-Time Predictive Quality Control

IntelliInspect is a comprehensive AI-powered application that enables real-time quality control prediction using Kaggle Production Line sensor data. This full-stack solution combines modern web technologies with advanced machine learning to simulate manufacturing defect detection in real-time, helping organizations optimize their quality control processes through predictive analytics.

## Architecture Overview

![Architecture Diagram](data/Diagrams/Software%20Architecture%20Diagram.png) Three-tier architecture orchestrating an Angular frontend, .NET Core backend, and Python FastAPI ML service. The system handles large-scale data processing through chunked uploads, trains XGBoost models for defect prediction, and provides real-time simulation streaming through WebSocket connections—all containerized and deployed using Docker Compose.

## Features Scorecard

### ✅ Must Have Features (100% Complete)

| Feature | Status | Description |
|---------|---------|-------------|
| **Screen 1: Upload Dataset** | ✅ Complete | Drag-and-drop interface with chunked upload support for files up to 2.5GB |
| **Screen 2: Date Ranges** | ✅ Complete | Calendar-based date pickers with intelligent validation and timeline visualization |
| **Screen 3: Model Training** | ✅ Complete | XGBoost model training with comprehensive metrics and performance charts |
| **Screen 4: Real-Time Simulation** | ✅ Complete | WebSocket-powered streaming predictions with live dashboards |
| **Angular Frontend (v18+)** | ✅ Complete | Professional UI with step-based navigation and theme management |
| **ASP.NET Core 8 Backend** | ✅ Complete | Robust API gateway with JWT authentication and WebSocket proxying |
| **Python 3.13 + FastAPI ML Service** | ✅ Complete | Sophisticated machine learning pipeline with XGBoost optimization |
| **Docker Deployment** | ✅ Complete | Complete containerization with docker-compose orchestration |

### ✅ Good to Have Features (Substantially Complete)

| Feature | Status | Description |
|---------|---------|-------------|
| **Feature Importance Visualization** | ✅ Complete | Base64-encoded plots showing model insights and feature contributions |
| **Live Streaming Charts** | ✅ Complete | Real-time confidence score visualization and prediction statistics |
| **Performance Optimization** | ✅ Complete | Chunked uploads and memory-efficient processing for large datasets |
| **Retry/Resume Logic** | ✅ Complete | Robust simulation state management with start/stop controls |

### ✅ Nice to Have Features (Bonus Achievements)

| Feature | Status | Description |
|---------|---------|-------------|
| **Authentication System** | ✅ Complete | JWT-based security with admin/user roles |
| **Theme Toggle** | ✅ Complete | Dynamic theming system with legacy/new visual modes |
| **Advanced Error Handling** | ✅ Complete | Comprehensive error propagation and user feedback systems |
| **Performance Optimization** | ✅ Complete | Chunked file processing for large CSV datasets |

## Data Flow Overview

![Data Flow Diagram](data/data flow architecture demonstrates how information moves through the system from initial CSV upload through chunked processing, model training, and real-time simulation streaming. The system employs a sophisticated proxy pattern where the .NET backend orchestrates communication between the Angular frontend and Python ML service, ensuring secure, scalable data processing.

## Deployment Instructions

### Prerequisites
- Docker and Docker Compose installed on your system
- Minimum 8GB RAM recommended for optimal performance
- Available ports: 4200 (frontend), 5000 (backend), 8000 (ML service)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone 
   cd intelliinspect
   ```

2. **Deploy the entire application:**
   ```bash
   sudo docker compose up --build -d
   ```

3. **Access the application:**
   - Open your web browser and navigate to `http://localhost:4200`
   - The application will be ready when all containers are running

4. **Verify deployment:**
   ```bash
   docker compose ps
   ```
   All services should show "running" status.

### Stopping the Application

```bash
sudo docker compose down
```

## Usage Guide

### Step 1: Upload Dataset
- Navigate to the **Upload Dataset** screen
- Use the drag-and-drop interface or click "Choose File" to upload your CSV dataset
- The system supports files up to 2.5GB using sophisticated chunked upload processing
- View extracted metadata including total records, columns, pass rate, and date range
- Proceed to the next step once file processing is complete

### Step 2: Define Date Ranges
- Configure three sequential, non-overlapping date ranges:
  - **Training Period**: Historical data for model training
  - **Testing Period**: Data for model evaluation and validation
  - **Simulation Period**: Data for real-time prediction simulation
- Use calendar-based date pickers with intelligent validation
- View timeline visualization showing data distribution across periods
- Validate ranges against the actual dataset before proceeding

### Step 3: Train Model
- Initiate XGBoost model training using the configured date ranges
- Monitor training progress with real-time status updates
- View comprehensive performance metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix visualization (donut chart)
  - Feature importance plots
- Training artifacts are automatically saved for simulation use

### Step 4: Real-Time Simulation
- Start the WebSocket-powered simulation to stream predictions in real-time
- Monitor live visualizations:
  - Real-time quality predictions line chart
  - Prediction confidence distribution (donut chart)
  - Live statistics: total predictions, pass/fail counts, average confidence
  - Streaming prediction table with timestamps and confidence scores
- Control simulation with start/stop functionality
- View final simulation summary upon completion

## Technology Stack

- **Frontend**: Angular v18+ with TypeScript, Chart.js for visualizations
- **Backend**: ASP.NET Core 8 with JWT authentication and WebSocket support  
- **ML Service**: Python 3.13, FastAPI, XGBoost, pandas, scikit-learn
- **Data Storage**: Redis for caching, Docker volumes for persistent storage
- **Deployment**: Docker containers orchestrated with Docker Compose
- **Networking**: Custom bridge network for secure service communication

## Authentication

The application includes JWT-based authentication with predefined accounts:
- **Admin**: Username: `admin`, Password: `admin12.`
- **User**: Username: `user`, Password: `user12.`

*Note: This is a demonstration authentication system. Production deployments should implement proper user management with a database backend.*