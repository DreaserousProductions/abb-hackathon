# Angular Frontend Workflow Documentation

## Overview

This Angular application provides a comprehensive machine learning workflow interface that guides users through a four-step process: file upload, data preprocessing, model training, and real-time simulation. The application employs a single-page architecture with step-based navigation, real-time WebSocket communication, and dynamic theming capabilities.

## Application Entry & Theming

The application initializes with a theme management system that allows users to toggle between "legacy" and "new" visual themes. The **ThemeService** acts as a centralized theme controller, using an observable pattern to broadcast theme changes across all components. When the application loads, components subscribe to theme updates and dynamically apply CSS classes to their host elements, ensuring consistent visual presentation throughout the user journey.

The routing system protects the main workflow behind authentication guards. Users must first authenticate through the login component, which validates credentials against the backend API and stores JWT tokens locally. Once authenticated, users are redirected to the main upload component, where the core workflow begins.

## The Multi-Step Upload Process

### Step 1: File Upload
The upload process begins with an intuitive drag-and-drop interface that accepts CSV files up to 2.5GB. The application implements a sophisticated chunked upload mechanism to handle large files reliably. When a user selects or drops a file, the frontend validates the file type and size before initiating the upload process.

The **UploadService** orchestrates the chunked upload by dividing large files into smaller segments and uploading them sequentially to the backend. Each chunk is processed individually, allowing for better error handling and progress tracking. The service communicates upload progress back to the UI through observables, enabling real-time progress bar updates. Once all chunks are successfully uploaded, the backend reassembles the file and processes it into a Parquet format, returning metadata including dataset ID, record counts, and date range information.

### Step 2: Date Range Configuration
After successful file processing, users enter the preprocessing step where they define temporal boundaries for their machine learning workflow. The interface presents three distinct date range inputs: training, testing, and simulation periods. The frontend implements client-side validation to ensure logical date progression and prevents overlapping ranges.

The **UploadService** facilitates backend validation by sending the configured date ranges to the server for verification against the actual dataset. The backend responds with record counts for each period and monthly distribution data. This validation step ensures that each time period contains sufficient data for meaningful analysis. The frontend displays visual feedback through timeline charts and data distribution visualizations, helping users understand their data segmentation choices.

### Step 3: Model Training
The training phase represents the most computationally intensive step of the workflow. When users initiate training, the frontend sends the validated date ranges and dataset metadata to the backend for XGBoost model training. Since this process can take considerable time, the application implements a sophisticated progress simulation system.

The frontend displays an animated progress bar that increments automatically over time intervals, providing visual feedback while the actual training occurs in the background. When the backend completes training, it returns comprehensive performance metrics including accuracy, precision, recall, and F1 scores, along with confusion matrix values and feature importance visualizations.

The component processes these results to generate interactive charts and performance summaries. The confusion matrix is rendered as a visual donut chart, while feature importance data is displayed through the backend-generated plots. This approach ensures users receive immediate feedback about their model's performance characteristics.

### Step 4: Real-Time Simulation
The final step establishes a WebSocket connection for real-time model prediction simulation. The **SimulationService** manages the bidirectional communication channel with the backend, handling connection lifecycle events and message routing.

When users start a simulation, the service sends initialization commands containing user and dataset identifiers. The backend begins streaming prediction results, which include row indices, prediction values, confidence scores, actual values, and correctness indicators. The frontend processes this streaming data to update multiple real-time visualizations simultaneously.

The application maintains running statistics including total predictions, pass/fail counts, accuracy metrics, and confidence distributions. A real-time line chart displays confidence scores over time, while summary statistics are continuously updated. Users can stop the simulation at any point, triggering a graceful WebSocket disconnection.

## State Management Architecture

The **upload.component.ts** serves as the central state controller for the entire workflow, orchestrating interactions between multiple services and managing the application's complex state transitions. This component maintains comprehensive state objects for each workflow step, including file upload status, date range configurations, training progress, and simulation statistics.

The component employs a reactive programming approach using RxJS observables to handle asynchronous operations and state updates. Service subscriptions are carefully managed to prevent memory leaks, with proper cleanup implemented in the component's destruction lifecycle.

Navigation between steps is controlled through a step-based system that tracks user progress and enforces sequential completion requirements. Users can navigate backward to previous steps but cannot advance until current step requirements are satisfied. This approach ensures data integrity and prevents incomplete workflow execution.

The state management system also handles error scenarios gracefully, providing appropriate user feedback and recovery mechanisms. When errors occur, the component resets relevant state sections and provides clear guidance for user correction.

## Data Flow and Integration

Throughout the workflow, data flows seamlessly between the frontend and backend through well-defined API contracts. The frontend maintains local state representations that mirror backend data structures, ensuring consistency across the application. Chart.js integration provides dynamic visualizations that update in response to data changes, creating an interactive and informative user experience.

The application's architecture supports both one-time operations (file upload, training) and continuous data streams (simulation), demonstrating a versatile approach to different types of user interactions. This design ensures scalability and maintainability while providing a smooth, professional user experience for complex machine learning workflows.

# .NET Backend Workflow and Architecture Documentation

## Overview

This .NET Core backend serves as an orchestration layer between an Angular frontend and a Python-based machine learning service, providing secure authentication, file processing capabilities, and real-time simulation streaming. The application implements a three-tier architecture where the .NET backend acts as both an API gateway and a WebSocket proxy, facilitating seamless communication between disparate services.

## Application Startup & Configuration

The application initializes through **Program.cs**, which establishes the foundational infrastructure for the entire system. The startup process configures several critical components that enable secure, cross-origin communication and service integration.[1]

**CORS Configuration**: The application implements a permissive CORS policy named "AllowAngularApp" that specifically allows connections from localhost:4200 and localhost origins. This policy enables credentials, headers, and methods necessary for the Angular frontend to communicate securely with the backend. The configuration supports both development and production scenarios where the frontend and backend operate on different ports.[1]

**JWT Authentication Pipeline**: The system implements industry-standard JWT token-based authentication using symmetric key encryption. The authentication pipeline validates tokens against configured issuer, audience, and signing key parameters, with tokens expiring after 30 minutes. This security layer protects all data manipulation and processing endpoints while allowing public access to authentication endpoints.[1]

**ML Service Integration**: The application configures environment-specific HttpClient instances for communicating with the Python ML service. In development mode, it targets localhost:8000, while in production it connects to ml_service:8000, supporting both local development and containerized deployment scenarios. The ML service client includes extended timeout configurations (3 minutes) to accommodate long-running training operations.[1]

**WebSocket Support**: The application enables WebSocket functionality through the middleware pipeline, essential for real-time simulation streaming capabilities.[1]

## Controller Responsibilities

### AuthController: Security Gateway
The **AuthController** implements a temporary authentication system using in-memory user storage without database persistence. The system includes two predefined users: an admin account (username: "admin", password: "admin12.") and a standard user account (username: "user", password: "user12."). **This represents a temporary authentication mechanism designed for demonstration purposes and lacks proper database integration**.[2][3]

The controller provides two primary endpoints: a login endpoint that validates credentials against BCrypt-hashed passwords and generates JWT tokens, and a verification endpoint that validates existing tokens and returns user role information. Upon successful authentication, the system returns both the JWT token and username, enabling the frontend to maintain authenticated sessions.[2]

### DataController: File Processing Orchestrator
The **DataController** serves as the primary interface for all data-related operations, implementing sophisticated chunked file upload capabilities and data validation workflows. This controller acts as a intelligent proxy, translating frontend requests into appropriate ML service communications while maintaining consistent error handling and response formatting.[4]

**Chunked Upload Architecture**: The system supports large file uploads (up to 2.5GB) through a three-phase process. The upload-chunk endpoint receives individual file segments along with upload identifiers, chunk indices, and user identifiers, forwarding these to the ML service for temporary storage. The finish-upload endpoint coordinates the reassembly of chunks into complete files and triggers the conversion to Parquet format for efficient processing.[4]

**Data Validation Pipeline**: The validate-ranges endpoint facilitates temporal data validation by forwarding date range configurations to the ML service and returning record counts and distribution information. This enables the frontend to make informed decisions about training, testing, and simulation periods based on actual data availability.[4]

### ModelController: Training and Simulation Coordinator
The **ModelController** orchestrates machine learning operations and manages real-time simulation streaming. This controller handles two distinct but related responsibilities: coordinating model training requests and managing WebSocket-based simulation streaming.[5]

**Training Orchestration**: The training endpoint accepts comprehensive training requests including user identifiers, dataset identifiers, and date range configurations, forwarding these to the ML service with appropriate timeout handling. The controller manages the extended processing times required for XGBoost model training while providing appropriate error handling and response formatting.[5]

**WebSocket Proxy Architecture**: The simulation-ws endpoint implements a sophisticated bidirectional WebSocket proxy that maintains simultaneous connections to both the Angular frontend and the Python ML service. This architecture enables real-time streaming of prediction results while abstracting the complexity of dual WebSocket management from the frontend application.[5]

## ML Service Integration

The backend implements a comprehensive proxy pattern for ML service communication, abstracting the Python FastAPI service behind a consistent .NET API interface. All ML service interactions utilize named HttpClient instances configured with environment-specific base URLs and extended timeouts to accommodate processing-intensive operations.[4][5][1]

**Request Translation**: The system automatically translates between .NET naming conventions (PascalCase) and Python conventions (camelCase) using JsonSerializer configuration, ensuring seamless data exchange between services. This translation layer eliminates potential serialization issues while maintaining clean, idiomatic code in both services.[5][4]

**Error Handling and Propagation**: The backend implements transparent error propagation, forwarding ML service error responses directly to the frontend while adding contextual information about the operation that failed. This approach maintains error fidelity while providing clear debugging information.[4][5]

**Environment-Aware Configuration**: The system automatically adapts to different deployment environments, using localhost addresses for development and container-based addressing for production deployments. This configuration supports seamless transitions between development, testing, and production environments.[5][1]

## WebSocket Proxy Logic

The **ModelController** implements a sophisticated WebSocket proxy that manages simultaneous connections to both the frontend and ML service. This architecture enables real-time, bidirectional communication while maintaining the security and consistency benefits of the .NET middleware pipeline.[5]

**Dual Connection Management**: When a WebSocket connection is requested, the system accepts the connection from the Angular frontend and simultaneously establishes a new connection to the FastAPI service. The proxy maintains both connections throughout the simulation session, ensuring reliable message delivery in both directions.[5]

**Message Relay Architecture**: The system implements concurrent message relay tasks that continuously monitor both WebSocket connections for incoming messages. When messages are received from either endpoint, they are immediately forwarded to the corresponding destination, creating a transparent communication channel.[5]

**Connection Lifecycle Management**: The proxy handles connection cleanup gracefully, ensuring that when one connection closes, the corresponding connection is also closed properly. This prevents resource leaks and ensures consistent state management across the simulation workflow.[5]

The WebSocket proxy architecture enables real-time machine learning simulation results to stream directly from the Python service to the Angular frontend while maintaining the authentication and authorization benefits of the .NET middleware pipeline. This design provides the performance benefits of direct WebSocket communication while preserving the security and logging capabilities of the backend infrastructure.

This architectural approach demonstrates a sophisticated understanding of modern distributed system design, providing a scalable, maintainable solution that effectively bridges different technology stacks while maintaining security, performance, and reliability requirements.

# FastAPI ML Service Workflow and Architecture Documentation

## Overview

The FastAPI ML service operates as a specialized machine learning backend that processes large-scale manufacturing data, trains XGBoost models for defect prediction, and provides real-time simulation capabilities. The service implements a modular architecture with clear separation between API routing, business logic, and data persistence, specifically designed to handle the computational demands of Bosch production line analysis with extreme class imbalance scenarios.

## API Layer Architecture

The FastAPI application initializes through **main.py**, which orchestrates four specialized routers that handle distinct aspects of the machine learning workflow. Each router maintains focused responsibilities while leveraging shared Pydantic models for request validation and response formatting.

**Router Responsibilities**: The **csv_upload** router manages chunked file uploads up to 2.5GB, implementing a two-phase process where individual chunks are received and temporarily stored before being reassembled into complete CSV files. The **csv_validation** router validates temporal date ranges against processed datasets, ensuring that training, testing, and simulation periods contain sufficient data for meaningful analysis. The **csv_training** router coordinates the complex model training pipeline, managing long-running XGBoost operations with sophisticated hyperparameter optimization. The **csv_simulation** router establishes WebSocket connections for real-time prediction streaming.

**Request Validation Framework**: Pydantic models provide comprehensive input validation and automatic documentation generation. The **ValidateRangesRequest** model ensures proper date range formatting and completeness, while **TrainModelRequest** validates training configurations including user identifiers and dataset specifications. The **FinishUploadPayload** model coordinates chunk reassembly parameters, and **TrainModelResponse** structures complex training results including performance metrics and Base64-encoded visualization plots.

## Service Layer: Core Machine Learning Pipeline

### Data Processing Workflow
The **data_processing_service** implements a revolutionary single-pass CSV processing algorithm that dramatically outperforms traditional two-pass approaches. When chunks are reassembled, the service processes massive CSV files (potentially millions of rows) through a streaming pipeline that simultaneously extracts metadata, performs data transformations, and writes optimized Parquet format output.

**Chunk Reassembly and Processing**: The system assembles uploaded chunks in sequential order, immediately feeding the complete CSV into the single-pass processor. This processor maintains running statistics including record counts, pass rates, and string-based date range extraction while converting data to compressed Parquet format. The algorithm avoids loading entire datasets into memory by processing data in 20,000-row chunks, enabling handling of multi-gigabyte files on standard hardware.

**Temporal Validation**: The service implements intelligent filename-based validation that extracts dataset temporal boundaries without loading complete files. When date ranges are submitted for validation, the system first performs boundary checks against embedded filename timestamps before calculating estimated record counts based on one-record-per-second assumptions, providing rapid feedback without expensive I/O operations.

### Advanced Model Training Pipeline
The **model_training_service** implements a sophisticated XGBoost training pipeline specifically optimized for Bosch production line defect detection, incorporating advanced techniques from manufacturing R methodologies that achieve precision rates exceeding 48%.

**Feature Engineering and Selection**: The training pipeline automatically identifies Bosch-specific feature patterns (L*_S*_F* naming conventions) and constructs engineered features including station-based aggregations, line-level statistics, and temporal pathway features. The system implements multi-stage feature selection using sparsity thresholds and variance filtering to manage the 970+ available features while maintaining computational efficiency.

**Precision-Optimized Training**: Given the extreme class imbalance typical in manufacturing defect data (often 1000:1 ratios), the service employs ADASYN resampling for superior boundary learning compared to traditional SMOTE approaches. The XGBoost configuration utilizes precision-optimized hyperparameters including scale_pos_weight adjustments, regularization parameters, and early stopping criteria specifically tuned for manufacturing scenarios.

**Threshold Optimization**: The system performs comprehensive threshold search across extended ranges (0.1 to 0.99) to maximize precision while maintaining minimum viable recall rates. This sophisticated threshold selection ensures that the final model prioritizes precision over recall, critical for manufacturing quality control where false positives are extremely costly.

## WebSocket and Real-Time Simulation Architecture

### Simulation Connection Management
The **csv_simulation** router implements a sophisticated WebSocket architecture through the **SimulationConnectionManager** class, which maintains active connection state and provides graceful stop signal coordination. This manager enables clients to start and stop simulations dynamically while ensuring proper resource cleanup and state consistency.

**Connection Lifecycle**: When clients establish WebSocket connections, the manager creates unique simulation identifiers combining user and dataset information. The system supports concurrent simulations for different users while preventing duplicate simulations for the same dataset. Stop signals propagate through asyncio Event objects, enabling immediate simulation termination without data corruption.

### Real-Time Prediction Streaming
The **simulation_service** orchestrates the real-time prediction pipeline by loading trained XGBoost models and preprocessed simulation datasets, then streaming individual predictions at one-second intervals to simulate production line monitoring scenarios.

**Data Pipeline Integration**: Simulation data undergoes identical preprocessing pipelines used during training, including multi-stage imputation, feature engineering, and variance filtering. This ensures prediction consistency between training and inference phases. The service loads preprocessed simulation data from the data_store directory and applies the trained model to generate predictions with confidence scores.

**Streaming Architecture**: The **run_simulation_for_websocket** generator function yields prediction results asynchronously, including row identifiers, prediction values, confidence scores, actual responses, and correctness indicators. This streaming approach enables real-time dashboard updates while maintaining memory efficiency through generator-based iteration rather than batch processing.

## Data Persistence and Storage Strategy

### Parquet-Based Data Storage
The service utilizes a **data_store** directory for persistent storage of processed datasets in optimized Parquet format. This approach provides several advantages including columnar compression, fast analytical queries, and schema preservation. Processed files incorporate smart naming conventions that embed temporal boundaries and dataset identifiers, enabling rapid file location and validation without database dependencies.

**File Lifecycle Management**: Original CSV files are automatically deleted after successful Parquet conversion to conserve storage space. Simulation-ready datasets are generated during training and persisted separately, ensuring that real-time simulation operations do not require expensive preprocessing during WebSocket connections.

### Redis Integration for Temporary Data
While the current implementation primarily uses file-based storage, the **redis_client** module provides infrastructure for temporary data caching and session management. Redis serves as a high-performance cache for frequently accessed metadata and provides mechanisms for storing serialized DataFrames with configurable expiration times.

**Caching Strategy**: The Redis implementation includes automatic JSON serialization for pandas DataFrames, connection pooling for performance optimization, and comprehensive error handling for network resilience. This infrastructure supports future enhancements including distributed processing and temporary result caching.

## Performance and Scalability Considerations

The FastAPI service architecture emphasizes memory efficiency and computational optimization throughout the machine learning pipeline. The single-pass CSV processing eliminates redundant file I/O operations, while chunk-based processing ensures consistent memory usage regardless of input file size. The modular service design enables horizontal scaling through containerization, with each service component independently deployable and configurable.

The WebSocket simulation architecture supports multiple concurrent connections while maintaining resource isolation between different user sessions. The XGBoost training pipeline includes comprehensive garbage collection and memory cleanup to prevent resource leaks during long-running training operations. This design ensures reliable performance under production workloads while maintaining the flexibility to handle diverse manufacturing datasets and use cases.

# Docker Container Orchestration Workflow Analysis

## Build & Dependency Workflow

The application employs a sophisticated multi-stage build strategy across all services, optimizing both build efficiency and runtime footprint. Docker Compose orchestrates the build process by constructing images in dependency order, leveraging Docker's layer caching to minimize rebuild times during development iterations.

**Build Architecture**: Each service implements multi-stage builds that separate build-time dependencies from runtime requirements. The frontend builds Angular assets using Node.js before transferring them to a lightweight Nginx container. The backend compiles .NET applications in an SDK container before deploying to a minimal ASP.NET runtime. The ML service installs Python scientific computing libraries in a build stage before copying only the virtual environment to a slim production image.

**Dependency Orchestration**: The startup sequence follows a logical data flow pattern: Redis initializes first as the foundational data layer, followed by the ML service that requires Redis connectivity, then the backend that depends on ML service availability, and finally the frontend that relies on backend API endpoints. This dependency chain ensures that each service has its required dependencies available before attempting connections, reducing startup failures and connection timeouts.

**Image Optimization**: The multi-stage approach significantly reduces final image sizes by excluding build tools, development dependencies, and temporary artifacts. This optimization improves deployment speed, reduces network transfer times, and minimizes the attack surface of production containers.

## Networking and Service Communication

The application utilizes a custom bridge network that establishes secure, isolated communication channels between services while maintaining clean service discovery patterns. This network architecture enables services to communicate using predictable DNS names while preventing external interference with internal service communication.

**Service Discovery**: The custom "abb-network" bridge automatically provides DNS resolution where services can reference each other by their compose service names. The backend communicates with the ML service via "http://ml_service:8000", while the frontend proxy directs API requests to "backend:5000". This approach eliminates the need for hardcoded IP addresses and provides resilience against container IP changes during scaling or restarts.

**Traffic Flow Architecture**: External traffic enters through the frontend Nginx proxy on port 4200, which intelligently routes API requests to the backend while serving static Angular assets directly. The backend acts as an orchestration layer, forwarding ML-specific requests to the Python service and managing WebSocket proxy connections for real-time simulation streaming. Internal communication remains isolated within the bridge network, ensuring security and performance optimization.

**WebSocket Proxying**: The Nginx configuration includes sophisticated WebSocket support with intelligent connection header mapping, enabling seamless real-time communication between the Angular frontend and FastAPI ML service through the .NET backend proxy. This three-tier WebSocket architecture maintains authentication and logging benefits while providing low-latency streaming capabilities.

## Data Persistence and Volumes

The orchestration implements strategic data persistence through carefully designed volume mappings that ensure critical application state survives container lifecycle events while maintaining development workflow efficiency.

**Persistent Storage Strategy**: The ML service mounts two critical host directories: "data_store" for processed Parquet datasets and "models" for trained XGBoost artifacts. This persistence strategy ensures that expensive data processing operations and time-intensive model training results remain available across container restarts, deployments, and scaling events.

**Development Workflow Integration**: The volume configuration supports development workflows by allowing direct access to generated datasets and models from the host system. Data scientists can examine processed files, backup trained models, and perform offline analysis without container introspection, significantly improving development productivity and debugging capabilities.

**State Management**: Redis operates as an ephemeral cache without persistent volumes, reflecting its role as a temporary data store for simulation sessions and intermediate processing results. This design choice optimizes container startup times while ensuring that critical long-term state resides in the persistent file system volumes.

## Actionable Architectural Insights

### Potential Bottlenecks

**Resource Allocation Concerns**: The current configuration lacks explicit resource limits, creating potential for resource contention during concurrent ML training operations or large file processing. The ML service, in particular, may consume excessive CPU and memory during XGBoost training, potentially impacting other services' responsiveness.

**Sequential Startup Dependencies**: The linear dependency chain creates a fragile startup sequence where failure of any upstream service prevents the entire application from becoming operational. During development or in environments with intermittent connectivity, this design may cause unnecessary startup delays or failures.

**Single Point of Failure**: The backend service acts as a critical proxy point for all ML operations and WebSocket connections. High traffic volumes or backend service failures would completely disable ML functionality, creating a bottleneck that could impact system availability.

### Security Considerations

**Privileged Execution Context**: All containers run as root users, violating security best practices and expanding the potential attack surface. Implementing dedicated user accounts within containers would significantly reduce privilege escalation risks.

**Network Exposure**: The current configuration exposes all service ports to the host system, potentially allowing direct access to internal services bypassing the intended frontend proxy. Production deployments should restrict external access to only the frontend service port.

**Secrets Management**: The configuration lacks explicit secrets management for potentially sensitive configuration values like Redis connections or internal API keys, relying instead on environment variables without encryption or rotation capabilities.

### Scalability & Best Practices

**Production Readiness Recommendations**: Implement comprehensive health checks for all services to enable Docker's restart policies and load balancer integration. Add resource limits and reservations to prevent resource starvation, particularly for the computationally intensive ML service. Consider implementing horizontal scaling for the backend service to handle increased API traffic.

**Operational Excellence**: Introduce structured logging with centralized log aggregation to improve debugging and monitoring capabilities. Implement service mesh or API gateway patterns to provide advanced traffic management, circuit breaking, and observability features. Consider migrating from depends_on to health check-based startup coordination for more robust service initialization.

**Data Architecture Enhancement**: Implement backup strategies for persistent volumes and consider migrating to managed storage solutions for production environments. Evaluate Redis clustering for high availability and implement data encryption for sensitive model artifacts and processed datasets.

This orchestration demonstrates solid foundational practices with multi-stage builds and logical service separation, but would benefit from production-hardening improvements focused on security, resilience, and operational monitoring capabilities.