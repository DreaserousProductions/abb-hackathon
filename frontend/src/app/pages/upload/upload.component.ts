// import { CommonModule } from '@angular/common';
// import { Component, HostBinding, OnDestroy } from '@angular/core';
// import { FormsModule } from '@angular/forms';
// import { TrainModelResponse, UploadResult, UploadService, ValidateRangesResponse } from '../../services/upload/upload.service';
// import { HttpEvent, HttpEventType } from '@angular/common/http';
// import { firstValueFrom, Subscription } from 'rxjs';
// import { SimulationMessage, SimulationResult, SimulationService } from '../../services/simulation/simulation.service';
// import { AppTheme, ThemeService } from '../../services/theme/theme.service';
// import { ViewChild, ElementRef, AfterViewInit } from '@angular/core';
// import { Chart, ChartConfiguration, ChartType, registerables } from 'chart.js';

// // Register Chart.js components
// Chart.register(...registerables);

// // Interfaces
// interface DateRange { start: string; end: string; }
// interface DateRanges { training: DateRange; testing: DateRange; simulation: DateRange; }
// interface RangeValidation { training: boolean | null; testing: boolean | null; simulation: boolean | null; }
// interface RangeCounts { training: number; testing: number; simulation: number; }
// interface TimelineData { type: string; label: string; count: number; percentage: number; }
// interface TrainingMetrics { accuracy: number; precision: number; recall: number; f1Score: number; }
// interface TrainingResult { metrics: TrainingMetrics; plots: { featureImportance: string; }; }
// interface ConfusionMatrix { truePositive: number; falsePositive: number; trueNegative: number; falseNegative: number; }
// interface SimulationStats { totalPredictions: number; passCount: number; failCount: number; currentAccuracy: number; correctCount: number; }
// interface RealtimePrediction { timestamp: Date; prediction: 'pass' | 'fail'; confidence: number; isCorrect: boolean; }
// interface ConfidenceDistribution { high: number; medium: number; low: number; }
// type SimulationStatus = 'idle' | 'running' | 'stopping' | 'completed';
// // Add this interface for monthly data structure
// interface MonthlyData {
//   month: string;
//   year: number;
//   trainingDays: number;
//   testingDays: number;
//   simulationDays: number;
//   totalDays: number;
// }

// @Component({
//   selector: 'app-upload',
//   standalone: true,
//   imports: [CommonModule, FormsModule],
//   templateUrl: './upload.component.html',
//   styleUrl: './comb-upload.component.css'
// })
// export class UploadComponent implements OnDestroy {
//   // Make ViewChild optional to handle cases where canvas isn't available
//   @ViewChild('monthlyChart', { static: false }) chartCanvas?: ElementRef<HTMLCanvasElement>;
//   private monthlyChart: Chart | null = null;

//   // Chart configuration properties
//   chartColors = {
//     training: '#3b82f6',    // Blue
//     testing: '#f59e0b',     // Amber  
//     simulation: '#10b981'   // Emerald
//   };

//   // Step management
//   currentStep = 0;
//   steps = [
//     { label: 'Upload' },
//     { label: 'Pre-process' },
//     { label: 'Train' },
//     { label: 'Simulate' }
//   ];

//   // File upload state
//   selectedFile: File | null = null;
//   isDragOver = false;
//   isUploading = false;
//   uploadProgress = 0;
//   errorMessage = '';
//   uploadResult: UploadResult | null = null;

//   // Date ranges state
//   dateRanges: DateRanges = { training: { start: '', end: '' }, testing: { start: '', end: '' }, simulation: { start: '', end: '' } };
//   rangeValidation: RangeValidation = { training: null, testing: null, simulation: null };
//   rangeCounts: RangeCounts = { training: 0, testing: 0, simulation: 0 };
//   isValidatingRanges = false;
//   timelineData: TimelineData[] | null = null;

//   // Training state
//   isTraining = false;
//   trainingProgress = 0;
//   trainingResults: TrainModelResponse | null = null;
//   confusionMatrix: ConfusionMatrix = { truePositive: 0, falsePositive: 0, trueNegative: 0, falseNegative: 0 };

//   // Simulation state
//   simulationStatus: SimulationStatus = 'idle';
//   simulationStats: SimulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0, correctCount: 0, };
//   realtimeData: RealtimePrediction[] = [];
//   confidenceDistribution: ConfidenceDistribution = { high: 0, medium: 0, low: 0 };

//   // Intervals
//   private simulationInterval: any;
//   private progressInterval: any;

//   private simSubscription?: Subscription;
//   constructor(private uploadService: UploadService, private simulationService: SimulationService, private themeService: ThemeService) { }

//   @HostBinding('class') themeClass: string = 'theme-new';
//   private themeSubscription!: Subscription;

//   ngOnInit(): void {
//     // Subscribe to theme changes
//     this.themeSubscription = this.themeService.getTheme().subscribe((theme: AppTheme) => {
//       // Update the host class based on the current theme
//       this.themeClass = theme === 'legacy' ? 'theme-legacy' : 'theme-new';
//     });
//   }

//   ngOnDestroy(): void {
//     this.clearIntervals();
//     this.simSubscription?.unsubscribe();
//     this.simulationService.closeConnection();
//     if (this.themeSubscription) {
//       this.themeSubscription.unsubscribe();
//     }

//     if (this.monthlyChart) {
//       this.monthlyChart.destroy();
//     }

//     // Your existing cleanup code...
//     this.clearIntervals();
//     this.simSubscription?.unsubscribe();
//     this.simulationService.closeConnection();
//     if (this.themeSubscription) {
//       this.themeSubscription.unsubscribe();
//     }
//   }

//   private clearIntervals(): void {
//     if (this.simulationInterval) clearInterval(this.simulationInterval);
//     if (this.progressInterval) clearInterval(this.progressInterval);
//   }

//   reset(): void {
//     this.selectedFile = null;
//     this.uploadResult = null;
//     this.errorMessage = '';
//     this.isUploading = false;
//     this.isDragOver = false;
//     this.resetDateRanges();
//     this.resetTraining();
//     this.resetSimulation();
//   }

//   private resetDateRanges(): void {
//     this.dateRanges = { training: { start: '', end: '' }, testing: { start: '', end: '' }, simulation: { start: '', end: '' } };
//     this.rangeValidation = { training: null, testing: null, simulation: null };
//     this.rangeCounts = { training: 0, testing: 0, simulation: 0 };
//     this.timelineData = null;
//   }

//   private resetTraining(): void {
//     this.isTraining = false;
//     this.trainingProgress = 0;
//     this.trainingResults = null;
//     this.confusionMatrix = { truePositive: 0, falsePositive: 0, trueNegative: 0, falseNegative: 0 };
//   }

//   resetSimulation(): void {
//     this.clearIntervals();
//     this.simulationStatus = 'idle';
//     this.simulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0, correctCount: 0 };
//     this.realtimeData = [];
//     this.confidenceDistribution = { high: 0, medium: 0, low: 0 };
//   }

//   // Upload, preprocess, training, simulation logic continues here...
//   onFileSelected(event: Event): void {
//     const input = event.target as HTMLInputElement;
//     if (input.files?.length) {
//       const file = input.files[0];
//       this.processFile(file);
//     }
//   }

//   /**
//    * Handles drag over events for drag-and-drop functionality
//    */
//   onDragOver(event: DragEvent): void {
//     event.preventDefault();
//     event.stopPropagation();
//     this.isDragOver = true;
//   }

//   /**
//    * Handles drag leave events
//    */
//   onDragLeave(event: DragEvent): void {
//     event.preventDefault();
//     event.stopPropagation();
//     this.isDragOver = false;
//   }

//   /**
//    * Handles file drop events for drag-and-drop functionality
//    */
//   onDrop(event: DragEvent): void {
//     event.preventDefault();
//     event.stopPropagation();
//     this.isDragOver = false;

//     const files = event.dataTransfer?.files;
//     if (files?.length) {
//       const file = files[0];
//       this.processFile(file);
//     }
//   }

//   /**
//    * Validates and processes the selected file
//    * Implements client-side validation as specified in the PDF
//    */
//   private processFile(file: File): void {
//     this.errorMessage = '';

//     // Validate file type (only .csv files allowed)
//     if (!file.name.toLowerCase().endsWith('.csv')) {
//       this.errorMessage = 'Please select a valid CSV file (.csv extension required)';
//       return;
//     }

//     // Validate file size (reasonable limit for demo purposes)
//     const maxSizeInMB = 2500;
//     const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
//     if (file.size > maxSizeInBytes) {
//       this.errorMessage = `File size must be less than ${maxSizeInMB}MB`;
//       return;
//     }

//     // File is valid, proceed with upload
//     this.selectedFile = file;
//     this.uploadFile(file);
//   }

//   /**
//    * Simulates file upload and processing
//    * In a real implementation, this would call the /api/data/upload endpoint
//    */
//   private uploadFile(file: File): void {
//     this.isUploading = true;
//     this.errorMessage = '';
//     this.uploadProgress = 0; // Start progress at 0

//     const userId = localStorage.getItem('username') || 'anonymous_user';

//     this.uploadService.uploadFileWithProgress(file, userId).subscribe({
//       // FIX: The 'next' handler now receives the final UploadResult directly
//       next: (result: UploadResult) => {
//         // No need to check for event type. This code runs when the upload is complete.
//         this.uploadResult = result;
//         console.log('Upload complete!', this.uploadResult);

//         // Since we know it's complete, we can set progress to 100%
//         this.uploadProgress = 100;
//       },
//       error: (err) => {
//         this.errorMessage = 'Failed to upload file. Please try again.';
//         console.error('Upload error:', err);
//         this.isUploading = false;
//         this.uploadProgress = 0; // Reset progress on error
//       },
//       complete: () => {
//         this.isUploading = false;
//         // Optional: Reset progress to 0 a moment after completion
//         setTimeout(() => this.uploadProgress = 0, 2000);
//       }
//     });
//   }

//   /**
//    * Removes the selected file and resets the upload state
//    */
//   removeFile(event: Event): void {
//     event.stopPropagation();
//     this.selectedFile = null;
//     this.uploadResult = null;
//     this.errorMessage = '';
//   }

//   /**
//    * Resets the entire upload component to initial state
//    */
//   // reset(): void {
//   //   this.selectedFile = null;
//   //   this.uploadResult = null;
//   //   this.errorMessage = '';
//   //   this.isUploading = false;
//   //   this.isDragOver = false;
//   // }

//   /**
//    * Advances to the next step
//    * Only enabled when upload is complete as per PDF requirements
//    */
//   /**
//     * Advances to the next step and sets initial date values if applicable.
//     */
//   next(): void {
//     if (this.currentStep < this.steps.length - 1) {
//       this.currentStep++;

//       // When moving to the Pre-processing step (index 1), set default dates
//       if (this.currentStep === 1 && this.uploadResult) {
//         this.dateRanges.training.start = this.formatDateForInput(this.uploadResult.dateRange.start);
//         this.dateRanges.simulation.end = this.formatDateForInput(this.uploadResult.dateRange.end);
//       }
//     }
//   }

//   /**
//    * Navigates to a specific step
//    * Only allows backward navigation or navigation to completed steps
//    */
//   goToStep(index: number): void {
//     if (index <= this.currentStep) {
//       this.currentStep = index;
//     }
//   }

//   /**
//    * Formats file size for display
//    */
//   formatFileSize(bytes: number): string {
//     if (bytes === 0) return '0 Bytes';

//     const k = 1024;
//     const sizes = ['Bytes', 'KB', 'MB', 'GB'];
//     const i = Math.floor(Math.log(bytes) / Math.log(k));

//     return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
//   }

//   /**
//    * Formats date range for display in summary card
//    */
//   formatDateRange(dateRange: DateRange): string {
//     const startDate = new Date(dateRange.start);
//     const endDate = new Date(dateRange.end);

//     const formatOptions: Intl.DateTimeFormatOptions = {
//       year: 'numeric',
//       month: 'short',
//       day: 'numeric'
//     };

//     return `${startDate.toLocaleDateString('en-US', formatOptions)} - ${endDate.toLocaleDateString('en-US', formatOptions)}`;
//   }

//   /**
//    * Utility function to create delays for simulation
//    */
//   private delay(ms: number): Promise<void> {
//     return new Promise(resolve => setTimeout(resolve, ms));
//   }

//   /**
//     * Formats a date string or Date object to be compatible with datetime-local input.
//     * Strips microseconds and ensures proper 'YYYY-MM-DDTHH:MM:SS' format.
//     */
//   private formatDateForInput(dateValue: string | Date): string {
//     if (!dateValue) {
//       return '';
//     }

//     // Create a new Date object whether the input is a string or a Date
//     const date = new Date(dateValue);

//     // Check if the date is valid before proceeding
//     if (isNaN(date.getTime())) {
//       return '';
//     }

//     const year = date.getFullYear();
//     const month = ('0' + (date.getMonth() + 1)).slice(-2);
//     const day = ('0' + date.getDate()).slice(-2);
//     const hours = ('0' + date.getHours()).slice(-2);
//     const minutes = ('0' + date.getMinutes()).slice(-2);
//     const seconds = ('0' + date.getSeconds()).slice(-2);

//     return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
//   }

//   // ADD THIS NEW METHOD
//   /**
//    * Automatically splits the date ranges into a 70/20/10 ratio.
//    */
//   autoSplitRanges(): void {
//     if (!this.uploadResult) {
//       return;
//     }

//     const startDate = new Date(this.uploadResult.dateRange.start);
//     const endDate = new Date(this.uploadResult.dateRange.end);

//     const totalDuration = endDate.getTime() - startDate.getTime();

//     // Calculate the end time for the training period (70% of the duration)
//     const trainingEndTime = startDate.getTime() + totalDuration * 0.7;
//     const trainingEndDate = new Date(trainingEndTime);

//     // Calculate the end time for the testing period (another 20% of the duration)
//     const testingEndTime = trainingEndTime + totalDuration * 0.2;
//     const testingEndDate = new Date(testingEndTime);

//     // Set the date ranges
//     this.dateRanges = {
//       training: {
//         start: this.formatDateForInput(startDate),
//         end: this.formatDateForInput(trainingEndDate)
//       },
//       testing: {
//         start: this.formatDateForInput(trainingEndDate),
//         end: this.formatDateForInput(testingEndDate)
//       },
//       simulation: {
//         start: this.formatDateForInput(testingEndDate),
//         end: this.formatDateForInput(endDate) // Ends at the dataset's overall end
//       }
//     };

//     // Trigger validation to update the UI state immediately
//     this.validateDateRanges();
//   }

//   // === DATE RANGES FUNCTIONALITY (STEP 2) ===

//   /**
//  * Get minimum date for date inputs (from upload result)
//  */
//   getMinDate(): string {
//     return this.uploadResult ? this.formatDateForInput(this.uploadResult.dateRange.start) : '';
//   }

//   /**
//    * Get maximum date for date inputs (from upload result)
//    */
//   getMaxDate(): string {
//     return this.uploadResult ? this.formatDateForInput(this.uploadResult.dateRange.end) : '';
//   }

//   /**
//    * Check if date ranges can be validated
//    */
//   canValidateRanges(): boolean {
//     return !!(
//       this.dateRanges.training.start && this.dateRanges.training.end &&
//       this.dateRanges.testing.start && this.dateRanges.testing.end &&
//       this.dateRanges.simulation.start && this.dateRanges.simulation.end
//     );
//   }

//   /**
//    * Validate individual date ranges (client-side validation)
//    */
//   validateDateRanges(): void {
//     // Reset validation state
//     this.rangeValidation = {
//       training: null,
//       testing: null,
//       simulation: null
//     };

//     // Validate training range
//     if (this.dateRanges.training.start && this.dateRanges.training.end) {
//       this.rangeValidation.training =
//         new Date(this.dateRanges.training.start) < new Date(this.dateRanges.training.end);
//     }

//     // Validate testing range
//     if (this.dateRanges.testing.start && this.dateRanges.testing.end) {
//       this.rangeValidation.testing =
//         new Date(this.dateRanges.testing.start) < new Date(this.dateRanges.testing.end) &&
//         new Date(this.dateRanges.testing.start) >= new Date(this.dateRanges.training.end);
//     }

//     // Validate simulation range
//     if (this.dateRanges.simulation.start && this.dateRanges.simulation.end) {
//       this.rangeValidation.simulation =
//         new Date(this.dateRanges.simulation.start) < new Date(this.dateRanges.simulation.end) &&
//         new Date(this.dateRanges.simulation.start) >= new Date(this.dateRanges.testing.end);
//     }
//   }

//   validateRanges(): void {
//     if (!this.uploadResult) {
//       this.errorMessage = "Cannot validate ranges without an uploaded file.";
//       return;
//     }

//     this.isValidatingRanges = true;
//     this.errorMessage = ''; // Clear previous errors

//     this.uploadService.validateDateRanges(
//       this.uploadResult.datasetId,
//       this.uploadResult.userId,
//       this.dateRanges
//     ).subscribe({
//       next: (response: ValidateRangesResponse) => {
//         if (response.status === 'Valid') {
//           // Update range counts from the backend response
//           this.rangeCounts = {
//             training: response.training.count,
//             testing: response.testing.count,
//             simulation: response.simulation.count,
//           };

//           // Generate timeline data from the backend response
//           const total = this.rangeCounts.training + this.rangeCounts.testing + this.rangeCounts.simulation;
//           if (total > 0) {
//             this.timelineData = [
//               {
//                 type: 'training',
//                 label: 'Training',
//                 count: this.rangeCounts.training,
//                 percentage: (this.rangeCounts.training / total) * 100
//               },
//               {
//                 type: 'testing',
//                 label: 'Testing',
//                 count: this.rangeCounts.testing,
//                 percentage: (this.rangeCounts.testing / total) * 100
//               },
//               {
//                 type: 'simulation',
//                 label: 'Simulation',
//                 count: this.rangeCounts.simulation,
//                 percentage: (this.rangeCounts.simulation / total) * 100
//               }
//             ];
//           } else {
//             this.timelineData = null; // No records found
//           }

//           // You can now also use response.monthlyCounts to display another chart if you wish
//           console.log('Monthly Counts:', response.monthlyCounts);

//         } else {
//           // Handle invalid status from backend if needed, though FastAPI handles this with 400 error
//           this.errorMessage = "Validation failed on the server.";
//         }
//         this.isValidatingRanges = false;
//       },
//       error: (err) => {
//         // Display the detailed error message from the backend
//         this.errorMessage = err.error?.detail || err.error || 'An unknown validation error occurred.';
//         console.error('Range validation error:', err);
//         this.timelineData = null; // Clear timeline on error
//         this.isValidatingRanges = false;
//       }
//     });
//   }

//   /**
//    * Check if all ranges are valid
//    */
//   areRangesValid(): boolean {
//     return !!(
//       this.rangeValidation.training === true &&
//       this.rangeValidation.testing === true &&
//       this.rangeValidation.simulation === true &&
//       this.timelineData
//     );
//   }

//   /**
//    * Format date range for display
//    */
//   // formatDateRange(range: DateRange): string {
//   //   if (!range.start || !range.end) return 'Not set';

//   //   const startDate = new Date(range.start);
//   //   const endDate = new Date(range.end);

//   //   const formatOptions: Intl.DateTimeFormatOptions = {
//   //     year: 'numeric',
//   //     month: 'short',
//   //     day: 'numeric'
//   //   };

//   //   return `${startDate.toLocaleDateString('en-US', formatOptions)} - ${endDate.toLocaleDateString('en-US', formatOptions)}`;
//   // }

//   // === TRAINING FUNCTIONALITY (STEP 3) ===

//   /**
//    * Start model training process
//    */
//   // async trainModel(): Promise<void> {
//   //   this.isTraining = true;
//   //   this.trainingProgress = 0;

//   //   try {
//   //     // Simulate training progress
//   //     this.progressInterval = setInterval(() => {
//   //       if (this.trainingProgress < 100) {
//   //         this.trainingProgress += Math.random() * 15;
//   //         if (this.trainingProgress > 100) {
//   //           this.trainingProgress = 100;
//   //         }
//   //       }
//   //     }, 200);

//   //     // Simulate training time
//   //     await this.delay(4000);

//   //     // Generate mock training results
//   //     this.trainingResults = {
//   //       metrics: {
//   //         accuracy: 0.92 + (Math.random() * 0.06), // 92-98%
//   //         precision: 0.89 + (Math.random() * 0.08), // 89-97%
//   //         recall: 0.87 + (Math.random() * 0.09), // 87-96%
//   //         f1Score: 0.88 + (Math.random() * 0.08) // 88-96%
//   //       },
//   //       plots: {
//   //         featureImportance: this.generateMockSHAPPlot()
//   //       }
//   //     };

//   //     // Generate confusion matrix
//   //     this.generateConfusionMatrix();

//   //   } catch (error) {
//   //     console.error('Training error:', error);
//   //   } finally {
//   //     this.isTraining = false;
//   //     this.trainingProgress = 100;
//   //     if (this.progressInterval) {
//   //       clearInterval(this.progressInterval);
//   //     }
//   //   }
//   // }

//   async trainModel(): Promise<void> {
//     if (!this.uploadResult) {
//       return;
//     }

//     if (!this.uploadResult.datasetId || !this.uploadResult.userId || !this.uploadResult.dateRange) {
//       console.error('Cannot train model: Missing datasetId, userId, or date ranges.');
//       // You should show an error to the user here (e.g., using a toast/snackbar)
//       return;
//     }

//     this.isTraining = true;
//     this.trainingProgress = 0;
//     this.trainingResults = null; // Clear previous results

//     // A real API call doesn't provide granular progress easily.
//     // We'll show an initial progress and then jump to 100 on completion.
//     // For a better UX, consider an indeterminate progress bar in your HTML template.
//     this.trainingProgress = 10;

//     try {
//       const response = await firstValueFrom(
//         this.uploadService.trainModel(this.uploadResult.datasetId, this.uploadResult.userId, this.dateRanges)
//       );
//       console.log(response);
//       this.trainingResults = response;
//       this.trainingProgress = 100;

//       // If you have other functions that depend on the results, call them here
//       this.generateConfusionMatrix(); // Assuming this function exists and uses this.trainingResults

//     } catch (error) {
//       console.error('Training API call failed:', error);
//       // Display a user-friendly error message
//       // e.g., this.toastService.showError('Model training failed. Please try again.');
//       this.trainingProgress = 0; // Reset progress on error
//     } finally {
//       this.isTraining = false;
//     }
//   }

//   /**
//    * Generate mock confusion matrix data
//    */
//   private generateConfusionMatrix(): void {
//     // Check if we have real results from the backend
//     if (!this.trainingResults) {
//       console.error("Cannot generate confusion matrix: training results are null.");
//       return;
//     }

//     // Directly assign the real values from the API response
//     const metrics = this.trainingResults.metrics;
//     console.log(metrics);
//     this.confusionMatrix = {
//       truePositive: metrics.truePositive,
//       falsePositive: metrics.falsePositive,
//       trueNegative: metrics.trueNegative,
//       falseNegative: metrics.falseNegative
//     };

//     console.log("Generated real confusion matrix:", this.confusionMatrix);
//   }

//   // === SIMULATION FUNCTIONALITY (STEP 4) ===

//   /**
//    * Toggle simulation state
//    */
//   toggleSimulation(): void {
//     if (this.simulationStatus === 'idle' || this.simulationStatus === 'completed') {
//       this.startSimulation();
//     } else if (this.simulationStatus === 'running') {
//       this.stopSimulation();
//     }
//   }

//   /**
//    * Starts the simulation by connecting to the WebSocket and subscribing to messages.
//    */
//   private startSimulation(): void {
//     if (this.uploadResult && (!this.uploadResult.userId || !this.uploadResult.datasetId)) {
//       console.error("Cannot start simulation: userId or datasetId is missing.");
//       return;
//     }
//     this.resetSimulationData();
//     this.simulationStatus = 'running';

//     this.simulationService.connect();
//     this.simSubscription = this.simulationService.messages$.subscribe({
//       next: (message: SimulationMessage) => this.processSimulationMessage(message),
//       error: (err) => {
//         console.error('WebSocket Error:', err);
//         this.simulationStatus = 'idle'; // Reset on error
//       },
//       complete: () => {
//         console.log('WebSocket stream completed.');
//         this.simulationStatus = 'completed';
//       }
//     });

//     // Send the start command after connecting
//     this.simulationService.startSimulation(this.uploadResult!.userId, this.uploadResult!.datasetId);
//   }

//   /**
//    * Sends a stop command to the WebSocket.
//    */
//   private stopSimulation(): void {
//     if (this.uploadResult && (!this.uploadResult.userId || !this.uploadResult.datasetId)) return;

//     this.simulationStatus = 'stopping';
//     this.simulationService.stopSimulation(this.uploadResult!.userId, this.uploadResult!.datasetId);
//   }

//   /**
//    * Processes each message received from the WebSocket server.
//    */
//   private processSimulationMessage(message: SimulationMessage): void {
//     if ('rowIndex' in message) {
//       // It's a data message
//       const result = message as SimulationResult;
//       const newPrediction: RealtimePrediction = {
//         timestamp: new Date(result.timestamp),
//         prediction: result.prediction === 1 ? 'pass' : 'fail',
//         confidence: result.confidence,
//         isCorrect: result.isCorrect
//       };
//       this.realtimeData.push(newPrediction);
//       this.updateSimulationStats(newPrediction);
//       this.updateConfidenceDistribution();
//     } else if (message.status) {
//       // It's a status message
//       if (message.status === 'stopped' || message.status === 'finished') {
//         this.simulationStatus = 'completed';
//         this.simSubscription?.unsubscribe();
//       }
//     } else if (message.error) {
//       console.error("Error from simulation server:", message.error);
//       this.stopSimulation();
//     }
//   }

//   /**
//    * Resets all data and stats for a new simulation run.
//    */
//   private resetSimulationData(): void {
//     this.realtimeData = [];
//     this.simulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, correctCount: 0, currentAccuracy: 0 };
//     this.updateConfidenceDistribution();
//   }

//   /**
//    * Updates the running statistics with each new prediction.
//    */
//   private updateSimulationStats(prediction: RealtimePrediction): void {
//     this.simulationStats.totalPredictions++;
//     prediction.prediction === 'pass' ? this.simulationStats.passCount++ : this.simulationStats.failCount++;
//     if (prediction.isCorrect) {
//       this.simulationStats.correctCount++;
//     }
//     this.simulationStats.currentAccuracy = (this.simulationStats.correctCount / this.simulationStats.totalPredictions) * 100;
//   }

//   /**
//    * Recalculates the confidence distribution pie chart data.
//    */
//   private updateConfidenceDistribution(): void {
//     if (this.realtimeData.length === 0) {
//       this.confidenceDistribution = { high: 0, medium: 0, low: 0 };
//       return;
//     }
//     let high = 0, medium = 0, low = 0;
//     this.realtimeData.forEach(p => {
//       if (p.confidence > 0.9) high++;
//       else if (p.confidence > 0.8) medium++;
//       else low++;
//     });
//     const total = this.realtimeData.length;
//     this.confidenceDistribution = {
//       high: (high / total) * 100,
//       medium: (medium / total) * 100,
//       low: (low / total) * 100
//     };
//   }

//   /**
//   * Get simulation status text
//   */
//   getSimulationStatusText(): string {
//     switch (this.simulationStatus) {
//       case 'idle': return 'Ready to start';
//       case 'running': return 'Streaming predictions';
//       case 'stopping': return 'Stopping...';
//       case 'completed': return 'Simulation complete';
//       default: return 'Unknown';
//     }
//   }

//   /**
//    * Get simulation button text
//    */
//   getSimulationButtonText(): string {
//     switch (this.simulationStatus) {
//       case 'idle': return 'Start Simulation';
//       case 'running': return 'Stop Simulation';
//       case 'stopping': return 'Stopping...';
//       case 'completed': return 'Restart Simulation';
//       default: return 'Start Simulation';
//     }
//   }
//   /**
//    * Calculates and returns a human-readable duration between two dates
//    * @param start - Start date (Date object or ISO string)
//    * @param end - End date (Date object or ISO string)
//    * @returns Formatted duration string (e.g., "2 hours", "5 days", "3 months")
//    */
//   getDuration(start: Date | string, end: Date | string): string {
//     if (!start || !end) {
//       return 'Invalid dates';
//     }

//     // Convert strings to Date objects if needed
//     const startDate = typeof start === 'string' ? new Date(start) : start;
//     const endDate = typeof end === 'string' ? new Date(end) : end;

//     // Check for invalid dates
//     if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
//       return 'Invalid dates';
//     }

//     if (startDate >= endDate) {
//       return 'Invalid range';
//     }

//     const diffMs = endDate.getTime() - startDate.getTime();

//     // Convert to different time units
//     const minutes = Math.floor(diffMs / (1000 * 60));
//     const hours = Math.floor(diffMs / (1000 * 60 * 60));
//     const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
//     const weeks = Math.floor(days / 7);
//     const months = Math.floor(days / 30.44); // Average days per month
//     const years = Math.floor(days / 365.25); // Account for leap years

//     // Return the most appropriate unit
//     if (years >= 1) {
//       return years === 1 ? '1 year' : `${years} years`;
//     } else if (months >= 1) {
//       return months === 1 ? '1 month' : `${months} months`;
//     } else if (weeks >= 1) {
//       return weeks === 1 ? '1 week' : `${weeks} weeks`;
//     } else if (days >= 1) {
//       return days === 1 ? '1 day' : `${days} days`;
//     } else if (hours >= 1) {
//       return hours === 1 ? '1 hour' : `${hours} hours`;
//     } else if (minutes >= 1) {
//       return minutes === 1 ? '1 minute' : `${minutes} minutes`;
//     } else {
//       return 'Less than 1 minute';
//     }
//   }

//   /**
//    * Formats a date range for display
//    * @param start - Start date (Date object or ISO string)
//    * @param end - End date (Date object or ISO string)
//    * @returns Formatted date range string (e.g., "Jan 15 - Feb 28, 2024")
//    */
//   formatDateRangeChart(start: Date | string, end: Date | string): string {
//     if (!start || !end) {
//       return 'Invalid dates';
//     }

//     // Convert strings to Date objects if needed
//     const startDate = typeof start === 'string' ? new Date(start) : start;
//     const endDate = typeof end === 'string' ? new Date(end) : end;

//     // Check for invalid dates
//     if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
//       return 'Invalid dates';
//     }

//     if (startDate >= endDate) {
//       return 'Invalid range';
//     }

//     const options: Intl.DateTimeFormatOptions = {
//       month: 'short',
//       day: 'numeric'
//     };

//     const optionsWithYear: Intl.DateTimeFormatOptions = {
//       month: 'short',
//       day: 'numeric',
//       year: 'numeric'
//     };

//     // Check if both dates are in the same year
//     if (startDate.getFullYear() === endDate.getFullYear()) {
//       // Same year - show year only at the end
//       const startFormatted = startDate.toLocaleDateString('en-US', options);
//       const endFormatted = endDate.toLocaleDateString('en-US', optionsWithYear);
//       return `${startFormatted} - ${endFormatted}`;
//     } else {
//       // Different years - show year for both dates
//       const startFormatted = startDate.toLocaleDateString('en-US', optionsWithYear);
//       const endFormatted = endDate.toLocaleDateString('en-US', optionsWithYear);
//       return `${startFormatted} - ${endFormatted}`;
//     }
//   }

//   @ViewChild('monthlyChart', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;
//   private monthlyChart: Chart | null = null;

//   // Chart configuration properties
//   chartColors = {
//     training: '#3b82f6',    // Blue
//     testing: '#f59e0b',     // Amber  
//     simulation: '#10b981'   // Emerald
//   };

//   /**
//    * Processes dateRanges to calculate monthly distribution of training, testing, and simulation days
//    */
//   private processDateRangesForChart(): MonthlyData[] {
//     if (!this.dateRanges.training.start || !this.dateRanges.simulation.end) {
//       return [];
//     }

//     // Get overall date range
//     const overallStart = new Date(this.dateRanges.training.start);
//     const overallEnd = new Date(this.dateRanges.simulation.end);

//     // Create array to store monthly data
//     const monthlyDataMap = new Map<string, MonthlyData>();

//     // Initialize months between start and end dates
//     const currentDate = new Date(overallStart.getFullYear(), overallStart.getMonth(), 1);
//     const endDate = new Date(overallEnd.getFullYear(), overallEnd.getMonth() + 1, 0);

//     while (currentDate <= endDate) {
//       const monthKey = `${currentDate.getFullYear()}-${currentDate.getMonth()}`;
//       const monthName = currentDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });

//       monthlyDataMap.set(monthKey, {
//         month: monthName,
//         year: currentDate.getFullYear(),
//         trainingDays: 0,
//         testingDays: 0,
//         simulationDays: 0,
//         totalDays: 0
//       });

//       currentDate.setMonth(currentDate.getMonth() + 1);
//     }

//     // Calculate training days per month
//     this.calculateDaysPerMonth(
//       new Date(this.dateRanges.training.start),
//       new Date(this.dateRanges.training.end),
//       monthlyDataMap,
//       'training'
//     );

//     // Calculate testing days per month
//     this.calculateDaysPerMonth(
//       new Date(this.dateRanges.testing.start),
//       new Date(this.dateRanges.testing.end),
//       monthlyDataMap,
//       'testing'
//     );

//     // Calculate simulation days per month
//     this.calculateDaysPerMonth(
//       new Date(this.dateRanges.simulation.start),
//       new Date(this.dateRanges.simulation.end),
//       monthlyDataMap,
//       'simulation'
//     );

//     // Calculate total days and convert records to proportional values
//     const monthlyData = Array.from(monthlyDataMap.values()).map(data => {
//       data.totalDays = data.trainingDays + data.testingDays + data.simulationDays;

//       // Convert days to estimated record counts (placeholder calculation)
//       // You can modify this logic based on your actual record distribution
//       const avgRecordsPerDay = 1000; // Placeholder value
//       data.trainingDays *= avgRecordsPerDay;
//       data.testingDays *= avgRecordsPerDay;
//       data.simulationDays *= avgRecordsPerDay;

//       return data;
//     });

//     return monthlyData.filter(data => data.totalDays > 0);
//   }

//   /**
//    * Helper method to calculate how many days of a period fall into each month
//    */
//   /**
//  * Helper method to calculate how many days of a period fall into each month
//  */
//   private calculateDaysPerMonth(
//     startDate: Date,
//     endDate: Date,
//     monthlyDataMap: Map<string, MonthlyData>,
//     periodType: 'training' | 'testing' | 'simulation'
//   ): void {
//     const currentDate = new Date(startDate);

//     while (currentDate <= endDate) {
//       const monthKey = `${currentDate.getFullYear()}-${currentDate.getMonth()}`;
//       const monthData = monthlyDataMap.get(monthKey);

//       if (monthData) {
//         // Use switch statement for type safety
//         switch (periodType) {
//           case 'training':
//             monthData.trainingDays += 1;
//             break;
//           case 'testing':
//             monthData.testingDays += 1;
//             break;
//           case 'simulation':
//             monthData.simulationDays += 1;
//             break;
//         }
//       }

//       currentDate.setDate(currentDate.getDate() + 1);
//     }
//   }

//   /**
//    * Creates and configures the Chart.js stacked bar chart
//    */
//   private createMonthlyChart(): void {
//     if (!this.chartCanvas?.nativeElement) {
//       console.error('Chart canvas not available');
//       return;
//     }

//     // Process the dateRanges data
//     const monthlyData = this.processDateRangesForChart();

//     if (monthlyData.length === 0) {
//       console.warn('No monthly data available for chart');
//       return;
//     }

//     // Prepare chart data
//     const labels = monthlyData.map(data => data.month);
//     const trainingData = monthlyData.map(data => data.trainingDays);
//     const testingData = monthlyData.map(data => data.testingDays);
//     const simulationData = monthlyData.map(data => data.simulationDays);

//     // Chart configuration
//     const config: ChartConfiguration = {
//       type: 'bar' as ChartType,
//       data: {
//         labels: labels,
//         datasets: [
//           {
//             label: 'Training',
//             data: trainingData,
//             backgroundColor: this.chartColors.training,
//             borderColor: this.chartColors.training,
//             borderWidth: 1,
//             stack: 'records'
//           },
//           {
//             label: 'Testing',
//             data: testingData,
//             backgroundColor: this.chartColors.testing,
//             borderColor: this.chartColors.testing,
//             borderWidth: 1,
//             stack: 'records'
//           },
//           {
//             label: 'Simulation',
//             data: simulationData,
//             backgroundColor: this.chartColors.simulation,
//             borderColor: this.chartColors.simulation,
//             borderWidth: 1,
//             stack: 'records'
//           }
//         ]
//       },
//       options: {
//         responsive: true,
//         maintainAspectRatio: false,
//         interaction: {
//           intersect: false,
//           mode: 'index'
//         },
//         plugins: {
//           title: {
//             display: true,
//             text: 'Monthly Data Distribution',
//             font: {
//               size: 16,
//               weight: 'bold'
//             }
//           },
//           legend: {
//             position: 'top',
//             labels: {
//               usePointStyle: true,
//               padding: 20
//             }
//           },
//           tooltip: {
//             mode: 'index',
//             intersect: false,
//             callbacks: {
//               label: (context) => {
//                 const label = context.dataset.label || '';
//                 const value = context.parsed.y || 0;
//                 return `${label}: ${value.toLocaleString()} records`;
//               },
//               footer: (tooltipItems) => {
//                 const total = tooltipItems.reduce((sum, item) => sum + (item.parsed.y || 0), 0);
//                 return `Total: ${total.toLocaleString()} records`;
//               }
//             }
//           }
//         },
//         scales: {
//           x: {
//             title: {
//               display: true,
//               text: 'Time Period (Months)',
//               font: {
//                 size: 14,
//                 weight: 'bold'
//               }
//             },
//             grid: {
//               display: false
//             }
//           },
//           y: {
//             title: {
//               display: true,
//               text: 'Number of Records',
//               font: {
//                 size: 14,
//                 weight: 'bold'
//               }
//             },
//             beginAtZero: true,
//             stacked: true,
//             ticks: {
//               callback: function (value) {
//                 return typeof value === 'number' ? value.toLocaleString() : value;
//               }
//             }
//           }
//         }
//       }
//     };

//     // Destroy existing chart if it exists
//     if (this.monthlyChart) {
//       this.monthlyChart.destroy();
//     }

//     // Create new chart
//     const ctx = this.chartCanvas.nativeElement.getContext('2d');
//     if (ctx) {
//       this.monthlyChart = new Chart(ctx, config);
//     }
//   }

//   /**
//    * Public method to update the chart when dateRanges change
//    */
//   updateMonthlyChart(): void {
//     this.createMonthlyChart();
//   }

//   /**
//    * Add this to your ngAfterViewInit lifecycle hook (or modify existing one)
//    */
//   ngAfterViewInit(): void {
//     // Add this line to your existing ngAfterViewInit or create the method if it doesn't exist
//     setTimeout(() => {
//       if (this.dateRanges.training.start) {
//         this.createMonthlyChart();
//       }
//     }, 100);
//   }

//   /**
//    * Call this method after validateRanges() succeeds to update the chart
//    */
//   private updateChartAfterValidation(): void {
//     // Add this call at the end of your validateRanges() success handler
//     setTimeout(() => {
//       this.updateMonthlyChart();
//     }, 100);
//   }
// }

import { CommonModule } from '@angular/common';
import { Component, HostBinding, OnDestroy, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TrainModelResponse, UploadResult, UploadService, ValidateRangesResponse } from '../../services/upload/upload.service';
import { HttpEvent, HttpEventType } from '@angular/common/http';
import { firstValueFrom, Subscription } from 'rxjs';
import { SimulationMessage, SimulationResult, SimulationService } from '../../services/simulation/simulation.service';
import { AppTheme, ThemeService } from '../../services/theme/theme.service';
import { Chart, ChartConfiguration, ChartType, registerables } from 'chart.js';

// Register Chart.js components
Chart.register(...registerables);

// Interfaces
interface DateRange { start: string; end: string; }
interface DateRanges { training: DateRange; testing: DateRange; simulation: DateRange; }
interface RangeValidation { training: boolean | null; testing: boolean | null; simulation: boolean | null; }
interface RangeCounts { training: number; testing: number; simulation: number; }
interface TimelineData { type: string; label: string; count: number; percentage: number; }
interface TrainingMetrics { accuracy: number; precision: number; recall: number; f1Score: number; }
interface TrainingResult { metrics: TrainingMetrics; plots: { featureImportance: string; }; }
interface ConfusionMatrix { truePositive: number; falsePositive: number; trueNegative: number; falseNegative: number; }
interface SimulationStats { totalPredictions: number; passCount: number; failCount: number; currentAccuracy: number; correctCount: number; }
interface RealtimePrediction { rowIndex: number; timestamp: Date; prediction: 'pass' | 'fail'; confidence: number; isCorrect: boolean; }
interface ConfidenceDistribution { high: number; medium: number; low: number; }
type SimulationStatus = 'idle' | 'running' | 'stopping' | 'completed';

// Add this interface for monthly data structure
interface MonthlyData {
  month: string;
  year: number;
  trainingDays: number;
  testingDays: number;
  simulationDays: number;
  totalDays: number;
}

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './upload.component.html',
  styleUrl: './comb-upload.component.css'
})
export class UploadComponent implements OnDestroy, AfterViewInit {
  // Step management
  currentStep = 0;
  steps = [
    { label: 'Upload' },
    { label: 'Pre-process' },
    { label: 'Train' },
    { label: 'Simulate' }
  ];

  // File upload state
  selectedFile: File | null = null;
  isDragOver = false;
  isUploading = false;
  uploadProgress = 0;
  errorMessage = '';
  uploadResult: UploadResult | null = null;

  // Date ranges state
  dateRanges: DateRanges = { training: { start: '', end: '' }, testing: { start: '', end: '' }, simulation: { start: '', end: '' } };
  rangeValidation: RangeValidation = { training: null, testing: null, simulation: null };
  rangeCounts: RangeCounts = { training: 0, testing: 0, simulation: 0 };
  isValidatingRanges = false;
  timelineData: TimelineData[] | null = null;

  // Training state
  isTraining = false;
  trainingProgress = 0;
  trainingResults: TrainModelResponse | null = null;
  confusionMatrix: ConfusionMatrix = { truePositive: 0, falsePositive: 0, trueNegative: 0, falseNegative: 0 };

  // Simulation state
  simulationStatus: SimulationStatus = 'idle';
  simulationStats: SimulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0, correctCount: 0, };
  realtimeData: RealtimePrediction[] = [];
  confidenceDistribution: ConfidenceDistribution = { high: 0, medium: 0, low: 0 };

  // Intervals
  private simulationInterval: any;
  private progressInterval: any;
  private simSubscription?: Subscription;

  // Chart.js properties
  @ViewChild('monthlyChart', { static: false }) chartCanvas?: ElementRef<HTMLCanvasElement>;
  private monthlyChart: Chart | null = null;

  // Chart configuration properties
  chartColors = {
    training: '#3b82f6',    // Blue
    testing: '#f59e0b',     // Amber  
    simulation: '#10b981'   // Emerald
  };

  constructor(private uploadService: UploadService, private simulationService: SimulationService, private themeService: ThemeService) { }

  @HostBinding('class') themeClass: string = 'theme-new';
  private themeSubscription!: Subscription;

  ngOnInit(): void {
    // Subscribe to theme changes
    this.themeSubscription = this.themeService.getTheme().subscribe((theme: AppTheme) => {
      // Update the host class based on the current theme
      this.themeClass = theme === 'legacy' ? 'theme-legacy' : 'theme-new';
    });
  }

  ngAfterViewInit(): void {
    // Don't create chart immediately - wait for data
    console.log('ngAfterViewInit called, canvas available:', !!this.chartCanvas?.nativeElement);
  }

  ngOnDestroy(): void {
    // Clean up chart
    if (this.monthlyChart) {
      this.monthlyChart.destroy();
    }

    // Your existing cleanup code...
    this.clearIntervals();
    this.simSubscription?.unsubscribe();
    this.simulationService.closeConnection();
    if (this.themeSubscription) {
      this.themeSubscription.unsubscribe();
    }
  }

  private clearIntervals(): void {
    if (this.simulationInterval) clearInterval(this.simulationInterval);
    if (this.progressInterval) clearInterval(this.progressInterval);
  }

  reset(): void {
    this.selectedFile = null;
    this.uploadResult = null;
    this.errorMessage = '';
    this.isUploading = false;
    this.isDragOver = false;
    this.resetDateRanges();
    this.resetTraining();
    this.resetSimulation();
  }

  private resetDateRanges(): void {
    this.dateRanges = { training: { start: '', end: '' }, testing: { start: '', end: '' }, simulation: { start: '', end: '' } };
    this.rangeValidation = { training: null, testing: null, simulation: null };
    this.rangeCounts = { training: 0, testing: 0, simulation: 0 };
    this.timelineData = null;
  }

  private resetTraining(): void {
    this.isTraining = false;
    this.trainingProgress = 0;
    this.trainingResults = null;
    this.confusionMatrix = { truePositive: 0, falsePositive: 0, trueNegative: 0, falseNegative: 0 };
  }

  resetSimulation(): void {
    this.clearIntervals();
    this.simulationStatus = 'idle';
    this.simulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0, correctCount: 0 };
    this.realtimeData = [];
    this.confidenceDistribution = { high: 0, medium: 0, low: 0 };
  }

  // Upload, preprocess, training, simulation logic continues here...
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      const file = input.files[0];
      this.processFile(file);
    }
  }

  /**
   * Handles drag over events for drag-and-drop functionality
   */
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = true;
  }

  /**
   * Handles drag leave events
   */
  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
  }

  /**
   * Handles file drop events for drag-and-drop functionality
   */
  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
    const files = event.dataTransfer?.files;
    if (files?.length) {
      const file = files[0];
      this.processFile(file);
    }
  }

  /**
   * Validates and processes the selected file
   * Implements client-side validation as specified in the PDF
   */
  private processFile(file: File): void {
    this.errorMessage = '';
    // Validate file type (only .csv files allowed)
    if (!file.name.toLowerCase().endsWith('.csv')) {
      this.errorMessage = 'Please select a valid CSV file (.csv extension required)';
      return;
    }

    // Validate file size (reasonable limit for demo purposes)
    const maxSizeInMB = 2500;
    const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
    if (file.size > maxSizeInBytes) {
      this.errorMessage = `File size must be less than ${maxSizeInMB}MB`;
      return;
    }

    // File is valid, proceed with upload
    this.selectedFile = file;
    this.uploadFile(file);
  }

  /**
   * Simulates file upload and processing
   * In a real implementation, this would call the /api/data/upload endpoint
   */
  private uploadFile(file: File): void {
    this.isUploading = true;
    this.errorMessage = '';
    this.uploadProgress = 0; // Start progress at 0
    const userId = localStorage.getItem('username') || 'anonymous_user';
    this.uploadService.uploadFileWithProgress(file, userId).subscribe({
      // FIX: The 'next' handler now receives the final UploadResult directly
      next: (result: UploadResult) => {
        // No need to check for event type. This code runs when the upload is complete.
        this.uploadResult = result;
        console.log('Upload complete!', this.uploadResult);
        // Since we know it's complete, we can set progress to 100%
        this.uploadProgress = 100;
      },
      error: (err) => {
        this.errorMessage = 'Failed to upload file. Please try again.';
        console.error('Upload error:', err);
        this.isUploading = false;
        this.uploadProgress = 0; // Reset progress on error
      },
      complete: () => {
        this.isUploading = false;
        // Optional: Reset progress to 0 a moment after completion
        setTimeout(() => this.uploadProgress = 0, 2000);
      }
    });
  }

  /**
   * Removes the selected file and resets the upload state
   */
  removeFile(event: Event): void {
    event.stopPropagation();
    this.selectedFile = null;
    this.uploadResult = null;
    this.errorMessage = '';
  }

  /**
   * Advances to the next step and sets initial date values if applicable.
   */
  next(): void {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      // When moving to the Pre-processing step (index 1), set default dates
      if (this.currentStep === 1 && this.uploadResult) {
        this.dateRanges.training.start = this.formatDateForInput(this.uploadResult.dateRange.start);
        this.dateRanges.simulation.end = this.formatDateForInput(this.uploadResult.dateRange.end);
      }
    }
  }

  /**
   * Navigates to a specific step
   * Only allows backward navigation or navigation to completed steps
   */
  goToStep(index: number): void {
    if (index <= this.currentStep) {
      this.currentStep = index;
    }
  }

  /**
   * Formats file size for display
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Formats date range for display in summary card
   */
  formatDateRange(dateRange: DateRange): string {
    const startDate = new Date(dateRange.start);
    const endDate = new Date(dateRange.end);
    const formatOptions: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    };
    return `${startDate.toLocaleDateString('en-US', formatOptions)} - ${endDate.toLocaleDateString('en-US', formatOptions)}`;
  }

  /**
   * Utility function to create delays for simulation
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Formats a date string or Date object to be compatible with datetime-local input.
   * Strips microseconds and ensures proper 'YYYY-MM-DDTHH:MM:SS' format.
   */
  private formatDateForInput(dateValue: string | Date): string {
    if (!dateValue) {
      return '';
    }

    // Create a new Date object whether the input is a string or a Date
    const date = new Date(dateValue);
    // Check if the date is valid before proceeding
    if (isNaN(date.getTime())) {
      return '';
    }

    const year = date.getFullYear();
    const month = ('0' + (date.getMonth() + 1)).slice(-2);
    const day = ('0' + date.getDate()).slice(-2);
    const hours = ('0' + date.getHours()).slice(-2);
    const minutes = ('0' + date.getMinutes()).slice(-2);
    const seconds = ('0' + date.getSeconds()).slice(-2);
    return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
  }

  /**
   * Automatically splits the date ranges into a 70/20/10 ratio.
   */
  autoSplitRanges(): void {
    if (!this.uploadResult) {
      return;
    }

    const startDate = new Date(this.uploadResult.dateRange.start);
    const endDate = new Date(this.uploadResult.dateRange.end);
    const totalDuration = endDate.getTime() - startDate.getTime();
    // Calculate the end time for the training period (70% of the duration)
    const trainingEndTime = startDate.getTime() + totalDuration * 0.7;
    const trainingEndDate = new Date(trainingEndTime);
    // Calculate the end time for the testing period (another 20% of the duration)
    const testingEndTime = trainingEndTime + totalDuration * 0.2;
    const testingEndDate = new Date(testingEndTime);
    // Set the date ranges
    this.dateRanges = {
      training: {
        start: this.formatDateForInput(startDate),
        end: this.formatDateForInput(trainingEndDate)
      },
      testing: {
        start: this.formatDateForInput(trainingEndDate),
        end: this.formatDateForInput(testingEndDate)
      },
      simulation: {
        start: this.formatDateForInput(testingEndDate),
        end: this.formatDateForInput(endDate) // Ends at the dataset's overall end
      }
    };
    // Trigger validation to update the UI state immediately
    this.validateDateRanges();
  }

  // === DATE RANGES FUNCTIONALITY (STEP 2) ===

  /**
   * Get minimum date for date inputs (from upload result)
   */
  getMinDate(): string {
    return this.uploadResult ? this.formatDateForInput(this.uploadResult.dateRange.start) : '';
  }

  /**
   * Get maximum date for date inputs (from upload result)
   */
  getMaxDate(): string {
    return this.uploadResult ? this.formatDateForInput(this.uploadResult.dateRange.end) : '';
  }

  /**
   * Check if date ranges can be validated
   */
  canValidateRanges(): boolean {
    return !!(
      this.dateRanges.training.start && this.dateRanges.training.end &&
      this.dateRanges.testing.start && this.dateRanges.testing.end &&
      this.dateRanges.simulation.start && this.dateRanges.simulation.end
    );
  }

  /**
   * Validate individual date ranges (client-side validation)
   */
  validateDateRanges(): void {
    // Reset validation state
    this.rangeValidation = {
      training: null,
      testing: null,
      simulation: null
    };
    // Validate training range
    if (this.dateRanges.training.start && this.dateRanges.training.end) {
      this.rangeValidation.training =
        new Date(this.dateRanges.training.start) < new Date(this.dateRanges.training.end);
    }

    // Validate testing range
    if (this.dateRanges.testing.start && this.dateRanges.testing.end) {
      this.rangeValidation.testing =
        new Date(this.dateRanges.testing.start) < new Date(this.dateRanges.testing.end) &&
        new Date(this.dateRanges.testing.start) >= new Date(this.dateRanges.training.end);
    }

    // Validate simulation range
    if (this.dateRanges.simulation.start && this.dateRanges.simulation.end) {
      this.rangeValidation.simulation =
        new Date(this.dateRanges.simulation.start) < new Date(this.dateRanges.simulation.end) &&
        new Date(this.dateRanges.simulation.start) >= new Date(this.dateRanges.testing.end);
    }
  }

  validateRanges(): void {
    if (!this.uploadResult) {
      this.errorMessage = "Cannot validate ranges without an uploaded file.";
      return;
    }

    this.isValidatingRanges = true;
    this.errorMessage = ''; // Clear previous errors

    this.uploadService.validateDateRanges(
      this.uploadResult.datasetId,
      this.uploadResult.userId,
      this.dateRanges
    ).subscribe({
      next: (response: ValidateRangesResponse) => {
        if (response.status === 'Valid') {
          // Update range counts from the backend response
          this.rangeCounts = {
            training: response.training.count,
            testing: response.testing.count,
            simulation: response.simulation.count,
          };

          // Generate timeline data from the backend response
          const total = this.rangeCounts.training + this.rangeCounts.testing + this.rangeCounts.simulation;
          if (total > 0) {
            this.timelineData = [
              {
                type: 'training',
                label: 'Training',
                count: this.rangeCounts.training,
                percentage: (this.rangeCounts.training / total) * 100
              },
              {
                type: 'testing',
                label: 'Testing',
                count: this.rangeCounts.testing,
                percentage: (this.rangeCounts.testing / total) * 100
              },
              {
                type: 'simulation',
                label: 'Simulation',
                count: this.rangeCounts.simulation,
                percentage: (this.rangeCounts.simulation / total) * 100
              }
            ];
          } else {
            this.timelineData = null; // No records found
          }

          // **CRITICAL FIX: Trigger chart creation after successful validation**
          this.updateChartAfterValidation();

          console.log('Monthly Counts:', response.monthlyCounts);
        } else {
          // Handle invalid status from backend if needed, though FastAPI handles this with 400 error
          this.errorMessage = "Validation failed on the server.";
        }
        this.isValidatingRanges = false;
      },
      error: (err) => {
        // Display the detailed error message from the backend
        this.errorMessage = err.error?.detail || err.error || 'An unknown validation error occurred.';
        console.error('Range validation error:', err);
        this.timelineData = null; // Clear timeline on error
        this.isValidatingRanges = false;
      }
    });
  }

  /**
   * Check if all ranges are valid
   */
  areRangesValid(): boolean {
    return !!(
      this.rangeValidation.training === true &&
      this.rangeValidation.testing === true &&
      this.rangeValidation.simulation === true &&
      this.timelineData
    );
  }

  // === TRAINING FUNCTIONALITY (STEP 3) ===

  async trainModel(): Promise<void> {
    if (!this.uploadResult) {
      return;
    }

    if (!this.uploadResult.datasetId || !this.uploadResult.userId || !this.uploadResult.dateRange) {
      console.error('Cannot train model: Missing datasetId, userId, or date ranges.');
      // You should show an error to the user here (e.g., using a toast/snackbar)
      return;
    }

    this.isTraining = true;
    this.trainingProgress = 0;
    this.trainingResults = null; // Clear previous results
    // A real API call doesn't provide granular progress easily.
    // We'll show an initial progress and then jump to 100 on completion.
    // For a better UX, consider an indeterminate progress bar in your HTML template.
    this.trainingProgress = 10;

    try {
      const response = await firstValueFrom(
        this.uploadService.trainModel(this.uploadResult.datasetId, this.uploadResult.userId, this.dateRanges)
      );
      console.log(response);
      this.trainingResults = response;
      this.trainingProgress = 100;
      // If you have other functions that depend on the results, call them here
      this.generateConfusionMatrix(); // Assuming this function exists and uses this.trainingResults
    } catch (error) {
      console.error('Training API call failed:', error);
      // Display a user-friendly error message
      // e.g., this.toastService.showError('Model training failed. Please try again.');
      this.trainingProgress = 0; // Reset progress on error
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * Generate mock confusion matrix data
   */
  private generateConfusionMatrix(): void {
    // Check if we have real results from the backend
    if (!this.trainingResults) {
      console.error("Cannot generate confusion matrix: training results are null.");
      return;
    }

    // Directly assign the real values from the API response
    const metrics = this.trainingResults.metrics;
    console.log(metrics);
    this.confusionMatrix = {
      truePositive: metrics.truePositive,
      falsePositive: metrics.falsePositive,
      trueNegative: metrics.trueNegative,
      falseNegative: metrics.falseNegative
    };
    console.log("Generated real confusion matrix:", this.confusionMatrix);
  }

  // === SIMULATION FUNCTIONALITY (STEP 4) ===

  /**
   * Toggle simulation state
   */
  toggleSimulation(): void {
    if (this.simulationStatus === 'idle' || this.simulationStatus === 'completed') {
      this.startSimulation();
    } else if (this.simulationStatus === 'running') {
      this.stopSimulation();
    }
  }

  /**
   * Starts the simulation by connecting to the WebSocket and subscribing to messages.
   */
  private startSimulation(): void {
    if (this.uploadResult && (!this.uploadResult.userId || !this.uploadResult.datasetId)) {
      console.error("Cannot start simulation: userId or datasetId is missing.");
      return;
    }

    this.resetSimulationData();
    this.simulationStatus = 'running';

    this.simulationService.connect();
    this.simSubscription = this.simulationService.messages$.subscribe({
      next: (message: SimulationMessage) => this.processSimulationMessage(message),
      error: (err) => {
        console.error('WebSocket Error:', err);
        this.simulationStatus = 'idle'; // Reset on error
      },
      complete: () => {
        console.log('WebSocket stream completed.');
        this.simulationStatus = 'completed';
      }
    });

    // Send the start command after connecting
    this.simulationService.startSimulation(this.uploadResult!.userId, this.uploadResult!.datasetId);
  }

  /**
   * Sends a stop command to the WebSocket.
   */
  private stopSimulation(): void {
    if (this.uploadResult && (!this.uploadResult.userId || !this.uploadResult.datasetId)) return;
    this.simulationStatus = 'stopping';
    this.simulationService.stopSimulation(this.uploadResult!.userId, this.uploadResult!.datasetId);
  }

  /**
   * Processes each message received from the WebSocket server.
   */
  private processSimulationMessage(message: SimulationMessage): void {
    if ('rowIndex' in message) {
      // It's a data message
      const result = message as SimulationResult;
      const newPrediction: RealtimePrediction = {
        rowIndex: result.rowIndex,
        timestamp: new Date(result.timestamp),
        prediction: result.prediction === 1 ? 'pass' : 'fail',
        confidence: result.confidence,
        isCorrect: result.isCorrect
      };

      this.realtimeData.push(newPrediction);
      this.updateSimulationStats(newPrediction);
      this.updateConfidenceDistribution();
    } else if (message.status) {
      // It's a status message
      if (message.status === 'stopped' || message.status === 'finished') {
        this.simulationStatus = 'completed';
        this.simSubscription?.unsubscribe();
      }
    } else if (message.error) {
      console.error("Error from simulation server:", message.error);
      this.stopSimulation();
    }
  }

  /**
   * Resets all data and stats for a new simulation run.
   */
  private resetSimulationData(): void {
    this.realtimeData = [];
    this.simulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, correctCount: 0, currentAccuracy: 0 };
    this.updateConfidenceDistribution();
  }

  /**
   * Updates the running statistics with each new prediction.
   */
  private updateSimulationStats(prediction: RealtimePrediction): void {
    this.simulationStats.totalPredictions++;
    prediction.prediction === 'pass' ? this.simulationStats.passCount++ : this.simulationStats.failCount++;
    if (prediction.isCorrect) {
      this.simulationStats.correctCount++;
    }

    this.simulationStats.currentAccuracy = (this.simulationStats.correctCount / this.simulationStats.totalPredictions) * 100;
  }

  /**
   * Recalculates the confidence distribution pie chart data.
   */
  private updateConfidenceDistribution(): void {
    if (this.realtimeData.length === 0) {
      this.confidenceDistribution = { high: 0, medium: 0, low: 0 };
      return;
    }

    let high = 0, medium = 0, low = 0;
    this.realtimeData.forEach(p => {
      if (p.confidence > 0.9) high++;
      else if (p.confidence > 0.8) medium++;
      else low++;
    });

    const total = this.realtimeData.length;
    this.confidenceDistribution = {
      high: (high / total) * 100,
      medium: (medium / total) * 100,
      low: (low / total) * 100
    };
  }

  /**
   * Get simulation status text
   */
  getSimulationStatusText(): string {
    switch (this.simulationStatus) {
      case 'idle': return 'Ready to start';
      case 'running': return 'Streaming predictions';
      case 'stopping': return 'Stopping...';
      case 'completed': return 'Simulation complete';
      default: return 'Unknown';
    }
  }

  /**
   * Get simulation button text
   */
  getSimulationButtonText(): string {
    switch (this.simulationStatus) {
      case 'idle': return 'Start Simulation';
      case 'running': return 'Stop Simulation';
      case 'stopping': return 'Stopping...';
      case 'completed': return 'Restart Simulation';
      default: return 'Start Simulation';
    }
  }

  /**
   * Calculates and returns a human-readable duration between two dates
   * @param start - Start date (Date object or ISO string)
   * @param end - End date (Date object or ISO string)
   * @returns Formatted duration string (e.g., "2 hours", "5 days", "3 months")
   */
  getDuration(start: Date | string, end: Date | string): string {
    if (!start || !end) {
      return 'Invalid dates';
    }

    // Convert strings to Date objects if needed
    const startDate = typeof start === 'string' ? new Date(start) : start;
    const endDate = typeof end === 'string' ? new Date(end) : end;

    // Check for invalid dates
    if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
      return 'Invalid dates';
    }

    if (startDate >= endDate) {
      return 'Invalid range';
    }

    const diffMs = endDate.getTime() - startDate.getTime();

    // Convert to different time units
    const minutes = Math.floor(diffMs / (1000 * 60));
    const hours = Math.floor(diffMs / (1000 * 60 * 60));
    const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const weeks = Math.floor(days / 7);
    const months = Math.floor(days / 30.44); // Average days per month
    const years = Math.floor(days / 365.25); // Account for leap years

    // Return the most appropriate unit
    if (years >= 1) {
      return years === 1 ? '1 year' : `${years} years`;
    } else if (months >= 1) {
      return months === 1 ? '1 month' : `${months} months`;
    } else if (weeks >= 1) {
      return weeks === 1 ? '1 week' : `${weeks} weeks`;
    } else if (days >= 1) {
      return days === 1 ? '1 day' : `${days} days`;
    } else if (hours >= 1) {
      return hours === 1 ? '1 hour' : `${hours} hours`;
    } else if (minutes >= 1) {
      return minutes === 1 ? '1 minute' : `${minutes} minutes`;
    } else {
      return 'Less than 1 minute';
    }
  }

  /**
   * Formats a date range for display
   * @param start - Start date (Date object or ISO string)
   * @param end - End date (Date object or ISO string)
   * @returns Formatted date range string (e.g., "Jan 15 - Feb 28, 2024")
   */
  formatDateRangeChart(start: Date | string, end: Date | string): string {
    if (!start || !end) {
      return 'Invalid dates';
    }

    // Convert strings to Date objects if needed
    const startDate = typeof start === 'string' ? new Date(start) : start;
    const endDate = typeof end === 'string' ? new Date(end) : end;

    // Check for invalid dates
    if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
      return 'Invalid dates';
    }

    if (startDate >= endDate) {
      return 'Invalid range';
    }

    const options: Intl.DateTimeFormatOptions = {
      month: 'short',
      day: 'numeric'
    };

    const optionsWithYear: Intl.DateTimeFormatOptions = {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    };

    // Check if both dates are in the same year
    if (startDate.getFullYear() === endDate.getFullYear()) {
      // Same year - show year only at the end
      const startFormatted = startDate.toLocaleDateString('en-US', options);
      const endFormatted = endDate.toLocaleDateString('en-US', optionsWithYear);
      return `${startFormatted} - ${endFormatted}`;
    } else {
      // Different years - show year for both dates
      const startFormatted = startDate.toLocaleDateString('en-US', optionsWithYear);
      const endFormatted = endDate.toLocaleDateString('en-US', optionsWithYear);
      return `${startFormatted} - ${endFormatted}`;
    }
  }

  // ========== CHART.JS FUNCTIONALITY ==========

  /**
   * Processes dateRanges to calculate monthly distribution of training, testing, and simulation days
   */
  private processDateRangesForChart(): MonthlyData[] {
    if (!this.dateRanges.training.start || !this.dateRanges.simulation.end) {
      return [];
    }

    // Get overall date range
    const overallStart = new Date(this.dateRanges.training.start);
    const overallEnd = new Date(this.dateRanges.simulation.end);

    // Create array to store monthly data
    const monthlyDataMap = new Map<string, MonthlyData>();

    // Initialize months between start and end dates
    const currentDate = new Date(overallStart.getFullYear(), overallStart.getMonth(), 1);
    const endDate = new Date(overallEnd.getFullYear(), overallEnd.getMonth() + 1, 0);

    while (currentDate <= endDate) {
      const monthKey = `${currentDate.getFullYear()}-${currentDate.getMonth()}`;
      const monthName = currentDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });

      monthlyDataMap.set(monthKey, {
        month: monthName,
        year: currentDate.getFullYear(),
        trainingDays: 0,
        testingDays: 0,
        simulationDays: 0,
        totalDays: 0
      });

      currentDate.setMonth(currentDate.getMonth() + 1);
    }

    // Calculate training days per month
    this.calculateDaysPerMonth(
      new Date(this.dateRanges.training.start),
      new Date(this.dateRanges.training.end),
      monthlyDataMap,
      'training'
    );

    // Calculate testing days per month
    this.calculateDaysPerMonth(
      new Date(this.dateRanges.testing.start),
      new Date(this.dateRanges.testing.end),
      monthlyDataMap,
      'testing'
    );

    // Calculate simulation days per month
    this.calculateDaysPerMonth(
      new Date(this.dateRanges.simulation.start),
      new Date(this.dateRanges.simulation.end),
      monthlyDataMap,
      'simulation'
    );

    // Calculate total days for each period to get records per day
    let totalTrainingDays = 0;
    let totalTestingDays = 0;
    let totalSimulationDays = 0;

    Array.from(monthlyDataMap.values()).forEach(data => {
      totalTrainingDays += data.trainingDays;
      totalTestingDays += data.testingDays;
      totalSimulationDays += data.simulationDays;
    });

    // Calculate records per day for each period using actual rangeCounts
    const trainingRecordsPerDay = totalTrainingDays > 0 ? this.rangeCounts.training / totalTrainingDays : 0;
    const testingRecordsPerDay = totalTestingDays > 0 ? this.rangeCounts.testing / totalTestingDays : 0;
    const simulationRecordsPerDay = totalSimulationDays > 0 ? this.rangeCounts.simulation / totalSimulationDays : 0;

    // Convert days to actual record counts using the real data
    const monthlyData = Array.from(monthlyDataMap.values()).map(data => {
      // Calculate actual records for each period in this month
      const trainingRecords = Math.round(data.trainingDays * trainingRecordsPerDay);
      const testingRecords = Math.round(data.testingDays * testingRecordsPerDay);
      const simulationRecords = Math.round(data.simulationDays * simulationRecordsPerDay);

      return {
        ...data,
        trainingDays: trainingRecords,
        testingDays: testingRecords,
        simulationDays: simulationRecords,
        totalDays: trainingRecords + testingRecords + simulationRecords
      };
    });

    return monthlyData.filter(data => data.totalDays > 0);
  }

  /**
   * Helper method to calculate how many days of a period fall into each month
   */
  private calculateDaysPerMonth(
    startDate: Date,
    endDate: Date,
    monthlyDataMap: Map<string, MonthlyData>,
    periodType: 'training' | 'testing' | 'simulation'
  ): void {
    const currentDate = new Date(startDate);

    while (currentDate <= endDate) {
      const monthKey = `${currentDate.getFullYear()}-${currentDate.getMonth()}`;
      const monthData = monthlyDataMap.get(monthKey);

      if (monthData) {
        // Use switch statement for type safety
        switch (periodType) {
          case 'training':
            monthData.trainingDays += 1;
            break;
          case 'testing':
            monthData.testingDays += 1;
            break;
          case 'simulation':
            monthData.simulationDays += 1;
            break;
        }
      }

      currentDate.setDate(currentDate.getDate() + 1);
    }
  }

  /**
   * Creates and configures the Chart.js stacked bar chart
   */
  private createMonthlyChart(): void {
    console.log('Creating monthly chart...');

    // Check if canvas is available
    if (!this.chartCanvas?.nativeElement) {
      console.error('Chart canvas not available');
      return;
    }

    // Check if we have the required data
    if (!this.dateRanges.training.start || !this.dateRanges.simulation.end) {
      console.warn('Date ranges not available for chart');
      return;
    }

    // Process the dateRanges data
    const monthlyData = this.processDateRangesForChart();

    if (monthlyData.length === 0) {
      console.warn('No monthly data available for chart');
      return;
    }

    console.log('Monthly data for chart:', monthlyData);

    // Prepare chart data
    const labels = monthlyData.map(data => data.month);
    const trainingData = monthlyData.map(data => data.trainingDays);
    const testingData = monthlyData.map(data => data.testingDays);
    const simulationData = monthlyData.map(data => data.simulationDays);

    // Destroy existing chart if it exists
    if (this.monthlyChart) {
      this.monthlyChart.destroy();
      this.monthlyChart = null;
    }

    // Get canvas context
    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) {
      console.error('Could not get canvas context');
      return;
    }

    // Chart configuration
    const config: ChartConfiguration = {
      type: 'bar' as ChartType,
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Training',
            data: trainingData,
            backgroundColor: this.chartColors.training,
            borderColor: this.chartColors.training,
            borderWidth: 1,
            stack: 'records'
          },
          {
            label: 'Testing',
            data: testingData,
            backgroundColor: this.chartColors.testing,
            borderColor: this.chartColors.testing,
            borderWidth: 1,
            stack: 'records'
          },
          {
            label: 'Simulation',
            data: simulationData,
            backgroundColor: this.chartColors.simulation,
            borderColor: this.chartColors.simulation,
            borderWidth: 1,
            stack: 'records'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          title: {
            display: true,
            text: 'Monthly Data Distribution',
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 20
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: (context) => {
                const label = context.dataset.label || '';
                const value = context.parsed.y || 0;
                return `${label}: ${value.toLocaleString()} records`;
              },
              footer: (tooltipItems) => {
                const total = tooltipItems.reduce((sum, item) => sum + (item.parsed.y || 0), 0);
                return `Total: ${total.toLocaleString()} records`;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Time Period (Months)',
              font: {
                size: 14,
                weight: 'bold'
              }
            },
            grid: {
              display: false
            }
          },
          y: {
            title: {
              display: true,
              text: 'Number of Records',
              font: {
                size: 14,
                weight: 'bold'
              }
            },
            beginAtZero: true,
            stacked: true,
            ticks: {
              callback: function (value) {
                return typeof value === 'number' ? value.toLocaleString() : value;
              }
            }
          }
        }
      }
    };

    try {
      // Create new chart
      this.monthlyChart = new Chart(ctx, config);
      console.log('Chart created successfully');
    } catch (error) {
      console.error('Error creating chart:', error);
    }
  }

  /**
   * Public method to update the chart when dateRanges change
   */
  updateMonthlyChart(): void {
    this.createMonthlyChart();
  }

  /**
   * Call this method after validateRanges() succeeds to update the chart
   */
  private updateChartAfterValidation(): void {
    console.log('Updating chart after validation...');
    // Use a longer timeout to ensure DOM is ready
    setTimeout(() => {
      this.updateMonthlyChart();
    }, 500);
  }

  getTotal(): number {
    if (!this.confusionMatrix) return 0;
    const { truePositive, falsePositive, falseNegative, trueNegative } = this.confusionMatrix;
    return truePositive + falsePositive + falseNegative + trueNegative;
  }

  getSegmentPercentage(value: number): number {
    if (!value) return 0;
    const total = this.getTotal();
    if (total === 0) {
      return 0;
    }
    // Return the percentage of the circle this value represents
    return (value / total) * 100;
  }
}
