import { CommonModule } from '@angular/common';
import { Component, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TrainModelResponse, UploadResult, UploadService, ValidateRangesResponse } from '../../services/upload/upload.service';
import { HttpEvent, HttpEventType } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

// Interfaces
interface DateRange { start: string; end: string; }
interface DateRanges { training: DateRange; testing: DateRange; simulation: DateRange; }
interface RangeValidation { training: boolean | null; testing: boolean | null; simulation: boolean | null; }
interface RangeCounts { training: number; testing: number; simulation: number; }
interface TimelineData { type: string; label: string; count: number; percentage: number; }
interface TrainingMetrics { accuracy: number; precision: number; recall: number; f1Score: number; }
interface TrainingResult { metrics: TrainingMetrics; plots: { featureImportance: string; }; }
interface ConfusionMatrix { truePositive: number; falsePositive: number; trueNegative: number; falseNegative: number; }
interface SimulationStats { totalPredictions: number; passCount: number; failCount: number; currentAccuracy: number; }
interface RealtimePrediction { timestamp: Date; prediction: 'pass' | 'fail'; confidence: number; }
interface ConfidenceDistribution { high: number; medium: number; low: number; }
type SimulationStatus = 'idle' | 'running' | 'stopping' | 'completed';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css'
})
export class UploadComponent implements OnDestroy {
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
  simulationStats: SimulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0 };
  realtimeData: RealtimePrediction[] = [];
  confidenceDistribution: ConfidenceDistribution = { high: 0, medium: 0, low: 0 };

  // Intervals
  private simulationInterval: any;
  private progressInterval: any;

  constructor(private uploadService: UploadService) { }

  ngOnDestroy(): void { this.clearIntervals(); }

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
    this.simulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0 };
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

    // Get the username from localStorage. Provide a fallback if it doesn't exist.
    const userId = localStorage.getItem('username') || 'anonymous_user';

    // Pass the file and userId to the service
    this.uploadService.uploadFileWithProgress(file, userId).subscribe({
      next: (event: HttpEvent<UploadResult>) => {
        if (event.type === HttpEventType.UploadProgress) {
          if (event.total) {
            this.uploadProgress = Math.round(100 * event.loaded / event.total);
          }
        } else if (event.type === HttpEventType.Response) {
          this.uploadResult = event.body;
          console.log('Upload complete!', this.uploadResult);
          this.uploadProgress = 0;
        }
      },
      error: (err) => {
        this.errorMessage = 'Failed to upload file. Please try again.';
        console.error('Upload error:', err);
        this.isUploading = false;
      },
      complete: () => {
        this.isUploading = false;
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
   * Resets the entire upload component to initial state
   */
  // reset(): void {
  //   this.selectedFile = null;
  //   this.uploadResult = null;
  //   this.errorMessage = '';
  //   this.isUploading = false;
  //   this.isDragOver = false;
  // }

  /**
   * Advances to the next step
   * Only enabled when upload is complete as per PDF requirements
   */
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

  // ADD THIS NEW METHOD
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

          // You can now also use response.monthlyCounts to display another chart if you wish
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

  /**
   * Format date range for display
   */
  // formatDateRange(range: DateRange): string {
  //   if (!range.start || !range.end) return 'Not set';

  //   const startDate = new Date(range.start);
  //   const endDate = new Date(range.end);

  //   const formatOptions: Intl.DateTimeFormatOptions = {
  //     year: 'numeric',
  //     month: 'short',
  //     day: 'numeric'
  //   };

  //   return `${startDate.toLocaleDateString('en-US', formatOptions)} - ${endDate.toLocaleDateString('en-US', formatOptions)}`;
  // }

  // === TRAINING FUNCTIONALITY (STEP 3) ===

  /**
   * Start model training process
   */
  // async trainModel(): Promise<void> {
  //   this.isTraining = true;
  //   this.trainingProgress = 0;

  //   try {
  //     // Simulate training progress
  //     this.progressInterval = setInterval(() => {
  //       if (this.trainingProgress < 100) {
  //         this.trainingProgress += Math.random() * 15;
  //         if (this.trainingProgress > 100) {
  //           this.trainingProgress = 100;
  //         }
  //       }
  //     }, 200);

  //     // Simulate training time
  //     await this.delay(4000);

  //     // Generate mock training results
  //     this.trainingResults = {
  //       metrics: {
  //         accuracy: 0.92 + (Math.random() * 0.06), // 92-98%
  //         precision: 0.89 + (Math.random() * 0.08), // 89-97%
  //         recall: 0.87 + (Math.random() * 0.09), // 87-96%
  //         f1Score: 0.88 + (Math.random() * 0.08) // 88-96%
  //       },
  //       plots: {
  //         featureImportance: this.generateMockSHAPPlot()
  //       }
  //     };

  //     // Generate confusion matrix
  //     this.generateConfusionMatrix();

  //   } catch (error) {
  //     console.error('Training error:', error);
  //   } finally {
  //     this.isTraining = false;
  //     this.trainingProgress = 100;
  //     if (this.progressInterval) {
  //       clearInterval(this.progressInterval);
  //     }
  //   }
  // }

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
   * Generate mock SHAP plot (base64 encoded)
   */
  private generateMockSHAPPlot(): string {
    // This would be a real base64 encoded SHAP plot from the backend
    // For demo purposes, using a placeholder
    return 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
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
   * Start real-time simulation
   */
  private startSimulation(): void {
    this.simulationStatus = 'running';
    this.resetSimulationData();

    // Simulate real-time predictions every second
    this.simulationInterval = setInterval(() => {
      this.generateRealtimePrediction();

      // Stop after certain number of predictions for demo
      if (this.simulationStats.totalPredictions >= 100) {
        this.completeSimulation();
      }
    }, 1000);
  }

  /**
   * Stop simulation
   */
  private stopSimulation(): void {
    this.simulationStatus = 'stopping';
    this.clearIntervals();

    setTimeout(() => {
      this.simulationStatus = 'completed';
    }, 1000);
  }

  /**
   * Complete simulation
   */
  private completeSimulation(): void {
    this.clearIntervals();
    this.simulationStatus = 'completed';
  }

  /**
   * Reset simulation for new run
   */
  // resetSimulation(): void {
  //   this.resetSimulationData();
  //   this.simulationStatus = 'idle';
  // }

  /**
   * Reset simulation data
   */
  private resetSimulationData(): void {
    this.simulationStats = {
      totalPredictions: 0,
      passCount: 0,
      failCount: 0,
      currentAccuracy: 0
    };
    this.realtimeData = [];
    this.updateConfidenceDistribution();
  }

  /**
   * Generate a single real-time prediction
   */
  private generateRealtimePrediction(): void {
    const prediction: RealtimePrediction = {
      timestamp: new Date(),
      prediction: Math.random() > 0.15 ? 'pass' : 'fail', // 85% pass rate
      confidence: 0.7 + (Math.random() * 0.3) // 70-100% confidence
    };

    this.realtimeData.push(prediction);
    this.updateSimulationStats(prediction);
    this.updateConfidenceDistribution();
  }

  /**
   * Update simulation statistics
   */
  private updateSimulationStats(prediction: RealtimePrediction): void {
    this.simulationStats.totalPredictions++;

    if (prediction.prediction === 'pass') {
      this.simulationStats.passCount++;
    } else {
      this.simulationStats.failCount++;
    }

    // Calculate current accuracy (mock actual vs predicted)
    const mockActualPass = Math.random() > 0.12; // 88% actual pass rate
    const predictedCorrectly =
      (prediction.prediction === 'pass' && mockActualPass) ||
      (prediction.prediction === 'fail' && !mockActualPass);

    if (predictedCorrectly) {
      this.simulationStats.currentAccuracy =
        ((this.simulationStats.currentAccuracy * (this.simulationStats.totalPredictions - 1)) + 100) /
        this.simulationStats.totalPredictions;
    } else {
      this.simulationStats.currentAccuracy =
        (this.simulationStats.currentAccuracy * (this.simulationStats.totalPredictions - 1)) /
        this.simulationStats.totalPredictions;
    }
  }

  /**
   * Update confidence distribution
   */
  private updateConfidenceDistribution(): void {
    if (this.realtimeData.length === 0) {
      this.confidenceDistribution = { high: 0, medium: 0, low: 0 };
      return;
    }

    let high = 0, medium = 0, low = 0;

    this.realtimeData.forEach(pred => {
      if (pred.confidence > 0.9) high++;
      else if (pred.confidence > 0.7) medium++;
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
}