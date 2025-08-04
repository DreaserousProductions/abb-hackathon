import { CommonModule } from '@angular/common';
import { Component, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';

// Interfaces
interface DateRange { start: string; end: string; }
interface UploadResult { totalRecords: number; numColumns: number; passRate: number; dateRange: DateRange; }
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
  currentStep = 3;
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
  trainingResults: TrainingResult | null = null;
  confusionMatrix: ConfusionMatrix = { truePositive: 0, falsePositive: 0, trueNegative: 0, falseNegative: 0 };

  // Simulation state
  simulationStatus: SimulationStatus = 'idle';
  simulationStats: SimulationStats = { totalPredictions: 0, passCount: 0, failCount: 0, currentAccuracy: 0 };
  realtimeData: RealtimePrediction[] = [];
  confidenceDistribution: ConfidenceDistribution = { high: 0, medium: 0, low: 0 };

  // Intervals
  private simulationInterval: any;
  private progressInterval: any;

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
    const maxSizeInMB = 100;
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
  private async uploadFile(file: File): Promise<void> {
    this.isUploading = true;
    this.errorMessage = '';

    try {
      // Simulate API call delay
      await this.delay(2000);

      // In a real implementation, you would:
      // 1. Create FormData with the file
      // 2. POST to /api/data/upload endpoint
      // 3. Handle the response according to the API contract

      // Mock response data matching the API contract from the PDF
      this.uploadResult = {
        totalRecords: this.generateMockRecordCount(),
        numColumns: this.generateMockColumnCount(),
        passRate: this.generateMockPassRate(),
        dateRange: {
          start: this.generateMockStartDate(),
          end: this.generateMockEndDate()
        }
      };

      console.log('File uploaded successfully:', this.uploadResult);

    } catch (error) {
      this.errorMessage = 'Failed to upload file. Please try again.';
      console.error('Upload error:', error);
    } finally {
      this.isUploading = false;
    }
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
  next(): void {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
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

  // Mock data generators for demonstration purposes
  // In a real implementation, this data would come from the ML service

  private generateMockRecordCount(): number {
    // Generate a realistic record count for manufacturing data
    return Math.floor(Math.random() * 900000) + 100000; // Between 100k and 1M
  }

  private generateMockColumnCount(): number {
    // Manufacturing datasets typically have many columns
    return Math.floor(Math.random() * 150) + 50; // Between 50 and 200 columns
  }

  private generateMockPassRate(): number {
    // Generate a realistic pass rate (typically high in manufacturing)
    return 0.85 + (Math.random() * 0.14); // Between 85% and 99%
  }

  private generateMockStartDate(): string {
    // Generate a start date in the past year
    const now = new Date();
    const yearAgo = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
    const randomDays = Math.floor(Math.random() * 365);
    const startDate = new Date(yearAgo.getTime() + (randomDays * 24 * 60 * 60 * 1000));
    return startDate.toISOString().split('T')[0];
  }

  private generateMockEndDate(): string {
    // Generate an end date more recent than start date
    const now = new Date();
    const daysAgo = Math.floor(Math.random() * 30); // Within last 30 days
    const endDate = new Date(now.getTime() - (daysAgo * 24 * 60 * 60 * 1000));
    return endDate.toISOString().split('T')[0];
  }

  // === DATE RANGES FUNCTIONALITY (STEP 2) ===

  /**
   * Get minimum date for date inputs (from upload result)
   */
  getMinDate(): string {
    return this.uploadResult?.dateRange.start || '';
  }

  /**
   * Get maximum date for date inputs (from upload result)
   */
  getMaxDate(): string {
    return this.uploadResult?.dateRange.end || '';
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

  /**
   * Validate ranges against backend (simulated API call)
   */
  async validateRanges(): Promise<void> {
    this.isValidatingRanges = true;

    try {
      // Simulate API call delay
      await this.delay(1500);

      // In a real implementation, this would call /api/data/validate-ranges
      // Generate mock data for demonstration
      this.rangeCounts = {
        training: Math.floor(Math.random() * 50000) + 10000,
        testing: Math.floor(Math.random() * 20000) + 5000,
        simulation: Math.floor(Math.random() * 15000) + 3000
      };

      // Generate timeline data
      const total = this.rangeCounts.training + this.rangeCounts.testing + this.rangeCounts.simulation;
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

    } catch (error) {
      console.error('Range validation error:', error);
    } finally {
      this.isValidatingRanges = false;
    }
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
  async trainModel(): Promise<void> {
    this.isTraining = true;
    this.trainingProgress = 0;

    try {
      // Simulate training progress
      this.progressInterval = setInterval(() => {
        if (this.trainingProgress < 100) {
          this.trainingProgress += Math.random() * 15;
          if (this.trainingProgress > 100) {
            this.trainingProgress = 100;
          }
        }
      }, 200);

      // Simulate training time
      await this.delay(4000);

      // Generate mock training results
      this.trainingResults = {
        metrics: {
          accuracy: 0.92 + (Math.random() * 0.06), // 92-98%
          precision: 0.89 + (Math.random() * 0.08), // 89-97%
          recall: 0.87 + (Math.random() * 0.09), // 87-96%
          f1Score: 0.88 + (Math.random() * 0.08) // 88-96%
        },
        plots: {
          featureImportance: this.generateMockSHAPPlot()
        }
      };

      // Generate confusion matrix
      this.generateConfusionMatrix();

    } catch (error) {
      console.error('Training error:', error);
    } finally {
      this.isTraining = false;
      this.trainingProgress = 100;
      if (this.progressInterval) {
        clearInterval(this.progressInterval);
      }
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
    const totalTest = this.rangeCounts.testing;
    const accuracy = this.trainingResults?.metrics.accuracy || 0.92;

    const truePositive = Math.floor(totalTest * 0.4 * accuracy);
    const trueNegative = Math.floor(totalTest * 0.6 * accuracy);
    const falsePositive = Math.floor(totalTest * 0.05);
    const falseNegative = totalTest - truePositive - trueNegative - falsePositive;

    this.confusionMatrix = {
      truePositive,
      falsePositive,
      trueNegative,
      falseNegative
    };
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