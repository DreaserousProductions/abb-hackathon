import { HttpClient, HttpHeaders, HttpRequest, HttpEvent } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment.development';

// --- DATA UPLOAD & VALIDATION INTERFACES ---
export interface DateRange {
  start: string;
  end: string;
}

export interface DateRanges {
  training: DateRange;
  testing: DateRange;
  simulation: DateRange;
}

export interface UploadResult {
  datasetId: string;
  userId: string;
  totalRecords: number;
  numColumns: number;
  passRate: number;
  dateRange: DateRange;
}

export interface ValidateRangesResponse {
  status: 'Valid' | 'Invalid';
  training: { count: number };
  testing: { count: number };
  simulation: { count: number };
  monthlyCounts: { [key: string]: number };
}

// --- NEW MODEL TRAINING INTERFACES ---
export interface TrainingMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  truePositive: number;
}

export interface TrainingPlots {
  featureImportance: string;
}

export interface TrainModelResponse {
  metrics: TrainingMetrics;
  plots: TrainingPlots;
}


@Injectable({
  providedIn: 'root'
})
export class UploadService {
  private dataApiUrl = `${environment.apiUrl}/Data`;
  private modelApiUrl = `${environment.apiUrl}/Model`; // <-- New URL for the ModelController

  constructor(private http: HttpClient) { }

  uploadFileWithProgress(file: File, userId: string): Observable<HttpEvent<UploadResult>> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('userId', userId);

    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

    const req = new HttpRequest('POST', `${this.dataApiUrl}/upload`, formData, {
      headers,
      reportProgress: true
    });

    return this.http.request<UploadResult>(req);
  }

  validateDateRanges(
    datasetId: string,
    userId: string,
    dateRanges: DateRanges
  ): Observable<ValidateRangesResponse> {
    const payload = { datasetId, userId, dateRanges };
    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

    return this.http.post<ValidateRangesResponse>(`${this.dataApiUrl}/validate-ranges`, payload, { headers });
  }

  // --- NEW METHOD FOR MODEL TRAINING ---
  trainModel(
    datasetId: string,
    userId: string,
    dateRanges: DateRanges
  ): Observable<TrainModelResponse> {
    const payload = { datasetId, userId, dateRanges };
    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

    // Call the new /api/Model/train endpoint
    return this.http.post<TrainModelResponse>(`${this.modelApiUrl}/train`, payload, { headers });
  }
}