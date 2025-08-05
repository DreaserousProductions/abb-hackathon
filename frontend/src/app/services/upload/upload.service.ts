// src/app/services/upload.service.ts

import { HttpClient, HttpHeaders, HttpEventType, HttpRequest, HttpEvent } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment.development';

// --- INTERFACES ---
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
  monthlyCounts: { [key: string]: number }; // e.g., { "2025-08": 1234 }
}

@Injectable({
  providedIn: 'root'
})
export class UploadService {
  private apiUrl = `${environment.apiUrl}/Data`;

  constructor(private http: HttpClient) { }

  uploadFileWithProgress(file: File, userId: string): Observable<HttpEvent<UploadResult>> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('userId', userId);

    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

    const req = new HttpRequest('POST', `${this.apiUrl}/upload`, formData, {
      headers,
      reportProgress: true
    });

    return this.http.request<UploadResult>(req);
  }

  // --- NEW VALIDATION METHOD ---
  validateDateRanges(
    datasetId: string,
    userId: string,
    dateRanges: DateRanges
  ): Observable<ValidateRangesResponse> {
    const payload = { datasetId, userId, dateRanges };
    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

    return this.http.post<ValidateRangesResponse>(`${this.apiUrl}/validate-ranges`, payload, { headers });
  }
}
