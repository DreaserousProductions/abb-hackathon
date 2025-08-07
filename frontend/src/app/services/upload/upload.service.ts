import { HttpClient, HttpHeaders, HttpRequest, HttpEvent } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { catchError, EMPTY, forkJoin, from, Observable, switchMap, tap } from 'rxjs';
import { environment } from '../../../environments/environment.development';
import { v4 as uuidv4 } from 'uuid';
// import * as pako from 'pako';

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
  parquetPath: string; // Add this line
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
  trainingPlot: string;
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

  // uploadFileWithProgress(file: File, userId: string): Observable<HttpEvent<UploadResult>> {
  //   const formData = new FormData();
  //   formData.append('file', file, file.name);
  //   formData.append('userId', userId);

  //   const token = localStorage.getItem('authToken');
  //   const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

  //   const req = new HttpRequest('POST', `${this.dataApiUrl}/upload`, formData, {
  //     headers,
  //     reportProgress: true
  //   });

  //   return this.http.request<UploadResult>(req);
  // }

  // The master function for chunked uploads
  // uploadFileWithProgress(file: File, userId: string): Observable<any> {
  //   const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunks
  //   const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  //   const uploadId = uuidv4(); // A unique ID for this specific upload
  //   const chunkRequests: Observable<any>[] = [];

  //   console.log(`Starting chunked upload for ${file.name} with ID: ${uploadId}`);
  //   console.log(`Total chunks: ${totalChunks}`);

  //   for (let i = 0; i < totalChunks; i++) {
  //     const start = i * CHUNK_SIZE;
  //     const end = Math.min(start + CHUNK_SIZE, file.size);
  //     const chunk = file.slice(start, end);

  //     chunkRequests.push(this.uploadChunk(chunk, i, uploadId, userId));
  //   }

  //   // `forkJoin` executes all chunk uploads in parallel and waits for them to complete.
  //   return forkJoin(chunkRequests).pipe(
  //     // When all chunks are uploaded, call the 'finish' endpoint
  //     switchMap(responses => {
  //       console.log("All chunks uploaded successfully. Finishing upload...");
  //       return this.finishUpload(uploadId, file.name, userId, totalChunks);
  //     }),
  //     catchError(error => {
  //       console.error("Error during chunk upload:", error);
  //       // You can add more robust error handling or retry logic here
  //       throw new Error("Chunk upload failed.");
  //     })
  //   );
  // }

  uploadFileWithProgress(file: File, userId: string): Observable<number | UploadResult> {
    return new Observable(observer => {
      const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunks
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
      const uploadId = uuidv4();
      let chunksUploaded = 0;

      console.log(`Starting chunked upload for ${file.name} with ID: ${uploadId}`);
      console.log(`Total chunks: ${totalChunks}`);

      const chunkRequests: Observable<any>[] = [];
      for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunk = file.slice(start, end);

        // Wrap each chunk upload to report progress on its completion
        chunkRequests.push(this.uploadChunk(chunk, i, uploadId, userId).pipe(
          tap(() => {
            chunksUploaded++;
            // Calculate progress up to 99% as chunks upload
            const progress = Math.round((chunksUploaded / totalChunks) * 100);
            observer.next(progress);
          })
        ));
      }

      // forkJoin runs all chunk uploads in parallel
      forkJoin(chunkRequests).pipe(
        switchMap(() => {
          console.log("All chunks uploaded successfully. Finishing upload...");
          // When all chunks are done, call the 'finish' endpoint
          return this.finishUpload(uploadId, file.name, userId, totalChunks);
        }),
        catchError(error => {
          console.error("Error during chunk upload:", error);
          observer.error("Chunk upload failed.");
          return EMPTY; // Stop the observable stream on error
        })
      ).subscribe({
        next: (finalResult: UploadResult) => {
          observer.next(finalResult); // Emit the final result object
          observer.complete();
        },
        error: (err) => {
          observer.error(err);
        }
      });
    });
  }

  private uploadChunk(chunk: Blob, chunkIndex: number, uploadId: string, userId: string): Observable<any> {
    const formData = new FormData();
    formData.append('file', chunk);
    formData.append('uploadId', uploadId);
    formData.append('chunkIndex', chunkIndex.toString());
    formData.append('userId', userId);

    // Manually get token and create headers
    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });

    // Pass the headers with the request
    return this.http.post(`${this.dataApiUrl}/upload-chunk`, formData, { headers });
  }

  private finishUpload(uploadId: string, fileName: string, userId: string, totalChunks: number): Observable<any> {
    const body = { uploadId, fileName, userId, totalChunks };

    // Manually get token and create headers
    const token = localStorage.getItem('authToken');
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });

    // Pass the headers with the request
    return this.http.post(`${this.dataApiUrl}/finish-upload`, body, { headers });
  }


  // Compress
  // uploadFileWithProgress(file: File, userId: string): Observable<HttpEvent<UploadResult>> {
  //   // NEW: We use RxJS `from` to convert our async compression logic (a Promise) into an Observable stream
  //   return from(this.compressAndPrepareFormData(file, userId)).pipe(
  //     // `switchMap` takes the FormData from our async function and uses it to make the HTTP request
  //     switchMap(formData => {
  //       const token = localStorage.getItem('authToken');
  //       const headers = new HttpHeaders({ Authorization: `Bearer ${token}` });

  //       const req = new HttpRequest('POST', `${this.dataApiUrl}/upload`, formData, {
  //         headers,
  //         reportProgress: true,
  //       });

  //       return this.http.request<UploadResult>(req);
  //     })
  //   );
  // }

  // // NEW: Helper async function to handle compression
  // private async compressAndPrepareFormData(file: File, userId: string): Promise<FormData> {
  //   console.log(`Original file size: ${(file.size / 1024 / 1024).toFixed(2)} MB`);

  //   // 1. Read file into buffer
  //   const fileBuffer = await file.arrayBuffer();

  //   // 2. Compress the buffer using pako
  //   console.log("Compressing file in browser...");
  //   const compressedData = pako.gzip(fileBuffer);
  //   console.log(`Compressed size: ${(compressedData.length / 1024 / 1024).toFixed(2)} MB`);

  //   // 3. Create a new Blob from the compressed data
  //   const compressedBlob = new Blob([compressedData]);

  //   // 4. Create FormData with the COMPRESSED blob
  //   const formData = new FormData();
  //   // IMPORTANT: Add .gz to the filename to signal to the server it's compressed
  //   formData.append('file', compressedBlob, file.name + '.gz');
  //   formData.append('userId', userId);

  //   return formData;
  // }

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