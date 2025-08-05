import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { environment } from '../../../environments/environment.development';

// --- INTERFACES FOR SIMULATION DATA ---
export interface SimulationResult {
  rowIndex: number;
  prediction: number;
  confidence: number;
  actual: number;
  isCorrect: boolean;
  timestamp: string;
}

export interface SimulationStatus {
  status?: 'stopped' | 'finished' | 'stopping';
  error?: string;
}

export type SimulationMessage = SimulationResult | SimulationStatus;

@Injectable({
  providedIn: 'root'
})
export class SimulationService {
  private socket$?: WebSocketSubject<any>;
  public messages$: Subject<SimulationMessage> = new Subject<SimulationMessage>();

  public connect(): void {
    if (!this.socket$ || this.socket$.closed) {
      const wsUrl = environment.apiUrl.replace(/^http/, 'ws');
      this.socket$ = webSocket(`${wsUrl}/Model/simulation-ws`);

      this.socket$.subscribe(
        // A message was received from the server.
        (msg) => this.messages$.next(msg),
        // An error occurred.
        (err) => this.messages$.error(err),
        // The connection was closed.
        () => this.messages$.complete()
      );
    }
  }

  public startSimulation(userId: string, datasetId: string): void {
    if (!this.socket$) {
      this.connect();
    }
    const startMessage = { action: 'start', userId, datasetId };
    this.socket$?.next(startMessage);
    console.log('Sent "start" message to WebSocket:', startMessage);
  }

  public stopSimulation(userId: string, datasetId: string): void {
    if (this.socket$) {
      const stopMessage = { action: 'stop', userId, datasetId };
      this.socket$.next(stopMessage);
      console.log('Sent "stop" message to WebSocket:', stopMessage);
    }
  }

  public closeConnection(): void {
    this.socket$?.complete(); // Closes the connection
  }
}