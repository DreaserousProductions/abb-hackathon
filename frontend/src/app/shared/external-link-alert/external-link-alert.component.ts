import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'app-external-link-alert',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './external-link-alert.component.html',
  styleUrl: './external-link-alert.component.css'
})
export class ExternalLinkAlertComponent {
  @Input() url: string = '';
  @Input() visible: boolean = false;
  @Output() onClose = new EventEmitter<void>();

  openLink() {
    window.open(this.url, '_blank');
    this.close();
  }

  close() {
    this.onClose.emit();
  }
}
