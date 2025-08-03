import { Component, Input, OnInit, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-alert',
  standalone: true,
  imports: [],
  templateUrl: './alert.component.html',
  styleUrls: ['./alert.component.css']
})
export class AlertComponent implements OnInit {
  @Input() message: string = "Message";
  @Output() dismiss = new EventEmitter<void>();
  die = false;

  ngOnInit(): void {
    setTimeout(() => {
      this.die = true;

      setTimeout(() => {
        this.dismiss.emit();
      }, 1200);

    }, 3500);
  }
}