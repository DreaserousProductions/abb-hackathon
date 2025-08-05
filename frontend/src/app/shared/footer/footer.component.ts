import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, OnInit } from '@angular/core';
import { AlertComponent } from "../alert/alert.component";

@Component({
  selector: 'app-footer',
  standalone: true,
  imports: [CommonModule, AlertComponent],
  templateUrl: './footer.component.html',
  styleUrl: './footer.component.css'
})
export class FooterComponent implements AfterViewInit {
  ngAfterViewInit(): void {
    document.addEventListener("DOMContentLoaded", () => {
      const footer = document.querySelector(".footer");

      if (footer) {
        const footerRect = footer.getBoundingClientRect();
        for (let i = 0; i < 30; i++) {
          const star = document.createElement('div');
          const x = Math.floor(Math.random() * footer.clientWidth);
          const y = Math.floor(Math.random() * (footer.clientHeight / 2));

          star.classList.add("star");
          star.style.position = 'absolute';
          star.style.left = `${x}px`;
          star.style.top = `${y + 10}px`;

          footer.insertBefore(star, footer.firstChild);
        }
      }
    });
  }

  alerts: { id: number, message: string }[] = [];
  private alertIdCounter = 0;

  showCustomAlert(message: string) {
    this.alerts.push({ id: this.alertIdCounter++, message });
  }

  removeAlert(id: number) {
    this.alerts = this.alerts.filter(alert => alert.id !== id);
  }

  share() {
    const shareUrl = 'https://abb.dreaserous.tech/';

    if (navigator.share) {
      navigator.share({
        title: document.title,
        text: 'Check out this Tech Startup!',
        url: shareUrl
      }).then(() => console.log('Successful share'))
        .catch((error) => console.error('Error sharing:', error));
    } else {
      navigator.clipboard.writeText(shareUrl).then(() => {
        this.showCustomAlert('Link copied to clipboard. You can now share it to anyone!');
      }).catch(() => {
        this.showCustomAlert('Could not copy link. Please copy it manually.');
      });
    }
  }
}
