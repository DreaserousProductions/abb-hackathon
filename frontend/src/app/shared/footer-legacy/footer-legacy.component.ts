import { Component } from '@angular/core';

@Component({
  selector: 'app-footer-legacy',
  standalone: true,
  imports: [],
  templateUrl: './footer-legacy.component.html',
  styleUrl: './footer-legacy.component.css'
})
export class FooterLegacyComponent {
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
        console.log('Link copied to clipboard. You can now share it to anyone!');
      }).catch(() => {
        console.log('Could not copy link. Please copy it manually.');
      });
    }
  }
}
